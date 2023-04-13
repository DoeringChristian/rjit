use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::{Context, DeviceBuffer, Module};
use cust::stream::{Stream, StreamFlags};
use cust::util::SliceExt;

use crate::backend;
use crate::schedule::{SVarId, ScheduleIr};
use crate::var::*;
use std::fmt::{Debug, Write};
use std::sync::Arc;

#[derive(Debug)]
pub struct Backend {
    ctx: Arc<Context>,
    stream: Option<cust::stream::Stream>,
}
impl Backend {
    pub fn new() -> Self {
        Self {
            ctx: Arc::new(cust::quick_init().unwrap()),
            stream: Some(Stream::new(StreamFlags::NON_BLOCKING, None).unwrap()),
        }
    }
}

impl backend::Backend for Backend {
    fn new_kernel(&self) -> Box<dyn backend::Kernel> {
        Box::new(CUDAKernel {
            ctx: self.ctx.clone(),
            asm: Default::default(),
            module: Default::default(),
        })
    }
    fn first_register(&self) -> usize {
        CUDAKernel::FIRST_REGISTER
    }

    fn buffer_uninit(&self, size: usize) -> Arc<dyn backend::Buffer> {
        unsafe { cust::sys::cuCtxSetCurrent(self.ctx.as_raw()) };
        Arc::new(Buffer {
            buffer: vec![0u8; size].as_slice().as_dbuf().unwrap(),
            ctx: self.ctx.clone(),
        })
    }

    fn buffer_from_slice(&self, slice: &[u8]) -> Arc<dyn backend::Buffer> {
        unsafe { cust::sys::cuCtxSetCurrent(self.ctx.as_raw()) };
        Arc::new(Buffer {
            buffer: slice.as_dbuf().unwrap(),
            ctx: self.ctx.clone(),
        })
    }
    fn synchronize(&self) {
        unsafe { cust::sys::cuCtxSetCurrent(self.ctx.as_raw()) };
        self.stream.as_ref().unwrap().synchronize().unwrap();
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        self.stream.take();
    }
}

#[derive(Debug)]
pub struct Buffer {
    buffer: DeviceBuffer<u8>,
    ctx: Arc<Context>,
}
impl backend::Buffer for Buffer {
    fn as_ptr(&self) -> u64 {
        self.buffer.as_device_ptr().as_raw()
    }
    fn as_vec(&self) -> Vec<u8> {
        self.buffer.as_host_vec().unwrap()
    }
}

pub struct Texture {
    ctx: Arc<Context>,
}

pub struct CUDAKernel {
    ctx: Arc<Context>,
    pub asm: String,
    pub module: Option<Module>,
}

impl Debug for CUDAKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CUDAKernel")
            .field("asm", &self.asm)
            .finish()
    }
}

impl backend::Kernel for CUDAKernel {
    fn execute_async(&mut self, ir: &mut ScheduleIr) {
        unsafe { cust::sys::cuCtxSetCurrent(self.ctx.as_raw()) };

        let func = self
            .module
            .as_ref()
            .expect("Need to compile Kernel before we can execute it!")
            .get_function(Self::ENTRY_POINT)
            .unwrap();

        let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();

        let grid_size = (ir.size() as u32 + block_size - 1) / block_size;

        let stream =
            cust::stream::Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None).unwrap();

        let mut params = vec![ir.size() as u64];
        params.extend(ir.buffers().iter().map(|b| b.as_ptr()));

        unsafe {
            stream
                .launch(
                    &func,
                    grid_size,
                    block_size,
                    0,
                    &[params.as_mut_ptr() as *mut std::ffi::c_void],
                )
                .unwrap();
        }

        // stream.synchronize().unwrap();
    }
    #[allow(unused_must_use)]
    fn assemble(&mut self, ir: &ScheduleIr) {
        self.asm.clear();
        let n_params = 1 + ir.buffers().len(); // Add 1 for size
        let n_regs = ir.n_regs();

        /* Special registers:
             %r0   :  Index
             %r1   :  Step
             %r2   :  Size
             %p0   :  Stopping predicate
             %rd0  :  Temporary for parameter pointers
             %rd1  :  Pointer to parameter table in global memory if too big
             %b3, %w3, %r3, %rd3, %f3, %d3, %p3: reserved for use in compound
             statements that must write a temporary result to a register.
        */

        writeln!(self.asm, ".version {}.{}", 8, 0);
        writeln!(self.asm, ".target {}", "sm_86");
        writeln!(self.asm, ".address_size 64");

        writeln!(self.asm, "");

        writeln!(self.asm, ".entry {}(", Self::ENTRY_POINT);
        writeln!(
            self.asm,
            "\t.param .align 8 .b8 params[{}]) {{",
            ((n_params + 1) * std::mem::size_of::<u64>())
        );
        writeln!(self.asm, "");

        writeln!(
            self.asm,
            "\t.reg.b8   %b <{n_regs}>; .reg.b16 %w<{n_regs}>; .reg.b32 %r<{n_regs}>;"
        );
        writeln!(
            self.asm,
            "\t.reg.b64  %rd<{n_regs}>; .reg.f32 %f<{n_regs}>; .reg.f64 %d<{n_regs}>;"
        );
        writeln!(self.asm, "\t.reg.pred %p <{n_regs}>;");
        writeln!(self.asm, "");

        write!(
            self.asm,
            "\tmov.u32 %r0, %ctaid.x;\n\
             \tmov.u32 %r1, %ntid.x;\n\
             \tmov.u32 %r2, %tid.x;\n\
             \tmad.lo.u32 %r0, %r0, %r1, %r2; // r0 <- Index\n"
        );

        writeln!(self.asm, "");

        writeln!(
            self.asm,
            "\t// Index Conditional (jump to done if Index >= Size)."
        );
        writeln!(
            self.asm,
            "\tld.param.u32 %r2, [params]; // r2 <- params[0] (Size)"
        );

        write!(
            self.asm,
            "\tsetp.ge.u32 %p0, %r0, %r2; // p0 <- r0 >= r2\n\
            \t@%p0 bra done; // if p0 => done\n\
            \t\n\
            \tmov.u32 %r3, %nctaid.x; // r3 <- nctaid.x\n\
            \tmul.lo.u32 %r1, %r3, %r1; // r1 <- r3 * r1\n\
            \t\n"
        );

        write!(self.asm, "body: // sm_{}\n", 86); // TODO: compute capability from device

        for id in ir.ids() {
            let var = ir.var(id);
            match var.param_ty {
                ParamType::None => self.assemble_var(ir, id),
                ParamType::Input => {
                    let param_offset = (var.param.unwrap() + 1) * 8;
                    // Load from params
                    writeln!(self.asm, "");
                    writeln!(self.asm, "\t// [{}]: {:?} =>", id, var);
                    if var.is_literal() {
                        writeln!(
                            self.asm,
                            "\tld.param.{} {}, [params+{}];",
                            var.ty.name_cuda(),
                            var.reg(),
                            param_offset
                        );
                        continue;
                    } else {
                        writeln!(self.asm, "\tld.param.u64 %rd0, [params+{}];", param_offset);
                    }
                    if var.size > 1 {
                        writeln!(
                            self.asm,
                            "\tmad.wide.u32 %rd0, %r0, {}, %rd0;",
                            var.ty.size()
                        );
                    }

                    if var.ty == VarType::Bool {
                        writeln!(self.asm, "\tld.global.cs.u8 %w0, [%rd0];");
                        writeln!(self.asm, "\tsetp.ne.u16 {}, %w0, 0;", var.reg());
                    } else {
                        writeln!(
                            self.asm,
                            "\tld.global.cs.{} {}, [%rd0];",
                            var.ty.name_cuda(),
                            var.reg(),
                        );
                    }
                }
                ParamType::Output => {
                    let param_offst = (var.param.unwrap() + 1) * 8;
                    self.assemble_var(ir, id);
                    // let offset = param_idx * 8;
                    write!(
                        self.asm,
                        "\n\t// Store:\n\
                        \tld.param.u64 %rd0, [params + {}]; // rd0 <- params[offset]\n\
                            \tmad.wide.u32 %rd0, %r0, {}, %rd0; // rd0 <- Index * ty.size() + \
                            params[offset]\n",
                        param_offst,
                        var.ty.size(),
                    );

                    if var.ty == VarType::Bool {
                        writeln!(self.asm, "\tselp.u16 %w0, 1, 0, {};", var.reg());
                        writeln!(self.asm, "\tst.global.cs.u8 [%rd0], %w0;");
                    } else {
                        writeln!(
                                self.asm,
                                "\tst.global.cs.{} [%rd0], {}; // (Index * ty.size() + params[offset])[Index] <- var",
                                var.ty.name_cuda(),
                                var.reg(),
                            );
                    }
                }
            }
        }

        // End of kernel:

        writeln!(self.asm, "\n\t//End of Kernel:");
        writeln!(
            self.asm,
            "\n\tadd.u32 %r0, %r0, %r1; // r0 <- r0 + r1\n\
            \tsetp.ge.u32 %p0, %r0, %r2; // p0 <- r1 >= r2\n\
            \t@!%p0 bra body; // if p0 => body\n\
            \n"
        );
        writeln!(self.asm, "done:");
        write!(
            self.asm,
            "\n\tret;\n\
        }}\n"
        );

        std::fs::write("/tmp/tmp.ptx", &self.asm).unwrap();

        log::trace!("{}", self.asm);
    }

    fn compile(&mut self) {
        self.module = Some(
            Module::from_ptx(
                &self.asm,
                &[
                    ModuleJitOption::OptLevel(OptLevel::O0),
                    ModuleJitOption::GenenerateDebugInfo(true),
                    ModuleJitOption::GenerateLineInfo(true),
                ],
            )
            .unwrap(),
        );
    }

    fn assembly(&self) -> &str {
        &self.asm
    }
}

impl CUDAKernel {
    const ENTRY_POINT: &str = "cujit";
    const FIRST_REGISTER: usize = 4;
    #[allow(unused_must_use)]
    fn assemble_var(&mut self, ir: &ScheduleIr, id: SVarId) {
        super::codegen::assemble_var(&mut self.asm, ir, id);
    }
}

#[cfg(test)]
mod test {
    use crate::jit::Jit;
    use crate::trace::{ReduceOp, Trace};
    use crate::{jit, trace};

    #[test]
    fn refcounting() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let x = ir.buffer_f32(&[1.; 10]);
        assert_eq!(x.var().rc, 1, "rc of x should be 1 (in x)");
        let y = x.add(&x);
        // let y = ir::add(&x, &x);

        assert_eq!(
            x.var().rc,
            3,
            "rc of x should be 3 (2 refs in y and 1 in x)"
        );

        assert_eq!(y.var().rc, 1, "rc of y should be 1 (in y)");

        ir.schedule(&[&y]);
        let mut jit = Jit::default();

        assert_eq!(
            y.var().rc,
            2,
            "rc of y should be 2 (1 in y and 1 in schedule)"
        );

        jit.eval(&mut ir.borrow_mut());

        assert_eq!(
            x.var().rc,
            1,
            "rc of x should be 1 (dependencies of y shuld be cleaned)"
        );
        assert_eq!(
            y.var().rc,
            1,
            "rc of y should be 2 (y from schedule should be cleaned)"
        );

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(y.to_host_f32(), vec![2f32; 10]);
    }
    #[test]
    fn load_add_f32() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let x = ir.buffer_f32(&[1.; 10]);
        // let y = ir::add(&x, &x);
        let y = x.add(&x);

        ir.schedule(&[&y]);
        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(y.to_host_f32(), vec![2f32; 10]);
    }
    #[test]
    fn load_gather_f32() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let x = ir.buffer_f32(&[1., 2., 3., 4., 5.]);
        let i = ir.buffer_u32(&[0, 1, 4]);
        let y = x.gather(&i, None);

        ir.schedule(&[&y]);
        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(y.to_host_f32(), vec![1., 2., 5.]);
    }
    #[test]
    fn reindex() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let x = ir.index(10);

        let i = ir.index(3);
        let c = ir.literal_u32(2);
        let i = i.add(&c);

        let y = x.gather(&i, None);

        ir.schedule(&[&y]);
        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(y.to_host_u32(), vec![2, 3, 4]);
    }
    #[test]
    fn index() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let i = ir.index(10);

        ir.schedule(&[&i]);
        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(i.to_host_u32(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
    #[test]
    fn gather_eval() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let tmp_x;
        let tmp_y;
        let tmp_z;
        let r = {
            let x = ir.index(3);
            dbg!();
            assert_eq!(x.var().rc, 1);
            let y = ir.buffer_u32(&[1, 2, 3]);
            dbg!();
            assert_eq!(y.var().rc, 1);

            // let z = ir::add(&x, &y);
            let z = x.add(&y);
            dbg!();
            assert_eq!(x.var().rc, 2);
            assert_eq!(y.var().rc, 2);
            assert_eq!(z.var().rc, 1);

            let r = z.gather(&ir.index(3), None);
            dbg!();
            assert_eq!(x.var().rc, 2);
            assert_eq!(y.var().rc, 2);
            assert_eq!(z.var().rc, 3);
            assert_eq!(r.var().rc, 1);
            tmp_x = x.id();
            tmp_y = y.id();
            tmp_z = z.id();
            r
        };
        assert_eq!(r.var().rc, 1);
        assert_eq!(ir.borrow_mut().get_var(tmp_x).unwrap().rc, 1);
        assert_eq!(ir.borrow_mut().get_var(tmp_y).unwrap().rc, 1);
        assert_eq!(ir.borrow_mut().get_var(tmp_z).unwrap().rc, 2); // z is referenced by r and the
                                                                   // schedule

        ir.schedule(&[&r]);
        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        assert_eq!(r.var().rc, 1);
        assert!(ir.borrow_mut().get_var(tmp_z).is_none());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(r.to_host_u32(), vec![1, 3, 5]);
    }
    #[test]
    fn paralell() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let x = ir.index(10);

        let y = ir.index(3);

        ir.schedule(&[&x, &y]);
        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(x.to_host_u32(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(y.to_host_u32(), vec![0, 1, 2]);
    }
    #[test]
    fn load_gather() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let x = ir.buffer_f32(&[1., 2., 3.]);

        ir.schedule(&[&x]);

        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(x.to_host_f32(), vec![1., 2., 3.]);
    }
    #[test]
    fn eval_scatter() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let x = ir.buffer_u32(&[0, 0, 0, 0]);
        let c = ir.literal_u32(1);
        let x = x.add(&c); // x: [1, 1, 1, 1]

        let i = ir.index(3);
        let c = ir.literal_u32(1);
        let i = i.add(&c); // i: [1, 2, 3]

        let y = ir.literal_u32(2);

        y.scatter(&x, &i, None); // x: [1, 2, 2, 2]

        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(x.to_host_u32(), vec![1, 2, 2, 2]);
    }
    #[test]
    fn scatter_twice() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let x = ir.buffer_u32(&[0, 0, 0, 0]);

        let i = ir.index(3);
        let c = ir.literal_u32(1);
        let i = i.add(&c);

        let y = ir.literal_u32(2);

        y.scatter(&x, &i, None);

        let i = ir.index(2);

        let y = ir.literal_u32(3);

        y.scatter(&x, &i, None);

        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(x.to_host_u32(), vec![3, 3, 2, 2]);
    }
    #[test]
    fn scatter_twice_add() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let x = ir.buffer_u32(&[0, 0, 0, 0]);

        let i = ir.index(3);
        let c = ir.literal_u32(1);
        let i = i.add(&c);

        let y = ir.literal_u32(2);

        y.scatter(&x, &i, None);

        let i = ir.index(2);

        let y = ir.literal_u32(3);

        y.scatter(&x, &i, None);

        let c = ir.literal_u32(1);
        let x = x.add(&c);

        ir.schedule(&[&x]);

        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(x.to_host_u32(), vec![4, 4, 3, 3]);
    }
    #[test]
    fn scatter_reduce() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        let x = ir.buffer_u32(&[0, 0, 0, 0]);

        let i = ir.buffer_u32(&[0, 0, 0]);

        let y = ir.literal_u32(1);

        y.scatter_reduce(&x, &i, None, ReduceOp::Add);

        ir.schedule(&[&x]);

        let mut jit = Jit::default();
        jit.eval(&mut ir.borrow_mut());

        insta::assert_snapshot!(jit.kernel_debug());

        assert_eq!(x.to_host_u32(), vec![3, 0, 0, 0]);
    }
}

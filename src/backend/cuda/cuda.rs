use std::ffi::{c_void, CStr, CString};
use std::fmt::{Debug, Write};
use std::sync::Arc;
use thiserror::Error;

use super::cuda_core::{Device, Instance};
use crate::backend;
use crate::schedule::{SVarId, ScheduleIr};
use crate::trace::VarType;
use crate::var::ParamType;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    CoreError(#[from] super::cuda_core::Error),
}

#[derive(Debug)]
pub struct Backend {
    instance: Arc<Instance>,
    device: Device,
}
impl Backend {
    pub fn new() -> Result<Self, Error> {
        let instance = Arc::new(Instance::new()?);
        let device = Device::create(&instance, 0)?;
        Ok(Self { instance, device })
    }
}

impl backend::Backend for Backend {
    fn new_kernel(&self) -> Box<dyn backend::Kernel> {
        Box::new(Kernel {
            asm: Default::default(),
            data: Default::default(),
            module: std::ptr::null_mut(),
            func: std::ptr::null_mut(),
            device: self.device.clone(),
        })
    }

    fn buffer_uninit(&self, size: usize) -> Arc<dyn crate::backend::Buffer> {
        unsafe {
            let ctx = self.device.ctx();
            let mut dptr = 0;
            ctx.cuMemAlloc_v2(&mut dptr, size).check().unwrap();
            Arc::new(Buffer {
                device: self.device.clone(),
                dptr,
                size,
            })
        }
    }

    fn buffer_from_slice(&self, slice: &[u8]) -> Arc<dyn crate::backend::Buffer> {
        unsafe {
            let size = slice.len();

            let ctx = self.device.ctx();

            let mut dptr = 0;
            ctx.cuMemAlloc_v2(&mut dptr, size).check().unwrap();
            ctx.cuMemcpyHtoD_v2(dptr, slice.as_ptr() as _, size)
                .check()
                .unwrap();
            Arc::new(Buffer {
                device: self.device.clone(),
                dptr,
                size,
            })
        }
    }
    fn synchronize(&self) {
        // todo!()
    }

    fn first_register(&self) -> usize {
        Kernel::FIRST_REGISTER
    }
}

impl Drop for Backend {
    fn drop(&mut self) {}
}

#[derive(Debug)]
pub struct Buffer {
    device: Device,
    dptr: u64,
    size: usize,
}
impl backend::Buffer for Buffer {
    fn as_ptr(&self) -> u64 {
        self.dptr
    }

    fn copy_to_host(&self, dst: &mut [u8]) {
        unsafe {
            let ctx = self.device.ctx();
            assert!(dst.len() <= self.size);

            ctx.cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut _, self.dptr, self.size)
                .check()
                .unwrap();
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.ctx().cuMemFree_v2(self.dptr).check().unwrap();
        }
    }
}

pub struct Texture {}

#[derive(Debug)]
pub struct Kernel {
    pub asm: String,
    pub data: Vec<u8>,
    pub module: cuda_rs::CUmodule,
    pub func: cuda_rs::CUfunction,
    device: Device,
}

impl Kernel {
    const ENTRY_POINT: &str = "cujit";
    const FIRST_REGISTER: usize = 4;
    #[allow(unused_must_use)]
    fn assemble_var(&mut self, ir: &ScheduleIr, id: SVarId) {
        super::codegen::assemble_var(&mut self.asm, ir, id);
    }
}

impl backend::Kernel for Kernel {
    fn execute_async(&mut self, ir: &mut crate::schedule::ScheduleIr) {
        let ctx = self.device.ctx();
        unsafe {
            let mut unused = 0;
            let mut block_size = 0;
            ctx.cuOccupancyMaxPotentialBlockSize(
                &mut unused,
                &mut block_size,
                self.func,
                None,
                0,
                0,
            )
            .check()
            .unwrap();
            let block_size = block_size as u32;

            let grid_size = (ir.size() as u32 + block_size - 1) / block_size;

            let stream = self
                .device
                .create_stream(cuda_rs::CUstream_flags_enum::CU_STREAM_DEFAULT)
                .unwrap();
            // let mut stream = std::ptr::null_mut();
            // ctx.cuStreamCreate(
            //     &mut stream,
            //     cuda_rs::CUstream_flags_enum::CU_STREAM_DEFAULT as _,
            // )
            // .check()
            // .unwrap();

            let mut params = vec![ir.size() as u64];
            params.extend(ir.buffers().iter().map(|b| b.as_ptr()));

            ctx.cuLaunchKernel(
                self.func,
                grid_size,
                1,
                1,
                block_size as _,
                1,
                1,
                0,
                *stream,
                [params.as_mut_ptr() as *mut std::ffi::c_void].as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .check()
            .unwrap();

            stream.synchronize().unwrap();
            // ctx.cuStreamSynchronize(stream).check().unwrap();
            // ctx.cuStreamDestroy_v2(stream).check().unwrap();
        }
    }
    fn compile(&mut self) {
        let ctx = self.device.ctx();
        unsafe {
            let asm = CString::new(self.asm.as_str()).unwrap();

            const log_size: usize = 16384;
            let mut error_log = [0u8; log_size];
            let mut info_log = [0u8; log_size];

            let mut options = [
                cuda_rs::CUjit_option_enum::CU_JIT_OPTIMIZATION_LEVEL,
                cuda_rs::CUjit_option_enum::CU_JIT_LOG_VERBOSE,
                cuda_rs::CUjit_option_enum::CU_JIT_INFO_LOG_BUFFER,
                cuda_rs::CUjit_option_enum::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                cuda_rs::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER,
                cuda_rs::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                cuda_rs::CUjit_option_enum::CU_JIT_GENERATE_LINE_INFO,
                cuda_rs::CUjit_option_enum::CU_JIT_GENERATE_DEBUG_INFO,
            ];

            let mut option_values = [
                4 as *mut c_void,
                1 as *mut c_void,
                info_log.as_mut_ptr() as *mut c_void,
                log_size as *mut c_void,
                error_log.as_mut_ptr() as *mut c_void,
                log_size as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
            ];

            let mut linkstate = std::ptr::null_mut();
            ctx.cuLinkCreate_v2(
                options.len() as _,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
                &mut linkstate,
            )
            .check()
            .unwrap();

            ctx.cuLinkAddData_v2(
                linkstate,
                cuda_rs::CUjitInputType::CU_JIT_INPUT_PTX,
                asm.as_ptr() as *mut c_void,
                self.asm.len(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
            .check()
            .or_else(|err| {
                let error_log = CStr::from_bytes_until_nul(&error_log).unwrap().to_str().unwrap();
                log::error!("Compilation failed. Please see the PTX listing and error message below:\n{}\n{}", error_log, err);
                Err(err)
            }).unwrap();

            let mut link_out = std::ptr::null_mut();
            let mut link_out_size = 0;
            ctx.cuLinkComplete(linkstate, &mut link_out, &mut link_out_size)
                .check()
                .or_else(|err| {
                    let error_log = CStr::from_bytes_until_nul(&error_log).unwrap().to_str().unwrap();
                    log::error!("Compilation failed. Please see the PTX listing and error message below:\n{}\n{}", error_log, err);
                    Err(err)
                }).unwrap();

            log::trace!(
                "Detailed linker output: {}",
                CStr::from_bytes_until_nul(&info_log)
                    .unwrap()
                    .to_str()
                    .unwrap()
            );

            let mut out: Vec<u8> = Vec::with_capacity(link_out_size);
            std::ptr::copy_nonoverlapping(link_out as *mut u8, out.as_mut_ptr(), link_out_size);
            out.set_len(link_out_size);

            ctx.cuLinkDestroy(linkstate).check().unwrap();

            self.data = out;
        }
        unsafe {
            let mut module = std::ptr::null_mut();
            ctx.cuModuleLoadData(&mut module, self.data.as_ptr() as *const c_void)
                .check()
                .unwrap();
            self.module = module;

            let fname = CString::new(Self::ENTRY_POINT).unwrap();
            let mut func = std::ptr::null_mut();
            ctx.cuModuleGetFunction(&mut func, self.module, fname.as_ptr() as *const i8)
                .check()
                .unwrap();
            self.func = func;
        }
    }

    fn assembly(&self) -> &str {
        &self.asm
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
}

#[cfg(test)]
mod test {
    use crate::jit::Jit;
    use crate::trace::{ReduceOp, Trace};

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

use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::Module;

use crate::ir::*;
use std::fmt::Write;

#[derive(Default)]
pub struct CUDACompiler {
    pub asm: String,
}

impl CUDACompiler {
    const ENTRY_POINT: &str = "cujit";
    pub fn module(&self) -> Module {
        let module = Module::from_ptx(
            &self.asm,
            &[
                ModuleJitOption::OptLevel(OptLevel::O0),
                ModuleJitOption::GenenerateDebugInfo(true),
                ModuleJitOption::GenerateLineInfo(true),
            ],
        )
        .unwrap();
        module
    }
    pub fn execute(&self, ir: &mut Ir) {
        let module = self.module();
        let func = module.get_function(Self::ENTRY_POINT).unwrap();

        let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();

        let grid_size = (ir.size() as u32 + block_size - 1) / block_size;

        let stream =
            cust::stream::Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            stream
                .launch(
                    &func,
                    grid_size,
                    block_size,
                    0,
                    &[ir.params.as_mut_ptr() as *mut std::ffi::c_void],
                )
                .unwrap();
        }

        stream.synchronize().unwrap();
    }
    #[allow(unused_must_use)]
    pub fn compile(&mut self, ir: &Ir) {
        let n_params = ir.params.len() as u64;
        let n_regs = ir.n_regs;

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
            (n_params * std::mem::size_of::<u64>() as u64)
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
            self.compile_var(ir, id);
        }

        // End of kernel:

        writeln!(self.asm, "\n\t//End of Kernel:");
        writeln!(
            self.asm,
            "\tadd.u32 %r0, %r0, %r1; // r0 <- r0 + r1\n\
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

        println!("{}", self.asm);

        std::fs::write("/tmp/tmp.ptx", &self.asm).unwrap();
    }

    #[allow(unused_must_use)]
    fn compile_var(&mut self, ir: &Ir, id: VarId) {
        let var = ir.var(id);
        writeln!(self.asm, "");
        writeln!(self.asm, "\t// [{}]: {:?} =>", id, var);

        match var.op {
            Op::Add(lhs, rhs) => {
                let lhs = ir.var(lhs);
                let rhs = ir.var(rhs);
                writeln!(
                    self.asm,
                    "\tadd.{} {}, {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    lhs.reg(),
                    rhs.reg(),
                );
            }
            Op::ConstF32(val) => {
                writeln!(
                    self.asm,
                    "\tmov.{} {}, 0F{:08x};",
                    var.ty.name_cuda(),
                    var.reg(),
                    unsafe { *(&val as *const f32 as *const u32) }
                );
            }
            Op::Load(param_idx) => {
                // Load from params
                writeln!(
                    self.asm,
                    "\tld.param.u64 %rd0, [params+{}];",
                    param_idx.offset()
                );
                writeln!(
                    self.asm,
                    "\tmad.wide.u32 %rd0, %r0, {}, %rd0;",
                    var.ty.size()
                );
                // TODO: boolean loading
                writeln!(
                    self.asm,
                    "\tld.global.cs.{} {}, [%rd0];",
                    var.ty.name_cuda(),
                    var.reg(),
                );
            }
            Op::LoadLiteral(param_idx) => {
                // let offset = param_idx * 8;
                writeln!(
                    self.asm,
                    "\tld.param.{} {}, [params+{}];",
                    var.ty.name_cuda(),
                    var.reg(),
                    param_idx.offset()
                );
            }
            Op::Store(src, param_idx) => {
                // let offset = param_idx * 8;
                // dbg!(offset);
                write!(
                    self.asm,
                    "\tld.param.u64 %rd0, [params + {}]; // rd0 <- params[offset]\n\
                    \tmad.wide.u32 %rd0, %r0, {}, %rd0; // rd0 <- Index * ty.size() + \
                    params[offset]\n",
                    param_idx.offset(),
                    var.ty.size(),
                );
                // TODO: boolean storage

                writeln!(
                    self.asm,
                    "\tst.global.cs.{} [%rd0], {}; // (Index * ty.size() + params[offset])[Index] <- var",
                    var.ty.name_cuda(),
                    ir.var(src).reg(),
                );
            }
        }
    }
}

#[cfg(test)]
mod test {
    use cust::util::SliceExt;

    use crate::ir::*;

    use super::CUDACompiler;

    #[test]
    fn load_add_store() {
        let ctx = cust::quick_init().unwrap();
        let device = cust::device::Device::get_device(0).unwrap();
        let mut ir = Ir::default();

        let size = 10;
        ir.set_size(size as _);

        let x_buf = vec![1f32; size].as_slice().as_dbuf().unwrap();
        let param_id = ir.push_param(x_buf.as_device_ptr().as_raw());

        let x = ir.push_var(Var {
            op: Op::Load(param_id),
            ty: VarType::F32,
            reg: 0,
        });
        let c = ir.push_var(Var {
            op: Op::ConstF32(2.),
            ty: VarType::F32,
            reg: 0,
        });
        let y = ir.push_var(Var {
            op: Op::Add(x, x),
            ty: VarType::F32,
            reg: 0,
        });
        let z = ir.push_var(Var {
            op: Op::Store(y, param_id),
            ty: VarType::F32,
            reg: 0,
        });

        let mut compiler = CUDACompiler::default();
        compiler.compile(&ir);
        compiler.execute(&mut ir);

        assert_eq!(&x_buf.as_host_vec().unwrap(), &[2.; 10]);
    }
}

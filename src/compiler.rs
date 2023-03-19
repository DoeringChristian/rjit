use crate::ir::*;
use std::fmt::Write;

#[derive(Default)]
pub struct Compiler {
    pub asm: String,
}

impl Compiler {
    #[allow(unused_must_use)]
    pub fn compile(&mut self, ir: &Ir) {
        let n_params = 10;
        let n_regs = 10;

        writeln!(self.asm, ".version {}.{}", 3, 0);
        writeln!(self.asm, ".target {}", "sm_21");
        writeln!(self.asm, ".address_size 64");

        writeln!(self.asm, "");

        writeln!(self.asm, ".entry cujit(");
        writeln!(self.asm, ".param .align 8 .b8 params[{}]) {{", n_params);

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
             \tmad.lo.u32 %r0, %r0, %r1, %r2;\n"
        );

        writeln!(self.asm, "");

        writeln!(self.asm, "\tld.param.u32 %r2, [params];");

        write!(
            self.asm,
            "\tsetp.ge.u32 %p0, %r0, %r2;\n\
            \t@%p0 bra done;\n\
            \t\n\
            \tmov.u32 %r3, %nctaid.x;\n\
            \tmul.lo.u32 %r1, %r3, %r1;\n\
            \t\n"
        );

        write!(self.asm, "body: // sm_{}\n", 86); // TODO: compute capability from device

        for id in ir.ids() {
            self.compile_var(ir, id);
        }

        // End of kernel:

        writeln!(
            self.asm,
            "\n\
            \tadd.u32 %r0, %r0, %r1;\n\
            \tsetp.ge.u32 %p0, %r0, %r2;\n\
            \t@!%p0 bra body;\n\
            \n\
            done:\n"
        );
        write!(
            self.asm,
            "\tret;\n\
        }}\n"
        );

        println!("{}", self.asm);
    }

    #[allow(unused_must_use)]
    fn compile_var(&mut self, ir: &Ir, id: VarId) {
        let var = ir.var(id);
        match var.op {
            Op::Add(lhs, rhs) => {
                writeln!(
                    self.asm,
                    "\tadd.{} {}, {}, {};",
                    var.ty.name_cuda(),
                    PVar(id, var),
                    ir.pvar(lhs),
                    ir.pvar(rhs),
                );
            }
            Op::ConstF32(val) => {
                writeln!(
                    self.asm,
                    "\tmov.{} {}, 0F{:08x};",
                    var.ty.name_cuda(),
                    PVar(id, var),
                    unsafe { *(&val as *const f32 as *const u32) }
                );
            }
        }
    }
}

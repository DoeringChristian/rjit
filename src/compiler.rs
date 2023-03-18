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

        let mut asm = String::new();
        writeln!(asm, ".version {}.{}", 3, 0);
        writeln!(asm, ".target {}", "sm_21");
        writeln!(asm, ".address_size 64");

        writeln!(asm, "");

        writeln!(asm, ".entry cujit(");
        writeln!(asm, ".param .align 8 .b8 params[{}]) {{", n_params);

        writeln!(
            asm,
            "\t.reg.b8   %b <{n_regs}>; .reg.b16 %w<{n_regs}>; .reg.b32 %r<{n_regs}>;"
        );
        writeln!(
            asm,
            "\t.reg.b64  %rd<{n_regs}>; .reg.f32 %f<{n_regs}>; .reg.f64 %d<{n_regs}>;"
        );
        writeln!(asm, "\t.reg.pred %p <{n_regs}>;");
        writeln!(asm, "");

        write!(
            asm,
            "\tmov.u32 %r0, %ctaid.x;\n\
             \tmov.u32 %r1, %ntid.x;\n\
             \tmov.u32 %r2, %tid.x;\n\
             \tmad.lo.u32 %r0, %r0, %r1, %r2;\n"
        );

        writeln!(asm, "");

        writeln!(asm, "\tld.param.u32 %r2, [params];");

        write!(
            asm,
            "\tsetp.ge.u32 %p0, %r0, %r2;\n\
            \t@%p0 bra done;\n\
            \t\n\
            \tmov.u32 %r3, %nctaid.x;\n\
            \tmul.lo.u32 %r1, %r3, %r1;\n\
            \t\n"
        );

        write!(asm, "body: // sm_{}\n", 86); // TODO: compute capability from device

        // End of kernel:

        writeln!(
            asm,
            "\n\
            \tadd.u32 %r0, %r0, %r1;\n\
            \tsetp.ge.u32 %p0, %r0, %r2;\n\
            \t@!%p0 bra body;\n\
            \n\
            done:\n"
        );
        write!(
            asm,
            "\tret;\n\
        }}\n"
        );

        println!("{}", asm);
        self.asm = asm;
    }
}

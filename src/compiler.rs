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

        writeln!(self.asm, ".entry cujit(");
        writeln!(self.asm, "\t.param .align 8 .b8 params[{}]) {{", n_params);
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

        std::fs::write("/tmp/tmp.ptx", &self.asm).unwrap();
    }

    #[allow(unused_must_use)]
    fn compile_var(&mut self, ir: &Ir, id: VarId) {
        let var = ir.var(id);
        writeln!(self.asm, "");
        writeln!(self.asm, "\t// {:?} =>", var);

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
            Op::Load(param_idx) => {
                let offset = param_idx * 8;
                // Load from params
                writeln!(
                    self.asm,
                    "\tld.{}.u64 %rd0, [{}+{}];",
                    "param", "params", offset
                ); // rd0 = params[offset] // with typeof(rd0) = (void *)
                writeln!(
                    self.asm,
                    "\tmad.wide.u32 %rd0, %r0, {}, %rd0;",
                    var.ty.size()
                ); // rd0 = r0 * ty.size() + rd0 <=> rd0 = index * ty.size() + params[offset]
                   // TODO: boolean loading
                writeln!(
                    self.asm,
                    "\tld.global.cs.{} {}, [%rd0];",
                    var.ty.name_cuda(),
                    PVar(id, var)
                );
            }
            Op::Store(param_idx) => {
                let offset = param_idx * 8;
                write!(
                    self.asm,
                    "\tld.{}.u64 %rd0, [{}+{}];\n\
                    \tmad.wide.u32 %rd0, %r0, {}, %rd0;\n",
                    "param",
                    "params",
                    offset,
                    var.ty.size(),
                ); // rd0 = params[offset]; rd0 = r0 * ty.size() + rd0 <=>
                   // rd0 = index * ty.size() + params[offset];
                   // with typeof(rd0) = (void *);

                // TODO: boolean storage

                writeln!(
                    self.asm,
                    "\tst.global.cs.{} [%rd0], {};",
                    var.ty.name_cuda(),
                    PVar(id, var)
                );
            }
        }
    }
}

use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::Module;

use crate::ir::*;
use std::collections::HashSet;
use std::fmt::Write;

#[derive(Default)]
pub struct CUDACompiler {
    pub asm: String,
    pub params: Vec<u64>,
    pub n_regs: usize,
    pub schedule_group: Vec<VarId>,
}

impl CUDACompiler {
    const ENTRY_POINT: &str = "cujit";
    const FIRST_REGISTER: usize = 4;
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
    fn size(&self) -> usize {
        self.params[0] as _
    }
    pub fn execute(&mut self, ir: &mut Ir) {
        let module = self.module();
        let func = module.get_function(Self::ENTRY_POINT).unwrap();

        let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();

        let grid_size = (self.size() as u32 + block_size - 1) / block_size;

        let stream =
            cust::stream::Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            stream
                .launch(
                    &func,
                    grid_size,
                    block_size,
                    0,
                    &[self.params.as_mut_ptr() as *mut std::ffi::c_void],
                )
                .unwrap();
        }

        stream.synchronize().unwrap();
    }
    pub fn preprocess(&mut self, ir: &mut Ir) {
        // Get all variables in schedule (schedule_group)
        self.schedule_group.clear();
        ir.deps(&ir.schedule)
            .for_each(|id| self.schedule_group.push(id));

        // TODO: calculate size
        self.n_regs = Self::FIRST_REGISTER;
        for id in self.schedule_group.iter() {
            // Set registers for variables
            let mut var = ir.var_mut(*id);
            var.reg = self.n_regs;
            self.n_regs += 1;

            // Push buffer pointer to params and set param_offset
            if let Some(buffer) = &var.buffer {
                let offset = self.params.len();
                self.params.push(buffer.as_device_ptr().as_raw());
                var.param_offset = offset as _;
            }
        }
        for id in ir.schedule.iter() {}
    }
    #[allow(unused_must_use)]
    pub fn compile(&mut self, ir: &Ir) {
        let n_params = self.params.len() as u64;
        let n_regs = self.n_regs;

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

        for id in self.schedule_group.clone().into_iter() {
            let var = ir.var(id);
            match var.param_ty {
                ParamType::None => self.compile_var(ir, id),
                ParamType::Literal => {
                    // let offset = param_idx * 8;
                    writeln!(
                        self.asm,
                        "\tld.param.{} {}, [params+{}];",
                        var.ty.name_cuda(),
                        var.reg(),
                        var.param_offset * 8,
                    );
                }
                ParamType::Input => {
                    // Load from params
                    writeln!(
                        self.asm,
                        "\tld.param.u64 %rd0, [params+{}];",
                        var.param_offset * 8
                    );
                    writeln!(
                        self.asm,
                        "\tmad.wide.u32 %rd0, %r0, {}, %rd0;",
                        var.ty.size()
                    );
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
                    self.compile_var(ir, id);
                    // let offset = param_idx * 8;
                    // dbg!(offset);
                    write!(
                        self.asm,
                        "\tld.param.u64 %rd0, [params + {}]; // rd0 <- params[offset]\n\
                            \tmad.wide.u32 %rd0, %r0, {}, %rd0; // rd0 <- Index * ty.size() + \
                            params[offset]\n",
                        var.param_offset * 8,
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
            // Op::Data => {}
            Op::Nop => {}
            Op::Neg => {
                if var.ty.is_uint() {
                    writeln!(
                        self.asm,
                        "\tneg.s{} {}, {};",
                        var.ty.size() * 8,
                        var.reg(),
                        ir.reg(var.deps[0])
                    );
                } else {
                    writeln!(
                        self.asm,
                        "\tneg.{} {}, {};\n",
                        var.ty.name_cuda(),
                        var.reg(),
                        ir.reg(var.deps[0])
                    );
                }
            }
            Op::Not => {
                writeln!(
                    self.asm,
                    "\tnot.{} {}, {};",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            }
            Op::Sqrt => {
                if var.ty.is_single() {
                    writeln!(
                        self.asm,
                        "\tsqrt.approx.ftz.{} {}, {};",
                        var.ty.name_cuda(),
                        var.reg(),
                        ir.reg(var.deps[0])
                    );
                } else {
                    writeln!(
                        self.asm,
                        "\tsqrt.rn.{} {}, {};",
                        var.ty.name_cuda(),
                        var.reg(),
                        ir.reg(var.deps[0])
                    );
                }
            }
            Op::Abs => {
                writeln!(
                    self.asm,
                    "\tabs.{} {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            }
            Op::Add => {
                writeln!(
                    self.asm,
                    "\tadd.{} {}, {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                );
            }
            Op::Sub => {
                writeln!(
                    self.asm,
                    "\tsub.{} {}, {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            }
            Op::Mul => {
                if var.ty.is_single() {
                    writeln!(
                        self.asm,
                        "\tmul.ftz.{} {}, {}, {};",
                        var.ty.name_cuda(),
                        var.reg(),
                        ir.reg(var.deps[0]),
                        ir.reg(var.deps[1])
                    );
                } else if var.ty.is_double() {
                    writeln!(
                        self.asm,
                        "\tmul.{} {}, {}, {};",
                        var.ty.name_cuda(),
                        var.reg(),
                        ir.reg(var.deps[0]),
                        ir.reg(var.deps[1])
                    );
                } else {
                    writeln!(
                        self.asm,
                        "\tmul.lo.{} {}, {}, {};",
                        var.ty.name_cuda(),
                        var.reg(),
                        ir.reg(var.deps[0]),
                        ir.reg(var.deps[1])
                    );
                }
            }
            Op::Div => {
                if var.ty.is_single() {
                    writeln!(
                        self.asm,
                        "\tdiv.approx.ftz.{} {}, {}, {};",
                        var.ty.name_cuda(),
                        var.reg(),
                        ir.reg(var.deps[0]),
                        ir.reg(var.deps[1])
                    );
                } else if var.ty.is_double() {
                    writeln!(
                        self.asm,
                        "\tdiv.rn.{} {}, {}, {};",
                        var.ty.name_cuda(),
                        var.reg(),
                        ir.reg(var.deps[0]),
                        ir.reg(var.deps[1])
                    );
                } else {
                    writeln!(
                        self.asm,
                        "\tdiv.{} {}, {}, {};",
                        var.ty.name_cuda(),
                        var.reg(),
                        ir.reg(var.deps[0]),
                        ir.reg(var.deps[1])
                    );
                }
            }
            Op::Mod => {
                writeln!(
                    self.asm,
                    "\trem.{} {}, {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            }
            Op::Mulhi => todo!(),
            Op::Fma => todo!(),
            Op::Min => todo!(),
            Op::Max => todo!(),
            Op::Cail => todo!(),
            Op::Floor => todo!(),
            Op::Round => todo!(),
            Op::Trunc => todo!(),
            Op::Eq => todo!(),
            Op::Neq => todo!(),
            Op::Lt => todo!(),
            Op::Le => todo!(),
            Op::Gt => todo!(),
            Op::Ge => todo!(),
            Op::Select => todo!(),
            Op::Popc => todo!(),
            Op::Clz => todo!(),
            Op::Ctz => todo!(),
            Op::And => todo!(),
            Op::Or => todo!(),
            Op::Xor => todo!(),
            Op::Shl => todo!(),
            Op::Shr => todo!(),
            Op::Rcp => todo!(),
            Op::Rsqrt => todo!(),
            Op::Sin => todo!(),
            Op::Cos => todo!(),
            Op::Exp2 => todo!(),
            Op::Log2 => todo!(),
            Op::Cast => todo!(),
            Op::Bitcast => todo!(),
            Op::Gather => todo!(),
            Op::Scatter => todo!(),
            Op::Idx => todo!(),
            Op::ConstF32(val) => {
                writeln!(
                    self.asm,
                    "\tmov.{} {}, 0F{:08x};",
                    var.ty.name_cuda(),
                    var.reg(),
                    unsafe { *(&val as *const _ as *const u32) }
                );
            }
            Op::ConstU32(val) => {
                writeln!(
                    self.asm,
                    "\tmov.{} {}, 0X{:08x};",
                    var.ty.name_cuda(),
                    var.reg(),
                    unsafe { *(&val as *const _ as *const u32) }
                );
            } // Op::Load(param_idx) => {
              //     // Load from params
              //     writeln!(
              //         self.asm,
              //         "\tld.param.u64 %rd0, [params+{}];",
              //         param_idx.offset()
              //     );
              //     writeln!(
              //         self.asm,
              //         "\tmad.wide.u32 %rd0, %r0, {}, %rd0;",
              //         var.ty.size()
              //     );
              //     if var.ty == VarType::Bool {
              //         writeln!(self.asm, "\tld.global.cs.u8 %w0, [%rd0];");
              //         writeln!(self.asm, "\tsetp.ne.u16 {}, %w0, 0;", var.reg());
              //     } else {
              //         writeln!(
              //             self.asm,
              //             "\tld.global.cs.{} {}, [%rd0];",
              //             var.ty.name_cuda(),
              //             var.reg(),
              //         );
              //     }
              // }
              // Op::LoadLiteral(param_idx) => {
              //     // let offset = param_idx * 8;
              //     writeln!(
              //         self.asm,
              //         "\tld.param.{} {}, [params+{}];",
              //         var.ty.name_cuda(),
              //         var.reg(),
              //         param_idx.offset()
              //     );
              // }
              // Op::Store(src, param_idx) => {
              //     // let offset = param_idx * 8;
              //     // dbg!(offset);
              //     write!(
              //         self.asm,
              //         "\tld.param.u64 %rd0, [params + {}]; // rd0 <- params[offset]\n\
              //         \tmad.wide.u32 %rd0, %r0, {}, %rd0; // rd0 <- Index * ty.size() + \
              //         params[offset]\n",
              //         param_idx.offset(),
              //         var.ty.size(),
              //     );
              //
              //     if var.ty == VarType::Bool {
              //         writeln!(self.asm, "\tselp.u16 %w0, 1, 0, {};", ir.reg(src));
              //         writeln!(self.asm, "\tst.global.cs.u8 [%rd0], %w0;");
              //     } else {
              //         writeln!(
              //             self.asm,
              //             "\tst.global.cs.{} [%rd0], {}; // (Index * ty.size() + params[offset])[Index] <- var",
              //             var.ty.name_cuda(),
              //             ir.reg(src)
              //         );
              //     }
              // }
        }
    }
}

#[cfg(test)]
mod test {}

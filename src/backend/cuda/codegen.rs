use crate::schedule::{SVarId, ScheduleIr};
use crate::var::{Op, ReduceOp};

fn reduce_op_name(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::None => "",
        ReduceOp::Add => ".add",
        ReduceOp::Mul => ".mul",
        ReduceOp::Min => ".min",
        ReduceOp::Max => ".max",
        ReduceOp::And => ".and",
        ReduceOp::Or => ".or",
    }
}

#[allow(warnings)]
pub fn assemble_var(
    asm: &mut impl std::fmt::Write,
    ir: &ScheduleIr,
    id: SVarId,
    buf_offset: usize,
    tex_offset: usize,
) {
    let var = ir.var(id);
    writeln!(asm, "");
    writeln!(asm, "\t// [{}]: {:?} =>", id, var);

    match var.op {
        Op::Nop => {}
        Op::Data => {}
        Op::TexUpload => {}
        Op::Literal => {
            writeln!(
                asm,
                "\tmov.{} {}, 0x{:x};\n",
                var.ty.name_cuda_bin(),
                var.reg(),
                var.literal
            );
        }
        Op::Neg => {
            if var.ty.is_uint() {
                writeln!(
                    asm,
                    "\tneg.s{} {}, {};",
                    var.ty.size() * 8,
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            } else {
                writeln!(
                    asm,
                    "\tneg.{} {}, {};\n",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            }
        }
        Op::Not => {
            writeln!(
                asm,
                "\tnot.{} {}, {};",
                var.ty.name_cuda_bin(),
                var.reg(),
                ir.reg(var.deps[0])
            );
        }
        Op::Sqrt => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\tsqrt.approx.ftz.{} {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            } else {
                writeln!(
                    asm,
                    "\tsqrt.rn.{} {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            }
        }
        Op::Abs => {
            writeln!(
                asm,
                "\tabs.{} {}, {};",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0])
            );
        }
        Op::Add => {
            writeln!(
                asm,
                "\tadd.{} {}, {}, {};",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0]),
                ir.reg(var.deps[1]),
            );
        }
        Op::Sub => {
            writeln!(
                asm,
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
                    asm,
                    "\tmul.ftz.{} {}, {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else if var.ty.is_double() {
                writeln!(
                    asm,
                    "\tmul.{} {}, {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else {
                writeln!(
                    asm,
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
                    asm,
                    "\tdiv.approx.ftz.{} {}, {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else if var.ty.is_double() {
                writeln!(
                    asm,
                    "\tdiv.rn.{} {}, {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else {
                writeln!(
                    asm,
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
                asm,
                "\trem.{} {}, {}, {};",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0]),
                ir.reg(var.deps[1])
            );
        }
        Op::Mulhi => {
            writeln!(
                asm,
                "\tmul.hi.{} {}, {}, {}",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0]),
                ir.reg(var.deps[1])
            );
        }
        Op::Fma => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\tfma.rn.ftz.{} {}, {}, {}, {}",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                    ir.reg(var.deps[2]),
                );
            } else if var.ty.is_double() {
                writeln!(
                    asm,
                    "\tfma.rn.{} {}, {}, {}, {}",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                    ir.reg(var.deps[2]),
                );
            } else {
                writeln!(
                    asm,
                    "\tmad.lo.{} {}, {}, {}, {}",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                    ir.reg(var.deps[2]),
                );
            }
        }
        Op::Min => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\tmin.ftz.{} {}, {}, {};\n",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                );
            } else {
                writeln!(
                    asm,
                    "\tmin.{} {}, {}, {};\n",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                );
            }
        }
        Op::Max => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\tmax.ftz.{} {}, {}, {};\n",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                );
            } else {
                writeln!(
                    asm,
                    "\tmax.{} {}, {}, {};\n",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                );
            }
        }
        Op::Cail => {
            writeln!(
                asm,
                "\tcvt.rpi.{0}.{0} {1}, {2};\n",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0]),
            );
        }
        Op::Floor => {
            writeln!(
                asm,
                "\tcvt.rmi.{0}.{0} {1}, {2};\n",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0]),
            );
        }
        Op::Round => {
            writeln!(
                asm,
                "\tcvt.rni.{0}.{0} {1}, {2};\n",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0]),
            );
        }
        Op::Trunc => {
            writeln!(
                asm,
                "\tcvt.rzi.{0}.{0} {1}, {2};\n",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0]),
            );
        }
        Op::Eq => {
            if var.ty.is_bool() {
                writeln!(
                    asm,
                    "\txor.{0} {1}, {2}, {3};\n\
                        \tnot.{0} {1}, {1};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                );
            } else {
                writeln!(
                    asm,
                    "\tsetp.eq.{}, {}, {}, {};\n",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                );
            }
        }
        Op::Neq => {
            if var.ty.is_bool() {
                writeln!(
                    asm,
                    "\txor.{} {}, {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else {
                writeln!(
                    asm,
                    "\tsetp.ne.{} {}, {}, {};\n",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            }
        }
        Op::Lt => {
            if var.ty.is_uint() {
                writeln!(
                    asm,
                    "\tsetp.lo.{} {}, {}, {};",
                    ir.var(var.deps[0]).ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else {
                writeln!(
                    asm,
                    "\tsetp.lt.{} {}, {}, {};",
                    ir.var(var.deps[0]).ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            }
        }
        Op::Le => {
            if var.ty.is_uint() {
                writeln!(
                    asm,
                    "\tsetp.ls.{} {}, {}, {};",
                    ir.var(var.deps[0]).ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else {
                writeln!(
                    asm,
                    "\tsetp.le.{} {}, {}, {};",
                    ir.var(var.deps[0]).ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            }
        }
        Op::Gt => {
            if var.ty.is_uint() {
                writeln!(
                    asm,
                    "\tsetp.hi.{} {}, {}, {};",
                    ir.var(var.deps[0]).ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else {
                writeln!(
                    asm,
                    "\tsetp.gt.{} {}, {}, {};",
                    ir.var(var.deps[0]).ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            }
        }
        Op::Ge => {
            if var.ty.is_uint() {
                writeln!(
                    asm,
                    "\tsetp.hs.{} {}, {}, {};",
                    ir.var(var.deps[0]).ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else {
                writeln!(
                    asm,
                    "\tsetp.ge.{} {}, {}, {};",
                    ir.var(var.deps[0]).ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            }
        }
        Op::Select => {
            if !ir.var(var.deps[0]).ty.is_bool() {
                writeln!(
                    asm,
                    "\tselp.{} {}, {}, {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[1]),
                    ir.reg(var.deps[2]),
                    ir.reg(var.deps[0])
                );
            } else {
                write!(
                    asm,
                    "\tand.pred %p3, {}, {};\n\
                        \tand.pred %p2, !{}, {};\n\
                        \tor.pred {}, %p2, %p3;\n",
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1]),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[2]),
                    var.reg()
                );
            }
        }
        Op::Popc => {
            if var.ty.size() == 4 {
                writeln!(
                    asm,
                    "\tpopc.{} {}, {};",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            } else {
                write!(
                    asm,
                    "\tpopc.{} %r3, {};\n\
                        \tcvt.{}.u32 {}, %r3;\n",
                    var.ty.name_cuda_bin(),
                    ir.reg(var.deps[0]),
                    var.ty.name_cuda(),
                    var.reg()
                );
            }
        }
        Op::Clz => {
            if var.ty.size() == 4 {
                writeln!(
                    asm,
                    "\tclz.{} {}, {};",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            } else {
                write!(
                    asm,
                    "\tclz.{} %r3, {};\n\
                        \tcvt.{}.u32 {}, %r3;\n",
                    var.ty.name_cuda_bin(),
                    ir.reg(var.deps[0]),
                    var.ty.name_cuda(),
                    var.reg()
                );
            }
        }
        Op::Ctz => {
            if var.ty.size() == 4 {
                write!(
                    asm,
                    "\tbrev.{} {}, {};\n\
                        \tclz.{} {}, {};\n",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    var.reg()
                );
            } else {
                write!(
                    asm,
                    "\tbrev.{} {}, {};\n\
                        \tclz.{} %r3, {};\n\
                        \tcvt.{}.u32 {}, %r3;\n",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    var.ty.name_cuda_bin(),
                    var.ty.name_cuda(),
                    var.reg(),
                    var.reg()
                );
            }
        }
        Op::And => {
            let d0 = ir.var(var.deps[0]);
            let d1 = ir.var(var.deps[1]);

            if d0.ty == d1.ty {
                writeln!(
                    asm,
                    "\tand.{} {}, {}, {};",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    d0.reg(),
                    d1.reg()
                );
            } else {
                writeln!(
                    asm,
                    "\tselp.{} {}, {}, 0, {};",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    d0.reg(),
                    d1.reg()
                );
            }
        }
        Op::Or => {
            let d0 = ir.var(var.deps[0]);
            let d1 = ir.var(var.deps[1]);

            if d0.ty == d1.ty {
                writeln!(
                    asm,
                    "\tor.{} {}, {}, {};",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    d0.reg(),
                    d1.reg()
                );
            } else {
                writeln!(
                    asm,
                    "\tselp.{} {}, -1, {}, {};",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    d0.reg(),
                    d1.reg()
                );
            }
        }
        Op::Xor => {
            writeln!(
                asm,
                "\txor.{} {}, {}, {};",
                var.ty.name_cuda_bin(),
                var.reg(),
                ir.reg(var.deps[0]),
                ir.reg(var.deps[1])
            );
        }
        Op::Shl => {
            if var.ty.size() == 4 {
                writeln!(
                    asm,
                    "\tshl.{} {}, {}, {}",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else {
                write!(
                    asm,
                    "\tcvt.u32.{} %r3, {};\n\
                        \tshl.{} {}, {}, %r3;\n",
                    ir.var(var.deps[1]).ty.name_cuda(),
                    ir.reg(var.deps[1]),
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            }
        }
        Op::Shr => {
            if var.ty.size() == 4 {
                writeln!(
                    asm,
                    "\tshr.{} {}, {}, {}",
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    ir.reg(var.deps[1])
                );
            } else {
                write!(
                    asm,
                    "\tcvt.u32.{} %r3, {};\n\
                        \tshr.{} {}, {}, %r3;\n",
                    ir.var(var.deps[1]).ty.name_cuda(),
                    ir.reg(var.deps[1]),
                    var.ty.name_cuda_bin(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            }
        }
        Op::Rcp => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\trcp.approx.ftz.{} {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            } else {
                writeln!(
                    asm,
                    "\trcp.rn.{} {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            }
        }
        Op::Rsqrt => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\trsqrt.approx.ftz.{} {}, {};",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0])
                );
            } else {
                write!(
                    asm,
                    "\trcp.rn.{} {}, {};\n\
                    \tsqrt.rn.{} {}, {};\n",
                    var.ty.name_cuda(),
                    var.reg(),
                    ir.reg(var.deps[0]),
                    var.ty.name_cuda(),
                    var.reg(),
                    var.reg()
                );
            }
        }
        Op::Sin => {
            writeln!(
                asm,
                "\tsin.approx.ftz.{} {}, {};",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0])
            );
        }
        Op::Cos => {
            writeln!(
                asm,
                "\tcos.approx.ftz.{} {}, {};",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0])
            );
        }
        Op::Exp2 => {
            writeln!(
                asm,
                "\tex2.approx.ftz.{} {}, {};",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0])
            );
        }
        Op::Log2 => {
            writeln!(
                asm,
                "\tlg2.approx.ftz.{} {}, {};",
                var.ty.name_cuda(),
                var.reg(),
                ir.reg(var.deps[0])
            );
        }
        Op::Cast => {
            let d0 = ir.var(var.deps[0]);
            if var.ty.is_bool() {
                if d0.ty.is_float() {
                    writeln!(
                        asm,
                        "\tsetp.ne.{} {}, {}, 0.0;",
                        d0.ty.name_cuda(),
                        var.reg(),
                        d0.reg()
                    );
                } else {
                    writeln!(
                        asm,
                        "\tsetp.ne.{} {}, {}, 0;",
                        d0.ty.name_cuda(),
                        var.reg(),
                        d0.reg()
                    );
                }
            } else if d0.ty.is_bool() {
                if var.ty.is_float() {
                    writeln!(
                        asm,
                        "\tselp.{} {}, 1.0, 0.0, {};",
                        var.ty.name_cuda(),
                        var.reg(),
                        d0.reg()
                    );
                } else {
                    writeln!(
                        asm,
                        "\tselp.{} {}, 1, 0, {};",
                        var.ty.name_cuda(),
                        var.reg(),
                        d0.reg()
                    );
                }
            } else if var.ty.is_float() && !d0.ty.is_float() {
                writeln!(
                    asm,
                    "\tcvt.rn.{}.{} {}, {};",
                    var.ty.name_cuda(),
                    d0.ty.name_cuda(),
                    var.reg(),
                    d0.reg()
                );
            } else if !var.ty.is_float() && d0.ty.is_float() {
                writeln!(
                    asm,
                    "\tcvt.rzi.{}.{} {}, {};",
                    var.ty.name_cuda(),
                    d0.ty.name_cuda(),
                    var.reg(),
                    d0.reg()
                );
            } else if var.ty.is_float() && d0.ty.is_float() {
                if var.ty.size() < d0.ty.size() {
                    writeln!(
                        asm,
                        "\tcvt.rn.{}.{} {}, {};",
                        var.ty.name_cuda(),
                        d0.ty.name_cuda(),
                        var.reg(),
                        d0.reg()
                    );
                } else {
                    writeln!(
                        asm,
                        "\tcvt.{}.{} {}, {};",
                        var.ty.name_cuda(),
                        d0.ty.name_cuda(),
                        var.reg(),
                        d0.reg()
                    );
                }
            }
        }
        Op::Bitcast => {
            writeln!(
                asm,
                "    mov.{} {}, {};",
                var.ty.name_cuda_bin(),
                var.reg(),
                ir.reg(var.deps[0])
            );
        }
        Op::Gather => {
            let src = ir.var(var.deps[0]);
            let index = ir.var(var.deps[1]);
            let mask = ir.var(var.deps[2]);
            let unmasked = mask.is_literal() && mask.literal != 0;
            let is_bool = var.ty.is_bool();

            // TODO: better buffer loading ( dont use as_ptr and get ptr from src in here).
            if !unmasked {
                writeln!(asm, "\t@!{} bra l_{}_masked;", mask.reg(), var.reg_idx());
            }

            // Load buffer ptr:
            let param_offset = (src.buf.unwrap() + buf_offset) * 8;

            writeln!(asm, "\tld.param.u64 %rd0, [params+{}];", param_offset,);

            // Perform actual gather:

            if var.ty.size() == 1 {
                write!(
                    asm,
                    "\tcvt.u64.{} %rd3, {};\n\
                        \tadd.u64 %rd3, %rd3, %rd0;\n",
                    index.ty.name_cuda(),
                    index.reg(),
                    // d0.reg()
                );
            } else {
                writeln!(
                    asm,
                    "\tmad.wide.{} %rd3, {}, {}, %rd0;",
                    index.ty.name_cuda(),
                    index.reg(),
                    var.ty.size(),
                    // d0.reg()
                );
            }
            if is_bool {
                write!(
                    asm,
                    "\tld.global.nc.u8 %w0, [%rd3];\n\
                        \tsetp.ne.u16 {}, %w0, 0;\n",
                    var.reg()
                );
            } else {
                writeln!(
                    asm,
                    "\tld.global.nc.{} {}, [%rd3];",
                    var.ty.name_cuda(),
                    var.reg()
                );
            }
            if !unmasked {
                write!(
                    asm,
                    "\tbra.uni l_{0}_done;\n\n\
                        l_{0}_masked:\n\
                            mov.{1} {2}, 0;\n\n\
                        l_{0}_done:\n",
                    var.reg_idx(),
                    var.ty.name_cuda_bin(),
                    var.reg()
                );
            }
        }
        Op::Scatter { op } => {
            let src = ir.var(var.deps[0]);
            let dst = ir.var(var.deps[1]);
            let idx = ir.var(var.deps[2]);
            let mask = ir.var(var.deps[3]);

            let unmasked = idx.is_literal() && idx.literal != 0;
            let is_bool = src.ty.is_bool();

            if !unmasked {
                writeln!(asm, "\t@!{} bra l_{}_done;\n", mask.reg(), var.reg_idx());
            }

            let param_offset = (dst.buf.unwrap() + buf_offset) * 8;

            writeln!(asm, "\tld.param.u64 %rd0, [params+{}];", param_offset,);

            if src.ty.size() == 1 {
                write!(
                    asm,
                    "\tcvt.u64.{} %rd3, {};\n\
                        \tadd.u64 %rd3, %rd3, %rd0;\n",
                    src.ty.name_cuda(),
                    idx.reg(),
                );
            } else {
                writeln!(
                    asm,
                    "\tmad.wide.{} %rd3, {}, {}, %rd0;",
                    idx.ty.name_cuda(),
                    idx.reg(),
                    src.ty.size(),
                );
            }

            let op_type = if op == ReduceOp::None { "st" } else { "red" };
            let op = reduce_op_name(op);
            if is_bool {
                writeln!(asm, "\tselp.u16 %w0, 1, 0, {};", src.reg());
                writeln!(asm, "\t{}.global{}.u8 [%rd3], %w0;", op_type, op);
            } else {
                writeln!(
                    asm,
                    "\t{}.global{}.{} [%rd3], {};",
                    op_type,
                    op,
                    src.ty.name_cuda(),
                    src.reg()
                );
            }

            if !unmasked {
                writeln!(asm, "\tl_{}_done:", var.reg_idx());
            }
        }
        Op::Idx => {
            writeln!(asm, "\tmov.{} {}, %r0;\n", var.ty.name_cuda(), var.reg());
        }
        Op::TexLookup { dim } => {
            let src = ir.var(var.deps[0]);

            // Load texture ptr:
            let param_offset = (src.tex.unwrap() + tex_offset) * 8;

            writeln!(asm, "\tld.param.u64 %rd0, [params+{}];", param_offset,);

            writeln!(asm, "\t.reg.f32 {}_out_<4>;", var.reg());
            if dim == 3 {
                writeln!(
                    asm,
                    "\ttex.3d.v4.f32.f32 {{{v}_out_0, {v}_out_1, {v}_out_2,
                             {v}_out_3}}, [%rd0, {{{d1}, {d2}, {d3}, {d3}}}];",
                    v = var.reg(),
                    // d0 = ir.reg(var.deps[0]),
                    d1 = ir.reg(var.deps[1]),
                    d2 = ir.reg(var.deps[2]),
                    d3 = ir.reg(var.deps[3])
                );
            } else if dim == 2 {
                writeln!(
                    asm,
                    "\ttex.2d.v4.f32.f32 {{{v}_out_0, {v}_out_1, {v}_out_2,
                             {v}_out_3}}, [%rd0, {{{d1}, {d2}}}];",
                    v = var.reg(),
                    // d0 = ir.reg(var.deps[0]),
                    d1 = ir.reg(var.deps[1]),
                    d2 = ir.reg(var.deps[2]),
                );
            } else if dim == 1 {
                writeln!(
                    asm,
                    "\ttex.1d.v4.f32.f32 {{{v}_out_0, {v}_out_1, {v}_out_2,
                             {v}_out_3}}, [%rd0, {{{d1}}}];",
                    v = var.reg(),
                    // d0 = ir.reg(var.deps[0]),
                    d1 = ir.reg(var.deps[1]),
                );
            } else {
                unimplemented!();
            }
        }
        Op::Extract { offset } => {
            writeln!(
                asm,
                "\tmov.{} {}, {}_out_{};",
                var.ty.name_cuda_bin(),
                var.reg(),
                ir.reg(var.deps[0]),
                offset
            );
        }
    }
}

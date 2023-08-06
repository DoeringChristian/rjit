use std::ops::Range;

use crate::schedule::{Env, SVarId, ScheduleIr, ScheduleVar};
use crate::trace::VarType;
use crate::util::roundup;
use crate::var::{Op, ReduceOp};

use super::params::ParamOffset;

pub const FIRST_REGISTER: usize = 4;

pub fn register_id(id: SVarId) -> usize {
    id.0 + FIRST_REGISTER
}

// Returns the register prefix for this variable
pub fn prefix(ty: &VarType) -> &'static str {
    match ty {
        VarType::Void => "%u",
        VarType::Bool => "%p",
        VarType::I8 => "%b",
        VarType::U8 => "%b",
        VarType::I16 => "%w",
        VarType::U16 => "%w",
        VarType::I32 => "%r",
        VarType::U32 => "%r",
        VarType::I64 => "%rd",
        VarType::U64 => "%rd",
        VarType::F16 => "%h",
        VarType::F32 => "%f",
        VarType::F64 => "%d",
    }
}
// Retuns the cuda/ptx Representation for this type
pub fn tyname(ty: &VarType) -> &'static str {
    match ty {
        VarType::Void => "???",
        VarType::Bool => "pred",
        VarType::I8 => "s8",
        VarType::U8 => "u8",
        VarType::I16 => "s16",
        VarType::U16 => "u16",
        VarType::I32 => "s32",
        VarType::U32 => "u32",
        VarType::I64 => "s64",
        VarType::U64 => "u64",
        VarType::F16 => "f16",
        VarType::F32 => "f32",
        VarType::F64 => "f64",
    }
}
pub fn tyname_bin(ty: &VarType) -> &'static str {
    match ty {
        VarType::Void => "???",
        VarType::Bool => "pred",
        VarType::I8 => "b8",
        VarType::U8 => "b8",
        VarType::I16 => "b16",
        VarType::U16 => "b16",
        VarType::I32 => "b32",
        VarType::U32 => "b32",
        VarType::I64 => "b64",
        VarType::U64 => "b64",
        VarType::F16 => "b16",
        VarType::F32 => "b32",
        VarType::F64 => "b64",
    }
}

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

pub struct Reg<'a>(pub SVarId, pub &'a ScheduleVar);
impl<'a> std::fmt::Display for Reg<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", prefix(&self.1.ty), register_id(self.0))
    }
}

pub fn assemble_entry(
    asm: &mut impl std::fmt::Write,
    ir: &ScheduleIr,
    env: &Env,
    entry_point: &str,
) -> std::fmt::Result {
    let reg = |id| Reg(id, ir.var(id));

    let n_params = 1 + env.buffers().len() + env.textures().len(); // Add 1 for size
    let n_regs = ir.n_vars() + FIRST_REGISTER;

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

    writeln!(asm, ".version {}.{}", 8, 0)?;
    writeln!(asm, ".target {}", "sm_86")?;
    writeln!(asm, ".address_size 64")?;

    writeln!(asm, "")?;

    writeln!(asm, ".entry {}(", entry_point)?;
    writeln!(
        asm,
        "\t.param .align 8 .b8 params[{}]) {{",
        ((n_params + 1) * std::mem::size_of::<u64>())
    )?;
    writeln!(asm, "")?;

    writeln!(
        asm,
        "\t.reg.b8   %b <{n_regs}>; .reg.b16 %w<{n_regs}>; .reg.b32 %r<{n_regs}>;"
    )?;
    writeln!(
        asm,
        "\t.reg.b64  %rd<{n_regs}>; .reg.f32 %f<{n_regs}>; .reg.f64 %d<{n_regs}>;"
    )?;
    writeln!(asm, "\t.reg.pred %p <{n_regs}>;")?;
    writeln!(asm, "")?;

    write!(
        asm,
        "\tmov.u32 %r0, %ctaid.x;\n\
            \tmov.u32 %r1, %ntid.x;\n\
            \tmov.u32 %r2, %tid.x;\n\
            \tmad.lo.u32 %r0, %r0, %r1, %r2; // r0 <- Index\n"
    )?;

    writeln!(asm, "")?;

    writeln!(
        asm,
        "\t// Index Conditional (jump to done if Index >= Size)."
    )?;
    writeln!(
        asm,
        "\tld.param.u32 %r2, [params]; // r2 <- params[0] (Size)"
    )?;

    write!(
        asm,
        "\tsetp.ge.u32 %p0, %r0, %r2; // p0 <- r0 >= r2\n\
           \t@%p0 bra done; // if p0 => done\n\
           \t\n\
           \tmov.u32 %r3, %nctaid.x; // r3 <- nctaid.x\n\
           \tmul.lo.u32 %r1, %r3, %r1; // r1 <- r3 * r1\n\
           \t\n"
    )?;

    write!(asm, "body: // sm_{}\n", 86)?; // TODO: compute capability from device

    let offsets = ParamOffset::from_env(env);
    for id in ir.ids() {
        // let var = ir.var(id);
        assemble_var(asm, ir, id, &offsets, "param")?;
    }

    // End of kernel:

    writeln!(asm, "\n\t//End of Kernel:")?;
    writeln!(
        asm,
        "\n\tadd.u32 %r0, %r0, %r1; // r0 <- r0 + r1\n\
           \tsetp.ge.u32 %p0, %r0, %r2; // p0 <- r1 >= r2\n\
           \t@!%p0 bra body; // if p0 => body\n\
           \n"
    )?;
    writeln!(asm, "done:")?;
    write!(
        asm,
        "\n\tret;\n\
       }}\n"
    )?;
    Ok(())
}

// #[allow(warnings)]
pub fn assemble_var(
    asm: &mut impl std::fmt::Write,
    ir: &ScheduleIr,
    vid: SVarId,
    offsets: &ParamOffset,
    params_type: &'static str,
) -> std::fmt::Result {
    let reg = |id| Reg(id, ir.var(id));
    let dep = |id, dep_idx: usize| ir.dep(id, dep_idx);

    let var = ir.var(vid);
    writeln!(asm, "")?;
    writeln!(asm, "\t// [{}]: {:?} =>", vid, var)?;

    match var.op {
        Op::Literal => {
            if let Some(opaque) = var.data.opaque() {
                writeln!(
                    asm,
                    "\tld.{params_type}.{} {}, [{}+{}];",
                    tyname_bin(&var.ty),
                    reg(vid),
                    "params",
                    offsets.opaque(opaque as _) * 8,
                )?;
                // writeln!(asm, "\tmov.{} {}, %rd0;", var.ty.name_cuda_bin(), var.reg(),)?;
            } else {
                writeln!(
                    asm,
                    "\tmov.{} {}, 0x{:x};\n",
                    tyname_bin(&var.ty),
                    reg(vid),
                    var.data.literal().unwrap()
                )?;
            }
        }
        Op::Neg => {
            if var.ty.is_uint() {
                writeln!(
                    asm,
                    "\tneg.s{} {}, {};",
                    var.ty.size() * 8,
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            } else {
                writeln!(
                    asm,
                    "\tneg.{} {}, {};\n",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            }
        }
        Op::Not => {
            writeln!(
                asm,
                "\tnot.{} {}, {};",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0))
            )?;
        }
        Op::Sqrt => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\tsqrt.approx.ftz.{} {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            } else {
                writeln!(
                    asm,
                    "\tsqrt.rn.{} {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            }
        }
        Op::Abs => {
            writeln!(
                asm,
                "\tabs.{} {}, {};",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0))
            )?;
        }
        Op::Add => {
            writeln!(
                asm,
                "\tadd.{} {}, {}, {};",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0)),
                reg(dep(vid, 1))
            )?;
        }
        Op::Sub => {
            writeln!(
                asm,
                "\tsub.{} {}, {}, {};",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0)),
                reg(dep(vid, 1)),
            )?;
        }
        Op::Mul => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\tmul.ftz.{} {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            } else if var.ty.is_double() {
                writeln!(
                    asm,
                    "\tmul.{} {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tmul.lo.{} {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            }
        }
        Op::Div => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\tdiv.approx.ftz.{} {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            } else if var.ty.is_double() {
                writeln!(
                    asm,
                    "\tdiv.rn.{} {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            } else {
                writeln!(
                    asm,
                    "\tdiv.{} {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            }
        }
        Op::Mod => {
            writeln!(
                asm,
                "\trem.{} {}, {}, {};",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0)),
                reg(dep(vid, 1))
            )?;
        }
        Op::Mulhi => {
            writeln!(
                asm,
                "\tmul.hi.{} {}, {}, {};",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0)),
                reg(dep(vid, 1))
            )?;
        }
        Op::Fma => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\tfma.rn.ftz.{} {}, {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                    reg(dep(vid, 2))
                )?;
            } else if var.ty.is_double() {
                writeln!(
                    asm,
                    "\tfma.rn.{} {}, {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                    reg(dep(vid, 2)),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tmad.lo.{} {}, {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                    reg(dep(vid, 2)),
                )?;
            }
        }
        Op::Min => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\tmin.ftz.{} {}, {}, {};\n",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tmin.{} {}, {}, {};\n",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            }
        }
        Op::Max => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\tmax.ftz.{} {}, {}, {};\n",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            } else {
                writeln!(
                    asm,
                    "\tmax.{} {}, {}, {};\n",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            }
        }
        Op::Ceil => {
            writeln!(
                asm,
                "\tcvt.rpi.{0}.{0} {1}, {2};\n",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0)),
            )?;
        }
        Op::Floor => {
            writeln!(
                asm,
                "\tcvt.rmi.{0}.{0} {1}, {2};\n",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0)),
            )?;
        }
        Op::Round => {
            writeln!(
                asm,
                "\tcvt.rni.{0}.{0} {1}, {2};\n",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0)),
            )?;
        }
        Op::Trunc => {
            writeln!(
                asm,
                "\tcvt.rzi.{0}.{0} {1}, {2};\n",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0)),
            )?;
        }
        Op::Eq => {
            if var.ty.is_bool() {
                writeln!(
                    asm,
                    "\txor.{0} {1}, {2}, {3};\n\
                        \tnot.{0} {1}, {1};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tsetp.eq.{}, {}, {}, {};\n",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            }
        }
        Op::Neq => {
            if var.ty.is_bool() {
                writeln!(
                    asm,
                    "\txor.{} {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            } else {
                writeln!(
                    asm,
                    "\tsetp.ne.{} {}, {}, {};\n",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            }
        }
        Op::Lt => {
            if var.ty.is_uint() {
                writeln!(
                    asm,
                    "\tsetp.lo.{} {}, {}, {};",
                    tyname(&ir.var(dep(vid, 0)).ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tsetp.lt.{} {}, {}, {};",
                    tyname(&ir.var(dep(vid, 0)).ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            }
        }
        Op::Le => {
            if var.ty.is_uint() {
                writeln!(
                    asm,
                    "\tsetp.ls.{} {}, {}, {};",
                    tyname(&ir.var(dep(vid, 0)).ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tsetp.le.{} {}, {}, {};",
                    tyname(&ir.var(dep(vid, 0)).ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            }
        }
        Op::Gt => {
            if var.ty.is_uint() {
                writeln!(
                    asm,
                    "\tsetp.hi.{} {}, {}, {};",
                    tyname(&ir.var(dep(vid, 0)).ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            } else {
                writeln!(
                    asm,
                    "\tsetp.gt.{} {}, {}, {};",
                    tyname(&ir.var(dep(vid, 0)).ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            }
        }
        Op::Ge => {
            if var.ty.is_uint() {
                writeln!(
                    asm,
                    "\tsetp.hs.{} {}, {}, {};",
                    tyname(&ir.var(dep(vid, 0)).ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            } else {
                writeln!(
                    asm,
                    "\tsetp.ge.{} {}, {}, {};",
                    tyname(&ir.var(dep(vid, 0)).ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            }
        }
        Op::Select => {
            if !ir.var(dep(vid, 1)).ty.is_bool() {
                writeln!(
                    asm,
                    "\tselp.{} {}, {}, {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 1)),
                    reg(dep(vid, 2)),
                    reg(dep(vid, 0))
                )?;
            } else {
                write!(
                    asm,
                    "\tand.pred %p3, {}, {};\n\
                        \tand.pred %p2, !{}, {};\n\
                        \tor.pred {}, %p2, %p3;\n",
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 2)),
                    reg(vid),
                )?;
            }
        }
        Op::Popc => {
            if var.ty.size() == 4 {
                writeln!(
                    asm,
                    "\tpopc.{} {}, {};",
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            } else {
                write!(
                    asm,
                    "\tpopc.{} %r3, {};\n\
                        \tcvt.{}.u32 {}, %r3;\n",
                    tyname_bin(&var.ty),
                    reg(dep(vid, 0)),
                    tyname(&var.ty),
                    reg(vid),
                )?;
            }
        }
        Op::Clz => {
            if var.ty.size() == 4 {
                writeln!(
                    asm,
                    "\tclz.{} {}, {};",
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            } else {
                write!(
                    asm,
                    "\tclz.{} %r3, {};\n\
                        \tcvt.{}.u32 {}, %r3;\n",
                    tyname_bin(&var.ty),
                    reg(dep(vid, 0)),
                    tyname(&var.ty),
                    reg(vid),
                )?;
            }
        }
        Op::Ctz => {
            if var.ty.size() == 4 {
                write!(
                    asm,
                    "\tbrev.{} {}, {};\n\
                        \tclz.{} {}, {};\n",
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(vid),
                )?;
            } else {
                write!(
                    asm,
                    "\tbrev.{} {}, {};\n\
                        \tclz.{} %r3, {};\n\
                        \tcvt.{}.u32 {}, %r3;\n",
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    tyname_bin(&var.ty),
                    tyname(&var.ty),
                    reg(vid),
                    reg(vid),
                )?;
            }
        }
        Op::And => {
            let d0 = ir.var(dep(vid, 0));
            let d1 = ir.var(dep(vid, 1));

            if d0.ty == d1.ty {
                writeln!(
                    asm,
                    "\tand.{} {}, {}, {};",
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tselp.{} {}, {}, 0, {};",
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            }
        }
        Op::Or => {
            let d0 = ir.var(dep(vid, 0));
            let d1 = ir.var(dep(vid, 1));

            if d0.ty == d1.ty {
                writeln!(
                    asm,
                    "\tor.{} {}, {}, {};",
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tselp.{} {}, -1, {}, {};",
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1)),
                )?;
            }
        }
        Op::Xor => {
            writeln!(
                asm,
                "\txor.{} {}, {}, {};",
                tyname_bin(&var.ty),
                reg(vid),
                reg(dep(vid, 0)),
                reg(dep(vid, 1))
            )?;
        }
        Op::Shl => {
            if var.ty.size() == 4 {
                writeln!(
                    asm,
                    "\tshl.{} {}, {}, {};",
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            } else {
                write!(
                    asm,
                    "\tcvt.u32.{} %r3, {};\n\
                        \tshl.{} {}, {}, %r3;\n",
                    tyname(&ir.var(dep(vid, 1)).ty),
                    reg(dep(vid, 1)),
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            }
        }
        Op::Shr => {
            if var.ty.size() == 4 {
                writeln!(
                    asm,
                    "\tshr.{} {}, {}, {};",
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    reg(dep(vid, 1))
                )?;
            } else {
                write!(
                    asm,
                    "\tcvt.u32.{} %r3, {};\n\
                        \tshr.{} {}, {}, %r3;\n",
                    tyname(&ir.var(dep(vid, 1)).ty),
                    reg(dep(vid, 1)),
                    tyname_bin(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            }
        }
        Op::Rcp => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\trcp.approx.ftz.{} {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            } else {
                writeln!(
                    asm,
                    "\trcp.rn.{} {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            }
        }
        Op::Rsqrt => {
            if var.ty.is_single() {
                writeln!(
                    asm,
                    "\trsqrt.approx.ftz.{} {}, {};",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0))
                )?;
            } else {
                write!(
                    asm,
                    "\trcp.rn.{} {}, {};\n\
                    \tsqrt.rn.{} {}, {};\n",
                    tyname(&var.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                    tyname(&var.ty),
                    reg(vid),
                    reg(vid),
                )?;
            }
        }
        Op::Sin => {
            writeln!(
                asm,
                "\tsin.approx.ftz.{} {}, {};",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0))
            )?;
        }
        Op::Cos => {
            writeln!(
                asm,
                "\tcos.approx.ftz.{} {}, {};",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0))
            )?;
        }
        Op::Exp2 => {
            writeln!(
                asm,
                "\tex2.approx.ftz.{} {}, {};",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0))
            )?;
        }
        Op::Log2 => {
            writeln!(
                asm,
                "\tlg2.approx.ftz.{} {}, {};",
                tyname(&var.ty),
                reg(vid),
                reg(dep(vid, 0))
            )?;
        }
        Op::Cast => {
            let d0 = ir.var(dep(vid, 0));
            if var.ty.is_bool() {
                if d0.ty.is_float() {
                    writeln!(
                        asm,
                        "\tsetp.ne.{} {}, {}, 0.0;",
                        tyname(&d0.ty),
                        reg(vid),
                        reg(dep(vid, 0)),
                    )?;
                } else {
                    writeln!(
                        asm,
                        "\tsetp.ne.{} {}, {}, 0;",
                        tyname(&d0.ty),
                        reg(vid),
                        reg(dep(vid, 0)),
                    )?;
                }
            } else if d0.ty.is_bool() {
                if var.ty.is_float() {
                    writeln!(
                        asm,
                        "\tselp.{} {}, 1.0, 0.0, {};",
                        tyname(&var.ty),
                        reg(vid),
                        reg(dep(vid, 0)),
                    )?;
                } else {
                    writeln!(
                        asm,
                        "\tselp.{} {}, 1, 0, {};",
                        tyname(&var.ty),
                        reg(vid),
                        reg(dep(vid, 0)),
                    )?;
                }
            } else if var.ty.is_float() && !d0.ty.is_float() {
                writeln!(
                    asm,
                    "\tcvt.rn.{}.{} {}, {};",
                    tyname(&var.ty),
                    tyname(&d0.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                )?;
            } else if !var.ty.is_float() && d0.ty.is_float() {
                writeln!(
                    asm,
                    "\tcvt.rzi.{}.{} {}, {};",
                    tyname(&var.ty),
                    tyname(&d0.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                )?;
            } else if var.ty.is_float() && d0.ty.is_float() && var.ty.size() < d0.ty.size() {
                writeln!(
                    asm,
                    "\tcvt.rn.{}.{} {}, {};",
                    tyname(&var.ty),
                    tyname(&d0.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tcvt.{}.{} {}, {};",
                    tyname(&var.ty),
                    tyname(&d0.ty),
                    reg(vid),
                    reg(dep(vid, 0)),
                )?;
            }
        }
        Op::Bitcast => {
            writeln!(
                asm,
                "\tmov.{} {}, {};",
                tyname_bin(&var.ty),
                reg(vid),
                reg(dep(vid, 0))
            )?;
        }
        Op::Gather => {
            let src = ir.var(dep(vid, 0));
            let index = dep(vid, 1);
            let mask = dep(vid, 2);
            let unmasked = ir.var(mask).is_literal()
                && ir.var(mask).data.literal().is_some()
                && ir.var(mask).data.literal().unwrap() != 0;
            let is_bool = var.ty.is_bool();

            // TODO: better buffer loading ( dont use as_ptr and get ptr from src in here).
            if !unmasked {
                writeln!(asm, "\t@!{} bra l_{}_masked;", reg(mask), register_id(vid))?;
            }

            // Load buffer ptr:
            let param_offset = offsets.buffer(src.data.buffer().unwrap() as _) * 8;

            writeln!(
                asm,
                "\tld.{params_type}.u64 %rd0, [params+{}];",
                param_offset,
            )?;

            // Perform actual gather:

            if var.ty.size() == 1 {
                write!(
                    asm,
                    "\tcvt.u64.{} %rd3, {};\n\
                        \tadd.u64 %rd3, %rd3, %rd0;\n",
                    tyname(&ir.var(index).ty),
                    reg(index),
                    // d0.reg()
                )?;
            } else {
                writeln!(
                    asm,
                    "\tmad.wide.{} %rd3, {}, {}, %rd0;",
                    tyname(&ir.var(index).ty),
                    reg(index),
                    var.ty.size(),
                    // d0.reg()
                )?;
            }
            if is_bool {
                write!(
                    asm,
                    "\tld.global.nc.u8 %w0, [%rd3];\n\
                        \tsetp.ne.u16 {}, %w0, 0;\n",
                    reg(vid),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tld.global.nc.{} {}, [%rd3];",
                    tyname(&var.ty),
                    reg(vid),
                )?;
            }
            if !unmasked {
                write!(
                    asm,
                    "\tbra.uni l_{0}_done;\n\n\
                        l_{0}_masked:\n\
                            mov.{1} {2}, 0;\n\n\
                        l_{0}_done:\n",
                    register_id(vid),
                    tyname_bin(&var.ty),
                    reg(vid),
                )?;
            }
        }
        Op::Scatter { op } => {
            let src = dep(vid, 0);
            let dst = ir.var(dep(vid, 1));
            let index = dep(vid, 2);
            let mask = dep(vid, 3);

            let unmasked = ir.var(mask).is_literal()
                && ir.var(mask).data.literal().is_some()
                && ir.var(mask).data.literal().unwrap() != 0;
            let index_zero = ir.var(index).data.literal().is_some_and(|data| data == 0);
            let is_bool = ir.var(src).ty.is_bool();

            if !unmasked {
                writeln!(asm, "\t@!{} bra l_{}_done;\n", reg(mask), register_id(vid))?;
            }

            let param_offset = offsets.buffer(dst.data.buffer().unwrap() as _) * 8;

            writeln!(
                asm,
                "\tld.{params_type}.u64 %rd0, [params+{}];",
                param_offset,
            )?;

            if index_zero {
                writeln!(asm, "\tmov.u64 %rd3, %rd0;")?;
            } else if ir.var(src).ty.size() == 1 {
                // Cast index to u64 if type size == 1
                write!(
                    asm,
                    "\tcvt.u64.{} %rd3, {};\n\
                        \tadd.u64 %rd3, %rd3, %rd0;\n",
                    tyname(&ir.var(index).ty),
                    reg(index),
                )?;
            } else {
                writeln!(
                    asm,
                    "\tmad.wide.{} %rd3, {}, {}, %rd0;",
                    tyname(&ir.var(index).ty),
                    reg(index),
                    ir.var(src).ty.size(),
                )?;
            }

            let op_type = if op == ReduceOp::None { "st" } else { "red" };
            let op = reduce_op_name(op);
            if is_bool {
                writeln!(asm, "\tselp.u16 %w0, 1, 0, {};", reg(src))?;
                writeln!(asm, "\t{}.global{}.u8 [%rd3], %w0;", op_type, op)?;
            } else {
                writeln!(
                    asm,
                    "\t{}.global{}.{} [%rd3], {};",
                    op_type,
                    op,
                    tyname(&ir.var(src).ty),
                    reg(src),
                )?;
            }

            if !unmasked {
                writeln!(asm, "\tl_{}_done:", register_id(vid))?;
            }
        }
        Op::Idx => {
            writeln!(asm, "\tmov.{} {}, %r0;\n", tyname(&var.ty), reg(vid))?;
        }
        Op::TexLookup { dim, channels } => {
            let channels_rounded = roundup(channels as _, 4);
            let src = ir.var(dep(vid, 0));

            let offset_range = offsets.texture_ranges(src.data.texture().unwrap() as usize);
            writeln!(asm, "\t.reg.f32 {}_out_<{channels_rounded}>;", reg(vid))?;
            let mut out_offset = 0;

            for param_offset in offset_range {
                // Load texture ptr
                writeln!(
                    asm,
                    "\tld.{params_type}.u64 %rd0, [params+{}];",
                    param_offset * 8,
                )?;

                // writeln!(asm, "\t.reg.f32 {}_out_<4>;", reg(vid))?;
                let out = format!(
                    "{{{v}_out_{o0}, {v}_out_{o1}, {v}_out_{o2},
                             {v}_out_{o3}}}",
                    v = reg(vid),
                    o0 = out_offset + 0,
                    o1 = out_offset + 1,
                    o2 = out_offset + 2,
                    o3 = out_offset + 3,
                );
                if dim == 3 {
                    writeln!(
                        asm,
                        "\ttex.3d.v4.f32.f32 {out}, [%rd0, {{{d1}, {d2}, {d3}, {d3}}}];",
                        // d0 = dep(vid, 0),
                        d1 = reg(dep(vid, 1)),
                        d2 = reg(dep(vid, 2)),
                        d3 = reg(dep(vid, 3)),
                    )?;
                } else if dim == 2 {
                    writeln!(
                        asm,
                        "\ttex.2d.v4.f32.f32 {out}, [%rd0, {{{d1}, {d2}}}];",
                        d1 = reg(dep(vid, 1)),
                        d2 = reg(dep(vid, 2)),
                    )?;
                } else if dim == 1 {
                    writeln!(
                        asm,
                        "\ttex.1d.v4.f32.f32 {out}, [%rd0, {{{d1}}}];",
                        d1 = reg(dep(vid, 1)),
                    )?;
                } else {
                    unimplemented!();
                }
                out_offset += 4;
            }
        }
        Op::Extract { offset } => {
            writeln!(
                asm,
                "\tmov.{} {}, {}_out_{};",
                tyname_bin(&var.ty),
                reg(vid),
                reg(dep(vid, 0)),
                offset
            )?;
        }
        Op::Loop {} => {
            todo!()
        }
        _ => {}
    }
    Ok(())
}

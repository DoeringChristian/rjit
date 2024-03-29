use crate::backend::cuda;
use crate::backend::cuda::codegen::{register_id, Reg};
use crate::backend::cuda::params::ParamOffset;
use crate::schedule::{Env, SVarId, ScheduleIr};
use crate::trace::Op;

pub fn assemble_var_rt(
    asm: &mut impl std::fmt::Write,
    ir: &ScheduleIr,
    vid: SVarId,
    offsets: &ParamOffset,
    params_type: &'static str,
) -> std::fmt::Result {
    let reg = |id| Reg(id, ir.var(id));
    let dep = |id, dep_idx: usize| ir.dep(id, dep_idx);

    let var = ir.var(vid);
    match var.op {
        Op::TraceRay { payload_count } => {
            writeln!(asm, "")?;
            writeln!(asm, "\t// [{}]: {:?} =>", vid, var)?;
            let handle_offset = offsets.accel(ir.var(dep(vid, 0)).data.accel().unwrap() as _) * 8;

            writeln!(
                asm,
                "\tld.{params_type}.u64 %rd0, [params+{}];",
                handle_offset
            )?;

            let mask = dep(vid, 1);
            // let pipeline = ir.var(var.deps[1]);
            // let sbt = ir.var(var.deps[2]);

            writeln!(asm, "\t.reg.u32 {}_out_<32>;", reg(vid))?;

            let masked = !ir.var(mask).is_literal()
                || ir.var(mask).data.literal().is_some()
                || ir.var(mask).data.literal().unwrap() == 0;
            if masked {
                writeln!(asm, "\t@!{} bra l_masked_{};", reg(mask), register_id(vid))?;
            }

            write!(
                asm,
                "\t.reg.u32 {v}_payload_type, {v}_payload_count;\n\
                \tmov.u32 {v}_payload_type, 0;\n\
                \tmov.u32 {v}_payload_count, {};\n",
                payload_count,
                v = reg(vid),
            )?;

            writeln!(asm, "call (")?;

            for i in 0..32 {
                writeln!(
                    asm,
                    "{}_out_{}{}",
                    reg(vid),
                    i,
                    if i + 1 < 32 { ", " } else { "" }
                )?;
            }

            writeln!(asm, "), _optix_trace_typed_32, (")?;

            writeln!(asm, "{}_payload_type, ", reg(vid))?;
            writeln!(asm, "%rd0, ")?;
            writeln!(asm, "{}, ", reg(dep(vid, 2)))?;
            writeln!(asm, "{}, ", reg(dep(vid, 3)))?;
            writeln!(asm, "{}, ", reg(dep(vid, 4)))?;

            writeln!(asm, "{}, ", reg(dep(vid, 5)))?;
            writeln!(asm, "{}, ", reg(dep(vid, 6)))?;
            writeln!(asm, "{}, ", reg(dep(vid, 7)))?;

            writeln!(asm, "{}, ", reg(dep(vid, 8)))?;
            writeln!(asm, "{}, ", reg(dep(vid, 9)))?;
            writeln!(asm, "{}, ", reg(dep(vid, 10)))?;

            writeln!(asm, "{}, ", reg(dep(vid, 11)))?;
            writeln!(asm, "{}, ", reg(dep(vid, 12)))?;
            writeln!(asm, "{}, ", reg(dep(vid, 13)))?;
            writeln!(asm, "{}, ", reg(dep(vid, 14)))?;
            writeln!(asm, "{}, ", reg(dep(vid, 15)))?;

            writeln!(asm, "{}_payload_count, ", reg(vid))?;

            for i in 0..payload_count {
                writeln!(
                    asm,
                    "{}{}",
                    reg(dep(vid, 16 + i)),
                    if i + 1 < 32 { "," } else { "" }
                )?;
            }

            for i in payload_count..32 {
                writeln!(
                    asm,
                    "{}_out_{}{}",
                    reg(vid),
                    i,
                    if i + 1 < 32 { "," } else { "" }
                )?;
            }

            writeln!(asm, ");")?;

            if masked {
                writeln!(asm, "\nl_masked_{}:", register_id(vid))?;
            }
        }
        _ => {
            crate::backend::cuda::codegen::assemble_var(asm, ir, vid, offsets, params_type)?;
        }
    }
    Ok(())
}

pub fn assemble_entry(
    asm: &mut impl std::fmt::Write,
    ir: &ScheduleIr,
    env: &Env,
    entry_point: &str,
) -> std::fmt::Result {
    let reg = |id| Reg(id, ir.var(id));

    let n_params = 1 + env.buffers().len() + env.textures().len() + env.accels().len(); // Add 1 for size
    let n_regs = ir.n_vars() + cuda::codegen::FIRST_REGISTER;

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

    writeln!(asm, ".const .align 8 .b8 params[{}];", 8 * n_params)?;
    writeln!(asm, ".entry {}(){{", entry_point)?;
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
        "\tcall (%r0), _optix_get_launch_index_x, ();\n\
            \tld.const.u32 %r1, [params + 4];\n\
            \tadd.u32 %r0, %r0, %r1;\n\n\
            body:\n"
    )?;

    let offsets = ParamOffset::from_env(env);
    for id in ir.ids() {
        assemble_var_rt(asm, ir, id, &offsets, "const")?;
    }

    write!(
        asm,
        "\n\tret;\n\
       }}\n"
    )?;
    Ok(())
}

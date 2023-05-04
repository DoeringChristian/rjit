use crate::schedule::{Env, SVarId, ScheduleIr};
use crate::trace::{Op, VarType};
use crate::var::ParamType;

pub fn assemble_var_rt(
    asm: &mut impl std::fmt::Write,
    ir: &ScheduleIr,
    id: SVarId,
    buf_offset: usize,
    tex_offset: usize,
    accel_offset: usize,
    params_type: &'static str,
) -> std::fmt::Result {
    let var = ir.var(id);
    match var.op {
        Op::TraceRay { payload_count } => {
            writeln!(asm, "")?;
            writeln!(asm, "\t// [{}]: {:?} =>", id, var)?;
            let handle_offset = ir.var(var.deps[0]).accel.unwrap() * 8;

            writeln!(
                asm,
                "\tld.{params_type}.u64 %rd0, [params+{}];",
                handle_offset
            )?;

            let mask = ir.var(var.deps[1]);
            // let pipeline = ir.var(var.deps[1]);
            // let sbt = ir.var(var.deps[2]);

            writeln!(asm, "\t.reg.u32 {}_out_<32>;", var.reg())?;

            let masked = !mask.is_literal() || mask.literal == 0;
            if masked {
                writeln!(asm, "\t@!{} bra l_masked_{};", mask.reg(), var.reg_idx())?;
            }

            write!(
                asm,
                "\t.reg.u32 {v}_payload_type, {v}_payload_count;\n\
                \tmov.u32 {v}_payload_type, 0;\n\
                \tmov.u32 {v}_payload_count, {};\n",
                payload_count,
                v = var.reg(),
            )?;

            writeln!(asm, "call (")?;

            for i in 0..32 {
                writeln!(
                    asm,
                    "{}_out_{}{}",
                    var.reg(),
                    i,
                    if i + 1 < 32 { ", " } else { "" }
                )?;
            }

            writeln!(asm, "), _optix_trace_typed_32, (")?;

            writeln!(asm, "{}_payload_type, ", var.reg())?;
            writeln!(asm, "%rd0, ")?;
            writeln!(asm, "{}, ", ir.reg(var.deps[2]))?;
            writeln!(asm, "{}, ", ir.reg(var.deps[3]))?;
            writeln!(asm, "{}, ", ir.reg(var.deps[4]))?;

            writeln!(asm, "{}, ", ir.reg(var.deps[5]))?;
            writeln!(asm, "{}, ", ir.reg(var.deps[6]))?;
            writeln!(asm, "{}, ", ir.reg(var.deps[7]))?;

            writeln!(asm, "{}, ", ir.reg(var.deps[8]))?;
            writeln!(asm, "{}, ", ir.reg(var.deps[9]))?;
            writeln!(asm, "{}, ", ir.reg(var.deps[10]))?;

            writeln!(asm, "{}, ", ir.reg(var.deps[11]))?;
            writeln!(asm, "{}, ", ir.reg(var.deps[12]))?;
            writeln!(asm, "{}, ", ir.reg(var.deps[13]))?;
            writeln!(asm, "{}, ", ir.reg(var.deps[14]))?;
            writeln!(asm, "{}, ", ir.reg(var.deps[15]))?;

            writeln!(asm, "{}_payload_count, ", var.reg())?;

            for i in 0..32 {
                writeln!(
                    asm,
                    "{}_out_{}{}",
                    var.reg(),
                    i,
                    if i + 1 < 32 { "," } else { "" }
                )?;
            }

            writeln!(asm, ");")?;

            if masked {
                writeln!(asm, "\nl_masked_{}:", var.reg_idx())?;
            }
        }
        _ => {
            crate::backend::cuda::codegen::assemble_var(
                asm,
                ir,
                id,
                buf_offset,
                tex_offset,
                params_type,
            )?;
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
    let n_params = 1 + env.buffers().len() + env.textures().len(); // Add 1 for size
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

    for id in ir.ids() {
        let var = ir.var(id);
        match var.param_ty {
            ParamType::None => assemble_var_rt(
                asm,
                ir,
                id,
                1,
                1 + env.buffers().len(),
                1 + env.buffers().len() + env.textures().len(),
                "const",
            )?,
            ParamType::Input => {
                let param_offset = (var.buf.unwrap() + 1) * 8;
                // Load from params
                writeln!(asm, "")?;
                writeln!(asm, "\t// [{}]: {:?} =>", id, var)?;
                if var.is_literal() {
                    writeln!(
                        asm,
                        "\tld.param.{} {}, [params+{}];",
                        var.ty.name_cuda(),
                        var.reg(),
                        param_offset
                    )?;
                    continue;
                } else {
                    writeln!(asm, "\tld.const.u64 %rd0, [params+{}];", param_offset)?;
                }
                if var.size > 1 {
                    writeln!(asm, "\tmad.wide.u32 %rd0, %r0, {}, %rd0;", var.ty.size())?;
                }

                if var.ty == VarType::Bool {
                    writeln!(asm, "\tld.global.cs.u8 %w0, [%rd0];")?;
                    writeln!(asm, "\tsetp.ne.u16 {}, %w0, 0;", var.reg())?;
                } else {
                    writeln!(
                        asm,
                        "\tld.global.cs.{} {}, [%rd0];",
                        var.ty.name_cuda(),
                        var.reg(),
                    )?;
                }
            }
            ParamType::Output => {
                let param_offset = (var.buf.unwrap() + 1) * 8;
                assemble_var_rt(
                    asm,
                    ir,
                    id,
                    1,
                    1 + env.buffers().len(),
                    1 + env.buffers().len() + env.textures().len(),
                    "const",
                )?;
                // let offset = param_idx * 8;
                write!(
                    asm,
                    "\n\t// Store:\n\
                           \tld.const.u64 %rd0, [params + {}]; // rd0 <- params[offset]\n\
                           \tmad.wide.u32 %rd0, %r0, {}, %rd0; // rd0 <- Index * ty.size() + \
                           params[offset]\n",
                    param_offset,
                    var.ty.size(),
                )?;

                if var.ty == VarType::Bool {
                    writeln!(asm, "\tselp.u16 %w0, 1, 0, {};", var.reg())?;
                    writeln!(asm, "\tst.global.cs.u8 [%rd0], %w0;")?;
                } else {
                    writeln!(
                               asm,
                               "\tst.global.cs.{} [%rd0], {}; // (Index * ty.size() + params[offset])[Index] <- var",
                               var.ty.name_cuda(),
                               var.reg(),
                           )?;
                }
            }
        }
    }

    write!(
        asm,
        "\n\tret;\n\
       }}\n"
    )?;
    Ok(())
}

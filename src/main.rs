use std::ffi::c_void;

use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::Module;
use cust::util::{DeviceCopyExt, SliceExt};

use self::compiler::CUDACompiler;
use self::ir::{Ir, Op, Var, VarId, VarType};

mod compiler;
mod ir;
mod iterators;

fn main() {
    let ctx = cust::quick_init().unwrap();
    let device = cust::device::Device::get_device(0).unwrap();

    let mut ir = Ir::default();

    let size = 10;
    ir.set_size(size as u64);
    let x_buf = vec![1f32; size].as_slice().as_dbuf().unwrap();
    dbg!(&x_buf);

    let buf_id = ir.push_param(x_buf.as_device_ptr().as_raw());

    let x = ir.push_var(Var {
        op: Op::Load(buf_id),
        ty: VarType::F32,
    });
    let c = ir.push_var(Var {
        op: Op::ConstF32(2.),
        ty: VarType::F32,
    });
    let y = ir.push_var(Var {
        op: Op::Add(x, x),
        ty: VarType::F32,
    });
    let z = ir.push_var(Var {
        op: Op::Store(y, buf_id),
        ty: VarType::F32,
    });

    let mut compiler = CUDACompiler::default();
    compiler.compile(&ir);

    let module = Module::from_ptx(
        &compiler.asm,
        &[
            ModuleJitOption::OptLevel(OptLevel::O4),
            ModuleJitOption::GenenerateDebugInfo(true),
            ModuleJitOption::GenerateLineInfo(true),
        ],
    )
    .unwrap();

    let func = module.get_function("cujit").unwrap();

    let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();

    let grid_size = (size as u32 + block_size - 1) / block_size;

    let stream = cust::stream::Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None).unwrap();

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

    dbg!(&x_buf);
    dbg!(x_buf.as_host_vec());
}

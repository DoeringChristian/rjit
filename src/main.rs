use std::ffi::c_void;

use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::Module;
use cust::util::{DeviceCopyExt, SliceExt};

use self::compiler::Compiler;
use self::ir::{Ir, Op, Var, VarId, VarType};

mod compiler;
mod ir;
mod iterators;

fn main() {
    let ctx = cust::quick_init().unwrap();
    let device = cust::device::Device::get_device(0).unwrap();

    let mut ir = Ir::default();

    let size = 10;
    let x = vec![1f32; size].as_slice().as_dbuf().unwrap();
    dbg!(&x);
    let x = x.cast::<u8>();
    dbg!(&x);

    let buf_id = ir.push_buf(x);

    let x = ir.push_var(Var {
        op: Op::Load(buf_id),
        ty: VarType::F32,
        // id: VarId(0),
    });
    let c = ir.push_var(Var {
        op: Op::ConstF32(2.),
        ty: VarType::F32,
    });
    let y = ir.push_var(Var {
        op: Op::Add(x, x),
        ty: VarType::F32,
        // id: VarId(0),
    });
    let z = ir.push_var(Var {
        op: Op::Store(y, buf_id),
        ty: VarType::F32,
    });

    let mut compiler = Compiler::default();
    compiler.compile(&ir);

    // dbg!(ir.buffers()[0].as_device_ptr().as_raw());

    let mut params = vec![(size as u64).to_le()];
    params.extend(
        ir.buffers()
            .iter()
            .map(|buf| buf.as_device_ptr().as_raw().to_le()),
    );
    println!("{:#018x?}", ir.buffers()[0].as_device_ptr().as_raw());

    // println!("{:#x?}", params.as_host_vec().unwrap()[0]);

    let module = Module::from_ptx(
        &compiler.asm,
        &[
            ModuleJitOption::OptLevel(OptLevel::O0),
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
                &[params.as_mut_ptr() as *mut std::ffi::c_void],
            )
            .unwrap();
    }

    dbg!(params);

    stream.synchronize().unwrap();

    let y = ir.buffers.pop().unwrap().cast::<f32>();
    dbg!(&y);
    dbg!(y.as_host_vec());
}

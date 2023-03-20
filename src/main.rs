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

    let x = vec![0.; 100].as_slice().as_dbuf().unwrap();
    dbg!(&x);
    let x = x.cast::<u8>();
    dbg!(&x);

    let buf_id = ir.push_buf(x);

    let x = ir.push_var(Var {
        op: Op::Load(buf_id),
        ty: VarType::F32,
        // id: VarId(0),
    });
    let y = ir.push_var(Var {
        op: Op::Add(x, x),
        ty: VarType::F32,
        // id: VarId(0),
    });
    let z = ir.push_var(Var {
        op: Op::Store(buf_id),
        ty: VarType::F32,
    });

    let mut compiler = Compiler::default();
    compiler.compile(&ir);

    let params = ir
        .buffers()
        .iter()
        .map(|buf| buf.as_device_ptr().as_raw())
        .collect::<Vec<_>>()
        .as_slice()
        .as_dbuf()
        .unwrap();

    let module = Module::from_ptx(
        &compiler.asm,
        &[
            ModuleJitOption::OptLevel(OptLevel::O4),
            ModuleJitOption::GenenerateDebugInfo(false),
            ModuleJitOption::GenerateLineInfo(false),
        ],
    )
    .unwrap();

    let func = module.get_function("cujit").unwrap();

    let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();

    let grid_size = (100 + block_size - 1) / block_size;

    let stream = cust::stream::Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None).unwrap();

    unsafe {
        cust::launch!(func <<< grid_size, block_size, 0, stream >>> (params.as_device_ptr()))
            .unwrap();
    };

    stream.synchronize().unwrap();

    // dbg!(ir.buffers());
    // let y = ir.buffers()[0].as_host_vec().unwrap();
    // let y = ir
    //     .buffers
    //     .pop()
    //     .unwrap()
    //     .cast::<f32>()
    //     .as_host_vec()
    //     .unwrap();

    let y = ir.buffers.pop().unwrap().cast::<f32>();
    dbg!(&y);

    let x = vec![0.; 100].as_slice().as_dbuf().unwrap();
    dbg!(&x);
    let x = x.cast::<u8>();
    dbg!(&x);
    let x = x.cast::<f32>();
    dbg!(&x);

    // dbg!(x.cast::<f32>().as_host_vec());

    // let x = {
    //     let x = vec![0; 100];
    //     x.as_slice().as_dbuf().unwrap()
    // };
}

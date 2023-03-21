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
        reg: 0,
    });
    let c = ir.push_var(Var {
        op: Op::ConstF32(2.),
        ty: VarType::F32,
        reg: 0,
    });
    let y = ir.push_var(Var {
        op: Op::Add(x, x),
        ty: VarType::F32,
        reg: 0,
    });
    let z = ir.push_var(Var {
        op: Op::Store(y, buf_id),
        ty: VarType::F32,
        reg: 0,
    });

    let mut compiler = CUDACompiler::default();
    compiler.compile(&ir);
    compiler.execute(&mut ir);

    dbg!(&x_buf);
    dbg!(x_buf.as_host_vec());
}

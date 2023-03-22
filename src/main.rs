use std::ffi::c_void;
use std::num::NonZeroU64;
use std::sync::Arc;

use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::Module;
use cust::util::{DeviceCopyExt, SliceExt};
use smallvec::smallvec;

use crate::ir::ParamType;

use self::compiler::CUDACompiler;
use self::ir::{Ir, Op, Var, VarId, VarType};

mod compiler;
mod ir;
mod iterators;
mod trace;

fn main() {
    let ctx = cust::quick_init().unwrap();
    let device = cust::device::Device::get_device(0).unwrap();

    let mut ir = Ir::default();

    let size = 10;
    // let x_buf = Arc::new(vec![1f32; size].as_slice().as_dbuf().unwrap().cast::<u8>());
    // dbg!(&x_buf);

    let x = ir.buffer_f32(&[1.; 10]);
    let c = ir.const_f32(1.);
    let y = ir.add(x, x);

    let mut compiler = CUDACompiler::default();
    compiler.schedule = vec![y];
    compiler.preprocess(&mut ir);
    compiler.compile(&ir);
    compiler.execute(&mut ir);

    dbg!(&x_buf);
    dbg!(x_buf.as_host_vec());
}

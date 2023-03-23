use std::ffi::c_void;
use std::num::NonZeroU64;
use std::sync::Arc;

use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::Module;
use cust::util::{DeviceCopyExt, SliceExt};
use smallvec::smallvec;

use crate::trace::ParamType;

use self::compiler::CUDACompiler;
use self::jit::Jit;
use self::trace::{Ir, Op, Var, VarId, VarType};

mod backend;
mod compiler;
mod iterators;
mod jit;
mod schedule;
mod trace;

fn main() {
    let ctx = cust::quick_init().unwrap();

    let mut ir = Ir::default();

    let x = ir.buffer_f32(&[1.; 10]);
    let c = ir.const_f32(1.);
    let d = ir.const_f32(2.);
    let e = ir.add(x, d);
    let y = ir.add(x, c);

    ir.scheduled = vec![y];

    dbg!(&ir);

    // ir.dec_rc(d);

    dbg!(&ir);

    // ir.dec_rc(e);

    let mut jit = Jit::default();
    jit.eval(&mut ir);

    dbg!(ir.to_host_f32(y));
}

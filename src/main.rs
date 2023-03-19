use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::Module;
use cust::util::{DeviceCopyExt, SliceExt};

use self::compiler::Compiler;
use self::ir::{Ir, Op, Var, VarId, VarType};

mod compiler;
mod ir;
mod iterators;

fn main() {
    let mut ir = Ir::default();
    let x = ir.push_var(Var {
        op: Op::ConstF32(1.),
        ty: VarType::F32,
        // id: VarId(0),
    });
    let y = ir.push_var(Var {
        op: Op::Add(x, x),
        ty: VarType::F32,
        // id: VarId(0),
    });

    let mut compiler = Compiler::default();
    compiler.compile(&ir);

    let ctx = cust::quick_init().unwrap();
    let device = cust::device::Device::get_device(0).unwrap();

    let module = Module::from_ptx(
        &compiler.asm,
        &[
            ModuleJitOption::OptLevel(OptLevel::O4),
            ModuleJitOption::GenenerateDebugInfo(false),
            ModuleJitOption::GenerateLineInfo(false),
        ],
    )
    .unwrap();

    // let x = {
    //     let x = vec![0; 100];
    //     x.as_slice().as_dbuf().unwrap()
    // };
}

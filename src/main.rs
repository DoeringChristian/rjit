use std::sync::Arc;

use crate::backend::cuda::CUDABackend;
use crate::backend::Backend;
use crate::ir::IR;

use self::jit::Jit;

mod backend;
mod ir;
mod jit;
mod schedule;

fn main() {
    // pretty_env_logger::init();
    ir::set_backend("cuda");

    let r = {
        let x = ir::index(3);
        let y = ir::buffer_u32(&[1, 2, 3]);

        let z = ir::add(&x, &y);

        let r = ir::gather(&z, &ir::index(3), None);
        r
    };
    dbg!(&IR);

    jit::schedule(&[&r]);
    jit::eval();
    dbg!();

    assert_eq!(ir::to_vec_u32(&r), vec![1, 3, 5]);

    // let x = ir.buffer_f32(&[1.; 10]);
    //
    // let c = ir.const_f32(1.);
    // let d = ir.const_f32(2.);
    // let e = ir.add(x, d);
    // let y = ir.add(x, c);

    // ir.schedule(&[y]);
    //
    // jit.eval(&mut ir);
}

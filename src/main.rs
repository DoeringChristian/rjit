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
    dbg!(IR.is_locked());

    let x = ir::index(10);
    dbg!(IR.is_locked());

    let y = ir::index(3);

    let mut jit = Jit::default();
    jit.schedule(&[&y, &x]);
    jit.eval(&mut IR.lock());

    assert_eq!(ir::to_vec_u32(&x), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(ir::to_vec_u32(&y), vec![0, 1, 2]);
}

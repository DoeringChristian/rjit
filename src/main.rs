use std::sync::Arc;

use crate::backend::cuda::CUDABackend;
use crate::backend::Backend;
use crate::trace::IR;

use self::jit::Jit;

mod backend;
mod jit;
mod schedule;
mod trace;
mod var;

fn main() {
    // pretty_env_logger::init();
    //
    IR.set_backend("cuda");
    dbg!(IR.is_locked());

    let x = IR.index(10);
    dbg!(IR.is_locked());

    let y = IR.index(3);

    IR.schedule(&[&y, &x]);
    let mut jit = Jit::default();
    jit.eval(&mut IR.lock());

    assert_eq!(x.to_vec_u32(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(y.to_vec_u32(), vec![0, 1, 2]);
}

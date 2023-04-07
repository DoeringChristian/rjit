use crate::trace::Trace;
use crate::var::ReduceOp;

use self::jit::Jit;

mod backend;
mod jit;
mod schedule;
mod trace;
mod var;

fn main() {
    let IR = Trace::default();
    IR.set_backend("cuda");

    let x = IR.buffer_u32(&[0, 0, 0, 0]);

    let i = IR.buffer_u32(&[0, 0, 0]);

    let y = IR.const_u32(1);

    y.scatter_reduce(&x, &i, None, ReduceOp::Add);

    IR.schedule(&[&x]);

    let mut jit = Jit::default();
    jit.eval(&mut IR.lock());

    assert_eq!(x.to_vec_u32(), vec![3, 0, 0, 0]);
}

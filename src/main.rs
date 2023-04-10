use crate::trace::Trace;
use crate::var::ReduceOp;

use self::jit::Jit;

mod backend;
mod jit;
mod schedule;
mod trace;
mod var;

fn main() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.buffer_u32(&[0, 0, 0, 0]);

    let i = ir.buffer_u32(&[0, 0, 0]);

    let y = ir.const_u32(1);

    y.scatter_reduce(&x, &i, None, ReduceOp::Add);

    ir.schedule(&[&x]);

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    assert_eq!(x.to_vec_u32(), vec![3, 0, 0, 0]);
}

use crate::trace::Trace;

use self::jit::Jit;

mod backend;
mod jit;
mod schedule;
mod trace;
mod var;

fn main() {
    let IR = Trace::default();
    IR.set_backend("cuda");

    let x = IR.buffer_u32(&[0, 0, 0]);

    let i = IR.index(3);

    let y = IR.const_u32(2);

    y.scatter(&x, &i, None);

    let mut jit = Jit::default();
    jit.eval(&mut IR.lock());

    assert_eq!(x.to_vec_u32(), vec![2, 2, 2]);
}

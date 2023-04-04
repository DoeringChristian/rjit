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

    let x = IR.index(3);

    let y = IR.index(3);

    IR.schedule(&[&y, &x]);
    dbg!(&IR);
    let mut jit = Jit::default();
    jit.eval(&mut IR.lock());

    assert_eq!(x.to_vec_u32(), vec![0, 1, 2]);
    assert_eq!(y.to_vec_u32(), vec![0, 1, 2]);
}

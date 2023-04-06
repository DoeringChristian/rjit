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

    let x = IR.buffer_u32(&[0, 0, 0, 0]);

    let i = IR.index(3);
    let c = IR.const_u32(1);
    let i = i.add(&c);

    let y = IR.const_u32(2);

    y.scatter(&x, &i, None);

    let i = IR.index(2);

    let y = IR.const_u32(3);

    y.scatter(&x, &i, None);

    let c = IR.const_u32(1);
    let x = x.add(&c);

    IR.schedule(&[&x]);

    let mut jit = Jit::default();
    jit.eval(&mut IR.lock());

    assert_eq!(x.to_vec_u32(), vec![4, 4, 3, 3]);
}

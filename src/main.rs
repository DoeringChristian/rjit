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

    let tmp_x;
    let tmp_y;
    let tmp_z;
    let r = {
        let x = IR.index(3);
        dbg!();
        let y = IR.buffer_u32(&[1, 2, 3]);
        dbg!();

        // let z = ir::add(&x, &y);
        let z = x.add(&y);
        dbg!();

        let r = z.gather(&IR.index(3), None);
        dbg!();
        tmp_x = x.id();
        tmp_y = y.id();
        tmp_z = z.id();
        r
    };

    IR.schedule(&[&r]);
    let mut jit = Jit::default();
    jit.eval(&mut IR.lock());

    assert_eq!(r.to_vec_u32(), vec![1, 3, 5]);
}

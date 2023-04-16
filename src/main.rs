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

    let x = ir.buffer_f32(&[0.1, 0.2]);
    let y = ir.buffer_f32(&[0.1, 0.2]);

    let tex = ir.texture(&[10, 10]);

    let res = tex.tex_lookup(&[&x, &y]);

    let r = res[0].clone();

    r.schedule();

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    dbg!(r.to_host_f32());
}

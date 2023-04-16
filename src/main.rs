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

    let data = ir.literal_f32(1.);

    data.scatter(&tex, &ir.index(10 * 10 * 4), None);

    let res = tex.tex_lookup(&[&x, &y]);

    let r = res[0].clone();

    r.schedule();

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    println!("{}", jit.kernel_debug());

    dbg!(r.to_host_f32());
}

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

    let x = ir.literal_f32(0.0);
    let y = ir.literal_f32(0.0);

    let data = ir.buffer_f32(&[1.; 40000]);

    let tex = data.to_texture(&[100, 100]);

    let res = tex.tex_lookup(&[&x, &y]);

    let r = res[0].clone();

    r.schedule();

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    println!("{}", jit.kernel_debug());

    dbg!(r.to_host_f32());
}

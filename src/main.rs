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

    let x = ir.literal_f32(0.5);
    let y = ir.literal_f32(0.5);

    let data = ir.buffer_f32(&[1.; 400]);

    let tex = data.to_texture(&[10, 10]);

    let res = tex.tex_lookup(&[&x, &y]);

    let r = res[0].clone();

    r.schedule();

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    println!("{}", jit.kernel_debug());

    dbg!(r.to_host_f32());
}

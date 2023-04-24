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

    let bools = ir.buffer_bool(&[true, true, true, true, true, true, true, true]);

    dbg!(bools.to_host_bool());

    let idx = bools.compress();

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    println!("{}", jit.kernel_debug());

    dbg!(idx.to_host_u32());
}

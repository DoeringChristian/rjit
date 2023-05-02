use crate::trace::Trace;
use crate::var::ReduceOp;

use self::jit::Jit;

mod backend;
mod jit;
mod schedule;
mod trace;
mod var;

fn main() {
    let backend = backend::optix::optix::Backend::new().unwrap();
}

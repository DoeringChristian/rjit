use std::sync::Arc;

use crate::backend::cuda::CUDABackend;
use crate::backend::Backend;

use self::jit::Jit;
use self::trace::Ir;

mod backend;
mod jit;
mod schedule;
mod trace;

fn main() {
    let backend: Arc<dyn Backend> = Arc::new(CUDABackend::new());

    let mut jit = Jit::new(&backend);

    let mut ir = Ir::new(&backend);

    let x = ir.buffer_f32(&[1.; 10]);
    let c = ir.const_f32(1.);
    let d = ir.const_f32(2.);
    let e = ir.add(x, d);
    let y = ir.add(x, c);

    ir.scheduled = vec![y];

    jit.eval(&mut ir);
    dbg!(&jit);
}

use std::sync::Arc;

use crate::backend::cuda::CUDABackend;
use crate::backend::Backend;
use crate::trace::Trace;

use self::jit::Jit;
use self::trace::Ir;

mod backend;
mod jit;
mod schedule;
mod trace;

fn main() {
    let backend: Arc<dyn Backend> = Arc::new(CUDABackend::new());
    let mut ir = Trace::new(&backend);

    let i = ir.index(10);

    ir.schedule(&[i.clone()]);

    ir.eval();

    assert_eq!(ir.to_vec_u32(&i), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    // let x = ir.buffer_f32(&[1.; 10]);
    // let c = ir.const_f32(1.);
    // let d = ir.const_f32(2.);
    // let e = ir.add(x, d);
    // let y = ir.add(x, c);

    // ir.schedule(&[y]);
    //
    // jit.eval(&mut ir);
}

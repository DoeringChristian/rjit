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
    let ir = Trace::new(&backend);

    let y = {
        let x = ir.index(10);

        let i = ir.index(3);
        dbg!(i.id());
        let c = ir.const_u32(2);
        let i = ir.add(&i, &c);

        let y = ir.gather(&x, &i, None);
        y
    };

    ir.schedule(&[&y]);
    ir.eval();

    assert_eq!(ir.to_vec_u32(&y), vec![2, 3, 4]);

    // let x = ir.buffer_f32(&[1.; 10]);
    // let c = ir.const_f32(1.);
    // let d = ir.const_f32(2.);
    // let e = ir.add(x, d);
    // let y = ir.add(x, c);

    // ir.schedule(&[y]);
    //
    // jit.eval(&mut ir);
}

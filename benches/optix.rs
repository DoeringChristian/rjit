use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rjit::*;

fn kernel_reuse(ir: &Trace) -> Result<()> {
    for _ in 0..1000 {
        let mut x = ir.array(&[0, 1, 2, 3, 4])?;
        for j in 0..100 {
            x = x.add(&ir.literal(j)?)?;
        }

        x.schedule();
        ir.eval()?;
    }

    assert_eq!(ir.n_variables(), 0); // Assert that trace is clean for next benchmark

    Ok(())
}
fn no_kernel_reuse(ir: &Trace) -> Result<()> {
    for i in 0..1000 {
        let mut x = ir.array(&(0..i + 1).collect::<Vec<_>>())?;
        for j in 0..100 {
            x = x.add(&ir.literal(j)?)?;
        }

        x.schedule();
        ir.eval()?;
    }

    assert_eq!(ir.n_variables(), 0); // Assert that trace is clean for next benchmark

    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    let ir = Trace::default();
    ir.set_backend(["optix"]).unwrap();
    c.bench_function("OptiX kernel reuse", |b| {
        b.iter(|| black_box(kernel_reuse(&ir)))
    });
    c.bench_function("OptiX no kernel reuse", |b| {
        b.iter(|| black_box(no_kernel_reuse(&ir)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

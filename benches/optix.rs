use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rjit::*;

fn optix_addition() -> Result<()> {
    let trace = Trace::default();
    trace.set_backend(&["optix"])?;

    let mut x = trace.array(&[1f32, 2., 3., 4., 5.])?;

    for i in 0..1000 {
        x = x.add(&trace.literal(i)?)?;
    }

    x.schedule();
    trace.eval()?;

    let result = x.to_host::<f32>()?;
    drop(result);

    Ok(())
}

fn optix_multiple_addition() -> Result<()> {
    let trace = Trace::default();
    trace.set_backend(&["optix"])?;

    let results = (0..1000)
        .into_iter()
        .map(|i| {
            let init = (0..1000).into_iter().map(|j| j as f32).collect::<Vec<_>>();
            let mut x = trace.array(&init)?;

            for i in 0..100 {
                x = x.add(&trace.literal(i)?)?;
            }
            x.schedule();
            trace.eval()?;
            Ok(())
        })
        .collect::<Result<Vec<_>>>();

    drop(results);

    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("OptiX addition", |b| b.iter(|| black_box(optix_addition())));
    c.bench_function("OptiX multiple addition", |b| {
        b.iter(|| black_box(optix_multiple_addition()))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

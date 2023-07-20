use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rjit::*;

fn trace_ray(ir: &Trace) -> Result<()> {
    let miss_and_closesthit_ptx = r##"
.version 8.0
.target sm_86
.address_size 64

.entry __miss__ms() {
	.reg .b32 %r<6>;
	mov.b32 %r0, 0;
	mov.b32 %r1, 0;
        
	call _optix_set_payload, (%r0, %r1);
	ret;
}

.entry __closesthit__ch() {
	.reg .b32 %i<5>;
	.reg .b32 %v<5>;
	mov.b32 %i0, 0;
	mov.b32 %i1, 1;
	mov.b32 %i2, 2;
	mov.b32 %i3, 3;
	mov.b32 %i4, 4;

        mov.b32 %v0, 1;
	call _optix_set_payload, (%i0, %v0);
        
        call (%v1), _optix_read_primitive_idx, ();
	call _optix_set_payload, (%i1, %v1);
        
        call (%v2), _optix_read_instance_id, ();
	call _optix_set_payload, (%i2, %v2);
        
	.reg .f32 %f<2>;
        call (%f0, %f1), _optix_get_triangle_barycentrics, ();
        mov.b32 %v3, %f0;
        mov.b32 %v4, %f1;
	call _optix_set_payload, (%i3, %v3);
	call _optix_set_payload, (%i4, %v4);
        
	ret;
}
"##;
    let indices = ir.array(&[0u32, 1, 2])?;
    let vertices = ir.array(&[1.0f32, 0., 1., 0., 1., 1., 1., 1., 1.])?;

    let desc = AccelDesc {
        sbt: crate::SBTDesc {
            hit_groups: &[HitGroupDesc {
                closest_hit: crate::ModuleDesc {
                    asm: miss_and_closesthit_ptx,
                    entry_point: "__closesthit__ch",
                },
                ..Default::default()
            }],
            miss_groups: &[MissGroupDesc {
                miss: crate::ModuleDesc {
                    asm: miss_and_closesthit_ptx,
                    entry_point: "__miss__ms",
                },
            }],
        },
        geometries: &[GeometryDesc::Triangles {
            vertices: &vertices,
            indices: &indices,
        }],
        instances: &[InstanceDesc {
            hit_group: 0,
            geometry: 0,
            transform: [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
        }],
    };
    let accel = ir.accel(desc)?;

    let payload = accel.trace_ray(
        &[
            &ir.literal(0u32)?,
            &ir.literal(0u32)?,
            &ir.literal(0u32)?,
            &ir.literal(0u32)?,
            &ir.literal(0u32)?,
        ],
        [
            &ir.array(&[0.6f32, 0.6])?,
            &ir.literal(0.6f32)?,
            &ir.literal(0.0f32)?,
        ],
        [
            &ir.literal(0.0f32)?,
            &ir.literal(0.0f32)?,
            &ir.array(&[1.0f32, -1.])?,
        ],
        &ir.literal(0.001f32)?,
        &ir.literal(1000.0f32)?,
        &ir.literal(0.0f32)?,
        None,
        None,
        None,
        None,
        None,
        None,
    )?;

    let valid = payload[0].cast(&VarType::Bool)?;

    let u = payload[3].bitcast(&VarType::F32)?;
    let v = payload[4].bitcast(&VarType::F32)?;

    valid.schedule();
    v.schedule();
    u.schedule();

    ir.eval()?;

    Ok(())
}

fn kernel_reuse(ir: &Trace) -> Result<()> {
    for _ in 0..100 {
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
    for i in 0..100 {
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
    c.bench_function("OptiX Trace Ray", |b| b.iter(|| black_box(trace_ray(&ir))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

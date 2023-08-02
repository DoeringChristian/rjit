use std::time::Instant;

use crate::backend::CompileOptions;
use crate::jit::Jit;
use crate::trace::{ReduceOp, Trace, VarType};
use crate::{backend, AccelDesc, GeometryDesc, HitGroupDesc, InstanceDesc, MissGroupDesc};
use anyhow::Result;

#[test]
fn refcounting() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.array(&[1f32; 10])?;
    assert_eq!(x.var().rc, 1, "rc of x should be 1 (in x)");
    let y = x.add(&x)?;
    // let y = ir::add(&x, &x);

    assert_eq!(
        x.var().rc,
        3,
        "rc of x should be 3 (2 refs in y and 1 in x)"
    );

    assert_eq!(y.var().rc, 1, "rc of y should be 1 (in y)");

    ir.schedule(&[&y]);

    assert_eq!(
        y.var().rc,
        2,
        "rc of y should be 2 (1 in y and 1 in schedule)"
    );

    ir.eval()?;

    assert_eq!(
        x.var().rc,
        1,
        "rc of x should be 1 (dependencies of y shuld be cleaned)"
    );
    assert_eq!(
        y.var().rc,
        1,
        "rc of y should be 2 (y from schedule should be cleaned)"
    );

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host::<f32>().unwrap(), vec![2f32; 10]);
    Ok(())
}
#[test]
fn load_add_f32() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.array(&[1f32; 10])?;
    // let y = ir::add(&x, &x);
    let y = x.add(&x)?;

    ir.schedule(&[&y]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host::<f32>().unwrap(), vec![2f32; 10]);
    Ok(())
}
#[test]
fn load_gather_f32() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.array(&[1f32, 2., 3., 4., 5.])?;
    let i = ir.array(&[0u32, 1, 4])?;
    let y = x.gather(&i, None)?;

    ir.schedule(&[&y]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host::<f32>().unwrap(), vec![1., 2., 5.]);
    Ok(())
}
#[test]
fn reindex() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.index(10);

    let i = ir.index(3);
    let c = ir.literal(2u32)?;
    let i = i.add(&c)?;

    let y = x.gather(&i, None)?;

    ir.schedule(&[&y]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host::<u32>().unwrap(), vec![2, 3, 4]);
    Ok(())
}
#[test]
fn index() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let i = ir.index(10);

    ir.schedule(&[&i]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(
        i.to_host::<u32>().unwrap(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    Ok(())
}
#[test]
fn gather_eval() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let r = {
        let x = ir.index(3);
        let y = ir.array(&[1u32, 2, 3])?;

        // let z = ir::add(&x, &y);
        let z = x.add(&y)?;

        let r = z.gather(&ir.index(3), None)?;
        r
    };
    // schedule

    ir.schedule(&[&r]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(r.to_host::<u32>().unwrap(), vec![1, 3, 5]);
    Ok(())
}
#[test]
fn paralell() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.index(10);

    let y = ir.index(3);

    ir.schedule(&[&x, &y]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(
        x.to_host::<u32>().unwrap(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    assert_eq!(y.to_host::<u32>().unwrap(), vec![0, 1, 2]);
    Ok(())
}
#[test]
fn load_gather() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.array(&[1.0f32, 2., 3.])?;

    ir.schedule(&[&x]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<f32>().unwrap(), vec![1., 2., 3.]);
    Ok(())
}
#[test]
fn eval_scatter() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.array(&[0u32, 0, 0, 0])?;
    let c = ir.literal(1u32)?;
    let x = x.add(&c)?; // x: [1, 1, 1, 1]

    let i = ir.index(3);
    let c = ir.literal(1u32)?;
    let i = i.add(&c)?; // i: [1, 2, 3]

    let y = ir.literal(2u32)?;

    y.scatter(&x, &i, None)?; // x: [1, 2, 2, 2]
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<u32>().unwrap(), vec![1, 2, 2, 2]);
    Ok(())
}
#[test]
fn scatter_twice() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.array(&[0u32, 0, 0, 0])?;

    let i = ir.index(3);
    let c = ir.literal(1u32)?;
    let i = i.add(&c)?;

    let y = ir.literal(2u32)?;

    y.scatter(&x, &i, None)?;

    let i = ir.index(2);

    let y = ir.literal(3u32)?;

    y.scatter(&x, &i, None)?;
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<u32>().unwrap(), vec![3, 3, 2, 2]);
    Ok(())
}
#[test]
fn scatter_twice_add() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.array(&[0u32, 0, 0, 0])?;

    let i = ir.index(3);
    let c = ir.literal(1u32)?;
    let i = i.add(&c)?;

    let y = ir.literal(2u32)?;

    y.scatter(&x, &i, None)?;

    let i = ir.index(2);

    let y = ir.literal(3u32)?;

    y.scatter(&x, &i, None)?;

    let c = ir.literal(1u32)?;
    let x = x.add(&c)?;

    ir.schedule(&[&x]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<u32>().unwrap(), vec![4, 4, 3, 3]);
    Ok(())
}
#[test]
fn scatter_reduce() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.array(&[0u32, 0, 0, 0])?;

    let i = ir.array(&[0u32, 0, 0])?;

    let y = ir.literal(1u32)?;

    y.scatter_reduce(&x, &i, None, ReduceOp::Add)?;

    ir.eval()?;

    // dbg!(&ir);

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<u32>().unwrap(), vec![3, 0, 0, 0]);
    Ok(())
}
#[test]
fn tex_lookup() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.array(&[0.5f32])?;
    let y = ir.array(&[0.5f32])?;

    let data = ir.array(&[1.0f32; 400])?;

    let tex = data.to_texture(&[10, 10], 4)?;

    let res = tex.tex_lookup(&[&x, &y])?;

    let r = res[0].clone();

    r.schedule();

    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(r.to_host::<f32>().unwrap(), vec![1.]);
    Ok(())
}
#[test]
fn trace_ray() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

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

    // for p in payload.iter() {
    //     p.schedule();
    // }
    let valid = payload[0].cast(&VarType::Bool)?;

    let u = payload[3].bitcast(&VarType::F32)?;
    let v = payload[4].bitcast(&VarType::F32)?;

    let dst = ir.array(&[0f32, 1f32])?;
    u.scatter(&dst, &ir.index(2), None)?;

    valid.schedule();
    v.schedule();
    u.schedule();

    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(valid.to_host::<bool>().unwrap(), vec![true, false]);
    let u = dst.to_host::<f32>().unwrap();
    let v = v.to_host::<f32>().unwrap();
    approx::assert_ulps_eq!(u[0], 0.39999998);
    approx::assert_ulps_eq!(u[1], 0.);
    approx::assert_ulps_eq!(v[0], 0.20000005);
    approx::assert_ulps_eq!(v[1], 0.);
    Ok(())
}
#[test]
fn trace_ray_scatter() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

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

    let dst = ir.array(&[0f32, 0f32])?;
    u.scatter(&dst, &ir.index(2), Some(&valid))?;

    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    // assert_eq!(valid.to_host_bool(), vec![true, false]);
    let u = dst.to_host::<f32>().unwrap();
    approx::assert_ulps_eq!(u[0], 0.39999998);
    approx::assert_ulps_eq!(u[1], 0.);
    Ok(())
}
#[test]
fn trace_ray_multi() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let mut times = vec![];

    for i in 0..100 {
        let start = Instant::now();
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

        times.push(start.elapsed());
    }

    Ok(())
}
#[test]
fn trace_ray_hit_group() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;
    // pretty_env_logger::init();

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

.entry __closesthit__ch1() {
	.reg .b32 %i<5>;
	.reg .b32 %v<5>;
	mov.b32 %i0, 0;
	mov.b32 %i1, 1;
	mov.b32 %i2, 2;
	mov.b32 %i3, 3;
	mov.b32 %i4, 4;

        mov.b32 %v0, 1;
	call _optix_set_payload, (%i0, %v0);
        
	ret;
}

.entry __closesthit__ch2() {
	.reg .b32 %i<5>;
	.reg .b32 %v<5>;
	mov.b32 %i0, 0;
	mov.b32 %i1, 1;
	mov.b32 %i2, 2;
	mov.b32 %i3, 3;
	mov.b32 %i4, 4;

        mov.b32 %v0, 2;
	call _optix_set_payload, (%i0, %v0);
        
	ret;
}
"##;
    let indices0 = ir.array(&[0u32, 1, 2])?;
    let vertices0 = ir.array(&[1.0f32, 0., 1., 0., 1., 1., 0., 0., 1.])?;
    let indices1 = ir.array(&[0u32, 1, 2])?;
    let vertices1 = ir.array(&[1.0f32, 0., 1., 0., 1., 1., 1., 1., 1.])?;

    let desc = AccelDesc {
        sbt: crate::SBTDesc {
            hit_groups: &[
                HitGroupDesc {
                    closest_hit: crate::ModuleDesc {
                        asm: miss_and_closesthit_ptx,
                        entry_point: "__closesthit__ch1",
                    },
                    ..Default::default()
                },
                HitGroupDesc {
                    closest_hit: crate::ModuleDesc {
                        asm: miss_and_closesthit_ptx,
                        entry_point: "__closesthit__ch2",
                    },
                    ..Default::default()
                },
            ],
            miss_groups: &[MissGroupDesc {
                miss: crate::ModuleDesc {
                    asm: miss_and_closesthit_ptx,
                    entry_point: "__miss__ms",
                },
            }],
        },
        geometries: &[
            GeometryDesc::Triangles {
                vertices: &vertices0,
                indices: &indices0,
            },
            GeometryDesc::Triangles {
                vertices: &vertices1,
                indices: &indices1,
            },
        ],
        instances: &[
            InstanceDesc {
                hit_group: 0,
                geometry: 0,
                transform: [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            },
            InstanceDesc {
                hit_group: 1,
                geometry: 1,
                transform: [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            },
        ],
    };
    let accel = ir.accel(desc)?;

    let payload = accel.trace_ray(
        &[&ir.literal(0u32)?],
        [
            // &ir.array(&[0.5f32, 0.5])?,
            &ir.literal(0.5f32)?,
            &ir.literal(0.5f32)?,
            &ir.literal(0.0f32)?,
        ],
        [
            &ir.array(&[-0.2f32, 0.2])?,
            &ir.array(&[-0.2f32, 0.2])?,
            &ir.literal(1f32)?,
            // &ir.array(&[1.0f32, 1.])?,
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

    let hit_id = payload[0].cast(&VarType::U32)?;
    hit_id.schedule();

    ir.eval()?;

    assert_eq!(hit_id.to_host::<u32>()?, [1, 2]);

    Ok(())
}
#[test]
fn trace_ray_shadowed() -> Result<()> {
    pretty_env_logger::try_init().ok();
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

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
.entry __miss__shadow() {
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
        
	ret;
}

"##;
    let indices0 = ir.array(&[0u32, 1, 2])?;
    let vertices0 = ir.array(&[1.0f32, 0., 1., 0., 1., 1., 0., 0., 1.])?;

    let desc = AccelDesc {
        sbt: crate::SBTDesc {
            hit_groups: &[HitGroupDesc {
                closest_hit: crate::ModuleDesc {
                    asm: miss_and_closesthit_ptx,
                    entry_point: "__closesthit__ch",
                },
                ..Default::default()
            }],
            miss_groups: &[
                MissGroupDesc {
                    miss: crate::ModuleDesc {
                        asm: miss_and_closesthit_ptx,
                        entry_point: "__miss__ms",
                    },
                },
                MissGroupDesc {
                    miss: crate::ModuleDesc {
                        asm: miss_and_closesthit_ptx,
                        entry_point: "__miss__shadow",
                    },
                },
            ],
        },
        geometries: &[GeometryDesc::Triangles {
            vertices: &vertices0,
            indices: &indices0,
        }],
        instances: &[InstanceDesc {
            hit_group: 0,
            geometry: 0,
            transform: [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
        }],
    };
    let accel = ir.accel(desc)?;

    let payload = accel.trace_ray(
        &[&ir.literal(0u32)?],
        [
            // &ir.array(&[0.5f32, 0.5])?,
            &ir.literal(0.5f32)?,
            &ir.literal(0.5f32)?,
            &ir.literal(0.0f32)?,
        ],
        [
            &ir.array(&[-0.2f32, 0.2])?,
            &ir.array(&[-0.2f32, 0.2])?,
            &ir.literal(1f32)?,
            // &ir.array(&[1.0f32, 1.])?,
        ],
        &ir.literal(0.001f32)?,
        &ir.literal(1000.0f32)?,
        &ir.literal(0.0f32)?,
        None,
        Some(&ir.literal(4u32)?),
        None,
        None,
        Some(&ir.literal(1u32)?),
        None,
    )?;

    let shadowed = payload[0].cast(&VarType::U32)?;
    shadowed.schedule();

    ir.eval()?;

    assert_eq!(shadowed.to_host::<u32>()?, [1, 0]);

    Ok(())
}
#[test]
fn sized_literal() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    let x = ir.sized_literal(0f32, 10)?;
    let x = x.add(&ir.literal(0f32)?)?;
    x.schedule();

    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<f32>().unwrap(), vec![0f32; 10]);
    Ok(())
}

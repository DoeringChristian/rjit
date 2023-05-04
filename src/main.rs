use std::sync::Arc;

use crate::backend::optix::CompileOptions;
use crate::trace::Trace;
use crate::var::ReduceOp;

use self::backend::optix::optix_core;
use self::jit::Jit;
use self::var::VarType;

mod backend;
mod jit;
mod schedule;
mod trace;
mod var;

fn main() {
    pretty_env_logger::init();

    let ir = Trace::default();
    ir.set_backend("optix");

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

    {
        let mut backend = ir.backend_as::<backend::optix::Backend>();

        backend.compile_options = CompileOptions {
            num_payload_values: 5,
        };
        backend.set_miss_from_str(("__miss__ms", miss_and_closesthit_ptx));
        backend.set_hit_from_strs(&[("__closesthit__ch", miss_and_closesthit_ptx)]);
    }
    let indices = ir.buffer_u32(&[0, 1, 2]);
    let vertices = ir.buffer_f32(&[1., 0., 1., 0., 1., 1., 1., 1., 1.]);

    let accel = ir.accel(&vertices, &indices);

    let payload = accel.trace_ray(
        5,
        [
            &ir.buffer_f32(&[0.6, 0.6]),
            &ir.literal_f32(0.6),
            &ir.literal_f32(0.),
        ],
        [
            &ir.literal_f32(0.),
            &ir.literal_f32(0.),
            &ir.buffer_f32(&[1., -1.]),
        ],
        &ir.literal_f32(0.001),
        &ir.literal_f32(1000.),
        &ir.literal_f32(0.),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    // for p in payload.iter() {
    //     p.schedule();
    // }
    let valid = payload[0].cast(&VarType::Bool);
    valid.schedule();

    let u = payload[3].bitcast(&VarType::F32);
    let v = payload[4].bitcast(&VarType::F32);
    u.schedule();
    v.schedule();

    let mut jit = Jit::default();
    jit.eval(&mut ir.lock());

    dbg!(valid.to_host_bool());
    dbg!(u.to_host_f32());
    dbg!(v.to_host_f32());
}

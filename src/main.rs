use std::sync::Arc;

use crate::trace::Trace;
use crate::var::ReduceOp;

use self::backend::optix::optix_core;
use self::jit::Jit;

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
	.reg .b32 %r<1>;
	mov.b32 %r0, 0;
	call _optix_set_payload, (%r0, %r0);
	ret;
}

.entry __closesthit__ch() {
	.reg .b32 %r<2>;
	mov.b32 %r0, 0;
	mov.b32 %r1, 1;
	call _optix_set_payload, (%r0, %r1);
	ret;
}
"##;

    {
        let mut backend = ir.backend();

        let backend_ref = backend
            .as_any_mut()
            .downcast_mut::<backend::optix::Backend>()
            .unwrap();

        *backend_ref.pipeline_state.lock().unwrap() = backend::optix::PipelineDesc {
            mco: optix_rs::OptixModuleCompileOptions {
                optLevel:
                    optix_rs::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
                debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
                ..Default::default()
            },
            pco: optix_rs::OptixPipelineCompileOptions {
                numAttributeValues: 0,
                pipelineLaunchParamsVariableName: b"params\0" as *const _ as *const _,
                exceptionFlags: optix_rs::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE as _,
                numPayloadValues: 1,
                ..Default::default()
            },
            miss: ("__miss__ms".into(), miss_and_closesthit_ptx.into()),
            hit: vec![("__closesthit__ch".into(), miss_and_closesthit_ptx.into())],
        };
    }
    let indices = ir.buffer_u32(&[1, 2, 3]);
    let vertices = ir.buffer_f32(&[1., 0., 1., 0., 1., 1., 1., 1., 1.]);

    let accel = ir.accel(&vertices, &indices);

    let payload = accel.trace_ray(
        1,
        [
            &ir.buffer_f32(&[0.5, 0.5]),
            &ir.literal_f32(0.5),
            &ir.literal_f32(0.),
        ],
        [
            &ir.literal_f32(0.),
            &ir.literal_f32(0.),
            &ir.literal_f32(1.),
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

    ir.schedule(&[&payload[0]]);

    // ir.schedule(&[&x]);
    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    // assert_eq!(x.to_host_u32(), vec![2, 4, 6]);

    // // let backend = backend::optix::optix::Backend::new().unwrap();
    // let instance = Arc::new(optix_core::Instance::new().unwrap());
    // let device = optix_core::Device::create(&instance, 0).unwrap();
    //
    // let miss_minimal = ".version 6.0 .target sm_50 .address_size 64 \
    //                      .entry __miss__dr() { ret; }";
    //
    // let mco = optix_rs::OptixModuleCompileOptions {
    //     optLevel: optix_rs::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
    //     debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
    //     ..Default::default()
    // };
    // let pco = optix_rs::OptixPipelineCompileOptions {
    //     numAttributeValues: 2,
    //     pipelineLaunchParamsVariableName: b"params" as *const _ as *const _,
    //     exceptionFlags: optix_rs::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE as _,
    //     ..Default::default()
    // };
    // let miss = optix_core::Module::create(&device, miss_minimal, mco, pco).unwrap();
    //
    // let miss_group = optix_core::ProgramGroup::create(
    //     &device,
    //     optix_core::ProgramGroupDesc::Miss {
    //         module: &miss,
    //         entry_point: "__miss__dr",
    //     },
    // )
    // .unwrap();
    //
    // drop(miss);
    // drop(miss_group);
}

use std::ffi::{CStr, CString};
use std::fmt::{Debug, Write};
use std::sync::Arc;

use crate::backend;
use crate::backend::cuda::{cuda_core, Buffer, Texture};
use crate::schedule::{Env, SVarId, ScheduleIr};
use crate::trace::VarType;
use crate::var::ParamType;
use optix_rs::{
    OptixAccelBuildOptions, OptixApi, OptixBuildFlags, OptixBuildInput,
    OptixBuildInputTriangleArray, OptixBuildInputType, OptixBuildInput__bindgen_ty_1,
    OptixBuildOperation, OptixDeviceContext, OptixDeviceContextOptions, OptixExceptionFlags,
    OptixModuleCompileOptions, OptixPipelineCompileOptions, OptixProgramGroup,
    OptixProgramGroupDesc, OptixVertexFormat,
};
use thiserror::Error;

use super::optix_core::{self, Device};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    OptixError(#[from] optix_core::Error),
}

pub struct Backend {
    instance: Arc<optix_core::Instance>,
    device: optix_core::Device,
    stream: Arc<cuda_core::Stream>,
}
impl Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backend").finish()
    }
}

impl Backend {
    pub fn new() -> Result<Self, Error> {
        let instance = Arc::new(optix_core::Instance::new()?);
        let device = optix_core::Device::create(&instance, 0)?;
        let stream = Arc::new(
            device
                .cuda_device()
                .create_stream(cuda_rs::CUstream_flags_enum::CU_STREAM_DEFAULT)
                .unwrap(),
        );
        Ok(Self {
            device,
            instance,
            stream,
        })
    }
}

impl backend::Backend for Backend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn new_kernel(&self) -> Box<dyn backend::Kernel> {
        Box::new(Kernel {
            device: self.device.clone(),
            stream: self.stream.clone(),
            asm: "".into(),
            entry_point: "__raygen__cujit".into(),
            pipeline: None,
        })
    }

    fn create_texture(&self, shape: &[usize], n_channels: usize) -> Arc<dyn backend::Texture> {
        Arc::new(Texture::create(
            self.device.cuda_device(),
            shape,
            n_channels,
        ))
    }

    fn buffer_uninit(&self, size: usize) -> Arc<dyn backend::Buffer> {
        Arc::new(Buffer::uninit(self.device.cuda_device(), size))
    }

    fn buffer_from_slice(&self, slice: &[u8]) -> Arc<dyn backend::Buffer> {
        Arc::new(Buffer::from_slice(self.device.cuda_device(), slice))
    }

    fn first_register(&self) -> usize {
        Kernel::FIRST_REGISTER
    }

    fn synchronize(&self) {
        self.stream.synchronize().unwrap();
    }
}

pub struct Kernel {
    device: Device,
    pub asm: String,
    entry_point: String,
    pipeline: Option<optix_core::Pipeline>,
    stream: Arc<cuda_core::Stream>,
}
impl Kernel {
    const FIRST_REGISTER: usize = 4;
}
impl Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("device", &self.device)
            .field("asm", &self.asm)
            .field("kernel_name", &self.entry_point)
            .finish()
    }
}

impl backend::Kernel for Kernel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[allow(unused_must_use)]
    fn assemble(&mut self, ir: &crate::schedule::ScheduleIr, env: &crate::schedule::Env) {
        self.asm.clear();

        super::codegen::assemble_entry(&mut self.asm, ir, env, &self.entry_point).unwrap();

        std::fs::write("/tmp/tmp.ptx", &self.asm).unwrap();

        log::trace!("{}", self.asm);
    }

    fn compile(&mut self) {
        let mco = OptixModuleCompileOptions {
            optLevel: optix_rs::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
            debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            ..Default::default()
        };
        let pco = OptixPipelineCompileOptions {
            numAttributeValues: 2,
            pipelineLaunchParamsVariableName: b"params\0" as *const _ as *const _,
            exceptionFlags: optix_rs::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE as _,
            ..Default::default()
        };

        let miss_minimal = ".version 6.0 .target sm_50 .address_size 64\n\
                                    .entry __miss__dr() { ret; }";

        let rgen = optix_core::Module::create(&self.device, &self.asm, mco, pco).unwrap();
        let miss = optix_core::Module::create(&self.device, miss_minimal, mco, pco).unwrap();
        let rgen_pg = optix_core::ProgramGroup::create(
            &self.device,
            optix_core::ProgramGroupDesc::RayGen {
                module: &rgen,
                entry_point: &self.entry_point,
            },
        )
        .unwrap();
        let miss_pg = optix_core::ProgramGroup::create(
            &self.device,
            optix_core::ProgramGroupDesc::Miss {
                module: &miss,
                entry_point: "__miss__dr",
            },
        )
        .unwrap();
        let pipeline = optix_core::Pipeline::create(
            &self.device,
            &pco,
            &optix_rs::OptixPipelineLinkOptions {
                maxTraceDepth: 1,
                debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            },
            rgen_pg,
            None,
            [miss_pg],
        )
        .unwrap();
        self.pipeline = Some(pipeline);
    }

    fn execute_async(&mut self, env: &mut crate::schedule::Env, size: usize) {
        let params = [size as u32, 0u32];
        let mut params = Vec::from(bytemuck::cast_slice(&params));
        params.extend(
            env.buffers()
                .iter()
                .map(|b| b.as_any().downcast_ref::<Buffer>().unwrap().ptr()),
        );
        params.extend(
            env.textures()
                .iter()
                .map(|b| b.as_any().downcast_ref::<Texture>().unwrap().ptr()),
        );

        log::trace!("params: {:02x?}", bytemuck::cast_slice::<_, u8>(&params));
        log::trace!("Optix Kernel Launch with {size} threads.");

        unsafe {
            let mut d_params = 0;
            let ctx = self.device.cuda_ctx();
            ctx.cuMemAlloc_v2(&mut d_params, 8 * params.len())
                .check()
                .unwrap();
            ctx.cuMemcpyHtoD_v2(d_params, params.as_ptr() as *const _, params.len() * 8)
                .check()
                .unwrap(); // TODO: Free somehow...

            self.pipeline
                .as_ref()
                .unwrap()
                .launch(&self.stream, d_params, params.len() * 8, size as u32)
                .unwrap();
            self.stream.synchronize().unwrap();

            ctx.cuMemFree_v2(d_params).check().unwrap();
        }
    }

    fn assembly(&self) -> &str {
        &self.asm
    }
}

pub struct Accel {
    device: Device,
}

impl Accel {
    pub fn create(device: &Device, vertices: &[&Arc<Buffer>]) -> Result<Self, Error> {
        let build_options = OptixAccelBuildOptions {
            buildFlags: OptixBuildFlags::OPTIX_BUILD_FLAG_NONE as _,
            operation: OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD,
            ..Default::default()
        };

        let vertices = <Buffer as backend::Buffer>::as_any(vertices[0])
            .downcast_ref::<Buffer>()
            .unwrap();

        let triangle_array = OptixBuildInputTriangleArray {
            vertexBuffers: todo!(),
            numVertices: todo!(),
            vertexFormat: OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3,
            vertexStrideInBytes: todo!(),
            indexBuffer: todo!(),
            numIndexTriplets: todo!(),
            indexFormat: todo!(),
            indexStrideInBytes: todo!(),
            preTransform: todo!(),
            flags: todo!(),
            numSbtRecords: todo!(),
            sbtIndexOffsetBuffer: todo!(),
            sbtIndexOffsetSizeInBytes: todo!(),
            sbtIndexOffsetStrideInBytes: todo!(),
            primitiveIndexOffset: todo!(),
            transformFormat: todo!(),
        };

        let mut build_input = OptixBuildInput {
            type_: OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            __bindgen_anon_1: OptixBuildInput__bindgen_ty_1::default(),
        };

        unsafe {
            <[u64]>::copy_from_slice(
                &mut build_input.__bindgen_anon_1.bindgen_union_field,
                std::slice::from_raw_parts(
                    &triangle_array as *const _ as *const _,
                    std::mem::size_of::<OptixBuildInputTriangleArray>() / 8,
                ),
            )
        };
        todo!()
    }
}

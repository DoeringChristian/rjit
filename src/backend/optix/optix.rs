use std::ffi::{CStr, CString};
use std::fmt::{Debug, Write};
use std::sync::{Arc, Mutex};

use crate::backend;
use crate::backend::cuda::cuda_core::Stream;
use crate::backend::cuda::{cuda_core, Buffer, Texture};
use crate::schedule::{Env, SVarId, ScheduleIr};
use crate::trace::VarType;
use crate::var::ParamType;
use optix_rs::{
    OptixAccelBufferSizes, OptixAccelBuildOptions, OptixAccelEmitDesc, OptixApi, OptixBuildFlags,
    OptixBuildInput, OptixBuildInputTriangleArray, OptixBuildInputType,
    OptixBuildInput__bindgen_ty_1, OptixBuildOperation, OptixDeviceContext,
    OptixDeviceContextOptions, OptixExceptionFlags, OptixGeometryFlags, OptixIndicesFormat,
    OptixModuleCompileOptions, OptixPipelineCompileOptions, OptixProgramGroup,
    OptixProgramGroupDesc, OptixProgramGroupOptions, OptixTraversableHandle, OptixVertexFormat,
};
use thiserror::Error;

use super::optix_core::{self, Device, Module, Pipeline, ProgramGroup};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    OptixError(#[from] optix_core::Error),
}

pub struct Backend {
    instance: Arc<optix_core::Instance>,
    device: optix_core::Device,
    stream: Arc<cuda_core::Stream>,
    pub compile_options: CompileOptions,
    pub miss: (String, Arc<Module>),
    pub hit: Vec<(String, Arc<Module>)>,
}
impl Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backend").finish()
    }
}

impl Backend {
    pub fn set_hit_from_strs(&mut self, hit: &[(&str, &str)]) {
        self.hit.extend(hit.iter().map(|(ep, ptx)| {
            (
                String::from(*ep),
                Arc::new(
                    Module::create(
                        &self.device,
                        ptx,
                        self.compile_options.mco(),
                        self.compile_options.pco(),
                    )
                    .unwrap(),
                ),
            )
        }));
    }
    pub fn set_miss_from_str(&mut self, miss: (&str, &str)) {
        self.miss = (
            String::from(miss.0),
            Arc::new(
                Module::create(
                    &self.device,
                    miss.1,
                    self.compile_options.mco(),
                    self.compile_options.pco(),
                )
                .unwrap(),
            ),
        );
    }
    pub fn new() -> Result<Self, Error> {
        let instance = Arc::new(optix_core::Instance::new()?);
        let device = optix_core::Device::create(&instance, 0)?;
        let stream = Arc::new(
            device
                .cuda_device()
                .create_stream(cuda_rs::CUstream_flags_enum::CU_STREAM_DEFAULT)
                .unwrap(),
        );

        let miss_minimal = ".version 6.0 .target sm_50 .address_size 64\n\
                                    .entry __miss__dr() { ret; }";

        let compile_options = CompileOptions::default();

        let miss = Arc::new(
            Module::create(
                &device,
                &miss_minimal,
                compile_options.mco(),
                compile_options.pco(),
            )
            .unwrap(),
        );
        Ok(Self {
            device,
            instance,
            stream,
            hit: vec![],
            miss: ("__miss__dr".into(), miss),
            compile_options,
        })
    }
}

unsafe impl Sync for Backend {}
unsafe impl Send for Backend {}
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
            hit: self.hit.clone(),
            miss: self.miss.clone(),
            compile_options: self.compile_options.clone(),
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

    fn create_accel(
        &self,
        vertices: &Arc<dyn backend::Buffer>,
        indices: &Arc<dyn backend::Buffer>,
    ) -> Arc<dyn backend::Accel> {
        Arc::new(Accel::create(&self.device, &self.stream, vertices, indices).unwrap())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub struct Kernel {
    device: Device,
    pub asm: String,
    entry_point: String,
    pipeline: Option<optix_core::Pipeline>,
    stream: Arc<cuda_core::Stream>,
    compile_options: CompileOptions,
    miss: (String, Arc<Module>),
    hit: Vec<(String, Arc<Module>)>,
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

unsafe impl Sync for Kernel {}
unsafe impl Send for Kernel {}
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
        let compile_options = self.compile_options.clone();

        let rgen = Arc::new(
            optix_core::Module::create(
                &self.device,
                &self.asm,
                compile_options.mco(),
                compile_options.pco(),
            )
            .unwrap(),
        );
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
                entry_point: &self.miss.0,
                module: &self.miss.1,
            },
        )
        .unwrap();

        let hit_pgs = self
            .hit
            .iter()
            .map(|(ep, ptx)| {
                let pg = optix_core::ProgramGroup::create(
                    &self.device,
                    optix_core::ProgramGroupDesc::HitGroup {
                        module_ch: Some(ptx),
                        entry_point_ch: Some(ep),
                        module_ah: None,
                        entry_point_ah: None,
                        module_is: None,
                        entry_point_is: None,
                    },
                )
                .unwrap();
                pg
            })
            .collect::<Vec<_>>();

        let pipeline = optix_core::Pipeline::create(
            &self.device,
            &compile_options.pco(),
            &optix_rs::OptixPipelineLinkOptions {
                maxTraceDepth: 1,
                debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            },
            rgen_pg,
            hit_pgs,
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
        params.extend(
            env.accels()
                .iter()
                .map(|a| a.as_any().downcast_ref::<Accel>().unwrap().ptr()),
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

unsafe impl Sync for Accel {}
unsafe impl Send for Accel {}

#[derive(Debug)]
pub struct Accel {
    device: Device,
    accel: u64,
    buffers: Vec<Arc<dyn backend::Buffer>>, // Keep buffers arround
}

impl Accel {
    pub fn ptr(&self) -> u64 {
        self.accel
    }
    pub fn create(
        device: &Device,
        stream: &Stream,
        vertices: &Arc<dyn backend::Buffer>,
        indices: &Arc<dyn backend::Buffer>,
    ) -> Result<Self, Error> {
        let build_options = OptixAccelBuildOptions {
            // buildFlags: OptixBuildFlags::OPTIX_BUILD_FLAG_NONE as u32,
            buildFlags: OptixBuildFlags::OPTIX_BUILD_FLAG_PREFER_FAST_TRACE as u32
                | OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION as u32,
            operation: OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD,
            ..Default::default()
        };

        let vertex_buffer = vertices.as_any().downcast_ref::<Buffer>().unwrap();
        let indices_buffer = indices.as_any().downcast_ref::<Buffer>().unwrap();

        let flags = [OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT];
        let triangle_array = OptixBuildInputTriangleArray {
            vertexBuffers: &vertex_buffer.ptr(),
            numVertices: (vertex_buffer.size() / (4 * 3)) as _,
            vertexFormat: OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3,
            // vertexStrideInBytes: 0,
            indexBuffer: indices_buffer.ptr(),
            indexFormat: OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
            numIndexTriplets: (indices_buffer.size() / (4 * 3)) as _,
            flags: &flags as *const _ as *const _,
            numSbtRecords: 1,
            ..Default::default()
        };
        let mut build_input = OptixBuildInput {
            type_: OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            __bindgen_anon_1: OptixBuildInput__bindgen_ty_1::default(),
        };
        unsafe {
            *build_input.__bindgen_anon_1.triangleArray.as_mut() = triangle_array;
        }

        let build_inputs = vec![build_input];

        let mut buffer_size = OptixAccelBufferSizes::default();
        unsafe {
            device
                .api()
                .optixAccelComputeMemoryUsage(
                    *device.ctx(),
                    &build_options,
                    build_inputs.as_ptr(),
                    build_inputs.len() as _,
                    &mut buffer_size,
                )
                .check()
                .unwrap()
        };

        let mut d_gas_tmp = 0;
        unsafe {
            device
                .cuda_ctx()
                .cuMemAlloc_v2(&mut d_gas_tmp, buffer_size.tempSizeInBytes)
                .check()
                .unwrap()
        };

        let mut d_gas = 0;
        unsafe {
            device
                .cuda_ctx()
                .cuMemAlloc_v2(&mut d_gas, buffer_size.outputSizeInBytes)
                .check()
                .unwrap()
        };

        let mut accel = 0;
        unsafe {
            device
                .api()
                .optixAccelBuild(
                    *device.ctx(),
                    stream.raw(),
                    &build_options,
                    build_inputs.as_ptr(),
                    build_inputs.len() as _,
                    d_gas_tmp,
                    buffer_size.tempSizeInBytes,
                    d_gas,
                    buffer_size.outputSizeInBytes,
                    &mut accel,
                    std::ptr::null(),
                    0,
                )
                .check()
                .unwrap();
        }

        unsafe {
            device.cuda_ctx().cuMemFree_v2(d_gas_tmp).check().unwrap();
        }

        let buffers = vec![vertices.clone(), indices.clone()];

        Ok(Self {
            device: device.clone(),
            accel,
            buffers,
        })
    }
}

impl backend::Accel for Accel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone, Default)]
pub struct CompileOptions {
    pub num_payload_values: i32,
}

impl CompileOptions {
    pub fn pco(&self) -> OptixPipelineCompileOptions {
        OptixPipelineCompileOptions {
            numAttributeValues: 2,
            pipelineLaunchParamsVariableName: b"params\0" as *const _ as *const _,
            exceptionFlags: optix_rs::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE as _,
            numPayloadValues: self.num_payload_values,
            ..Default::default()
        }
    }
    pub fn mco(&self) -> OptixModuleCompileOptions {
        OptixModuleCompileOptions {
            optLevel: optix_rs::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
            debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            ..Default::default()
        }
    }
}

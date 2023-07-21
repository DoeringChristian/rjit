use resource_pool::hashpool::Lease;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use crate::backend::cuda::cuda_core::{Event, Stream};
use crate::backend::cuda::{self, cuda_core, Buffer, Texture};
use crate::backend::{
    self, CompileOptions, HitGroupDesc, MissGroupDesc, ModuleDesc, SBTDesc, SBTInfo,
};
use crate::schedule::{Env, SVarId, ScheduleIr};
use crate::trace::VarType;
use anyhow::{anyhow, ensure, Result};
use optix_rs::{
    OptixAccelBufferSizes, OptixAccelBuildOptions, OptixAccelEmitDesc, OptixApi, OptixBuildFlags,
    OptixBuildInput, OptixBuildInputInstanceArray, OptixBuildInputTriangleArray,
    OptixBuildInputType, OptixBuildInput__bindgen_ty_1, OptixBuildOperation, OptixDeviceContext,
    OptixDeviceContextOptions, OptixExceptionFlags, OptixGeometryFlags, OptixIndicesFormat,
    OptixInstance, OptixInstanceFlags, OptixModuleCompileOptions, OptixPipelineCompileOptions,
    OptixProgramGroup, OptixProgramGroupDesc, OptixProgramGroupOptions, OptixTraversableHandle,
    OptixVertexFormat,
};
use thiserror::Error;

use super::optix_core::{self, Device, Module, Pipeline, ProgramGroup};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    OptixError(#[from] optix_core::Error),
    #[error("{}", .0)]
    CudaError(#[from] cuda_core::Error),
    #[error("{}", .0)]
    CudaBackendError(#[from] cuda::Error),
}

pub struct InternalBackend {
    instance: Arc<optix_core::Instance>,
    device: optix_core::Device,
    cuda_backend: cuda::Backend,
    // pub compile_options: Mutex<backend::CompileOptions>,
    // pub miss: Mutex<Option<(String, Arc<Module>)>>,
    // pub hit: Mutex<Vec<(String, Arc<Module>)>>,
}

pub struct Backend(Arc<InternalBackend>);

impl Deref for Backend {
    type Target = InternalBackend;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
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

        let miss_minimal = ".version 6.0 .target sm_50 .address_size 64\n\
                                    .entry __miss__dr() { ret; }";

        let compile_options = backend::CompileOptions::default();

        let kernels = Arc::new(
            cuda_core::Module::from_ptx(
                device.cuda_device(),
                include_str!("../cuda/kernels/kernels_70.ptx"),
            )
            .unwrap(),
        );

        let cuda_backend = cuda::Backend(Arc::new(cuda::InternalBackend {
            device: device.cuda_device().clone(),
            kernels,
        }));

        Ok(Self(Arc::new(InternalBackend {
            device,
            instance,
            cuda_backend,
            // hit: Mutex::new(vec![]),
            // miss: Mutex::new(Some(("__miss__dr".into(), miss))),
            // compile_options: Mutex::new(compile_options),
        })))
    }
}

unsafe impl Sync for Backend {}
unsafe impl Send for Backend {}
impl backend::Backend for Backend {
    fn create_texture(
        &self,
        shape: &[usize],
        n_channels: usize,
    ) -> Result<Arc<dyn backend::Texture>> {
        Ok(Arc::new(Texture::create(
            self.device.cuda_device(),
            shape,
            n_channels,
        )?))
    }

    fn buffer_uninit(&self, size: usize) -> Result<Arc<dyn backend::Buffer>> {
        Ok(Arc::new(Buffer::uninit(&self.cuda_backend, size)?))
    }

    fn buffer_from_slice(&self, slice: &[u8]) -> Result<Arc<dyn backend::Buffer>> {
        Ok(Arc::new(Buffer::from_slice(&self.cuda_backend, slice)?))
    }

    fn create_accel(&self, desc: backend::AccelDesc) -> Result<Arc<dyn backend::Accel>> {
        Ok(Arc::new(Accel::create(&self.device, desc)?))
    }

    fn compile_kernel(&self, ir: &ScheduleIr, env: &Env) -> Result<Arc<dyn backend::Kernel>> {
        if env.accels().is_empty() {
            Ok(Arc::new(crate::backend::cuda::Kernel::compile(
                self.device.cuda_device(),
                ir,
                env,
            )?))
        } else {
            Ok(Arc::new(Kernel::compile(&self.device, ir, env)?))
        }
    }

    fn ident(&self) -> &'static str {
        "OptiX"
    }

    // fn assemble_kernel(&self, asm: &str, entry_point: &str) -> Result<Arc<dyn backend::Kernel>> {
    //     Ok(Arc::new(crate::backend::cuda::Kernel::assemble(
    //         self.device.cuda_device(),
    //         asm,
    //         entry_point,
    //     )?))
    // }
}

pub struct Kernel {
    device: Device,
    pub asm: String,
    pipeline: optix_core::Pipeline,
}
impl Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("device", &self.device)
            // .field("asm", &self.asm)
            // .field("kernel_name", &self.entry_point)
            .finish()
    }
}

impl Kernel {
    pub fn compile(
        device: &Device,
        // compile_options: &CompileOptions,
        // miss: &(String, Arc<Module>),
        // hit: &[(String, Arc<Module>)],
        ir: &ScheduleIr,
        env: &Env,
    ) -> Result<Self> {
        let entry_point = "__raygen__cujit";
        // Assemble
        let mut asm = String::new();

        super::codegen::assemble_entry(&mut asm, ir, env, entry_point).unwrap();

        std::fs::write("/tmp/optix.ptx", &asm).unwrap();

        log::trace!("{}", asm);

        // Compile

        // let compile_options = self.compile_options.clone();
        let n_payloads = ir.n_payloads();

        let pco = OptixPipelineCompileOptions {
            numAttributeValues: 2,
            numPayloadValues: n_payloads as _,
            pipelineLaunchParamsVariableName: b"params\0" as *const _ as *const _,
            exceptionFlags: optix_rs::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE as _,
            ..Default::default()
        };
        let mco = OptixModuleCompileOptions {
            optLevel: optix_rs::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
            debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            ..Default::default()
        };

        ensure!(
            env.accels()
                .windows(2)
                .all(|w| w[0].sbt_hash() == w[1].sbt_hash()),
            "All acceleration structures have to request the same SBT!"
        );

        ensure!(
            env.accels()
                .iter()
                .all(|accel| accel.downcast_ref::<Accel>().is_some()),
            "Could not downcast Acceleration Structure!"
        );

        let rgen = Arc::new(optix_core::Module::create(device, &asm, &mco, &pco)?);
        let rgen_pg = optix_core::ProgramGroup::create(
            &device,
            optix_core::ProgramGroupDesc::RayGen {
                module: &rgen,
                entry_point,
            },
        )?;
        let accel = &env.accels()[0];

        let miss_groups = accel
            .sbt_info()
            .miss_groups
            .iter()
            .map(|g| {
                let module = Arc::new(Module::create(&device, &g.miss.asm, &mco, &pco)?);
                Ok(optix_core::ProgramGroup::create(
                    &device,
                    optix_core::ProgramGroupDesc::Miss {
                        entry_point: &g.miss.entry_point,
                        module: &module,
                    },
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        let hit_groups = accel
            .sbt_info()
            .hit_groups
            .iter()
            .map(|g| {
                let module_ch = Arc::new(Module::create(&device, &g.closest_hit.asm, &mco, &pco)?);

                let module_ah = g
                    .any_hit
                    .as_ref()
                    .map(|m| Ok(Arc::new(Module::create(&device, &m.asm, &mco, &pco)?)))
                    .map_or(Ok(None), |r: Result<_>| r.map(Some))?;

                let module_is = g
                    .intersection
                    .as_ref()
                    .map(|m| Ok(Arc::new(Module::create(&device, &m.asm, &mco, &pco)?)))
                    .map_or(Ok(None), |r: Result<_>| r.map(Some))?;

                Ok(optix_core::ProgramGroup::create(
                    &device,
                    optix_core::ProgramGroupDesc::HitGroup {
                        module_ch: Some(&module_ch),
                        entry_point_ch: Some(&g.closest_hit.entry_point),
                        module_ah: module_ah.as_ref(),
                        entry_point_ah: g.any_hit.as_ref().map(|m| m.entry_point.as_ref()),
                        module_is: module_is.as_ref(),
                        entry_point_is: g.intersection.as_ref().map(|m| m.entry_point.as_ref()),
                    },
                )?)
            })
            .collect::<Result<Vec<_>>>()?;

        dbg!(&hit_groups);

        let pipeline = optix_core::Pipeline::create(
            &device,
            &pco,
            &optix_rs::OptixPipelineLinkOptions {
                maxTraceDepth: 1,
                debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            },
            rgen_pg,
            hit_groups,
            miss_groups,
        )?;
        Ok(Self {
            pipeline,
            device: device.clone(),
            asm,
        })
    }
}

unsafe impl Sync for Kernel {}
unsafe impl Send for Kernel {}
impl backend::Kernel for Kernel {
    fn execute_async(
        &self,
        env: &mut crate::schedule::Env,
        size: usize,
    ) -> Result<Arc<dyn backend::DeviceFuture>> {
        let params = [Ok(bytemuck::cast::<_, u64>([size as u32, 0u32]))]
            .into_iter()
            .chain(env.opaques().iter().map(|o| Ok(*o)))
            .chain(env.buffers().iter().map(|b| {
                b.buffer
                    .downcast_ref::<Buffer>()
                    .ok_or(anyhow!("Could not downcast Buffer!"))
                    .map(|b| b.ptr())
            }))
            .chain(env.textures().iter().map(|t| {
                t.downcast_ref::<Texture>()
                    .ok_or(anyhow!("Could not downcast Texture!"))
                    .map(|t| t.ptr())
            }))
            .chain(env.accels().iter().map(|a| {
                a.downcast_ref::<Accel>()
                    .ok_or(anyhow!("Could not downcast Acceleration Structure!"))
                    .map(|a| a.ptr())
            }))
            .collect::<Result<Vec<_>>>()?;

        log::trace!("Optix Kernel Launch with {size} threads.");

        let params: &[u8] = bytemuck::cast_slice(&params);
        let params_buf = self.device.cuda_device().lease_buffer(params.len())?;
        params_buf.copy_from_slice(&params)?;

        unsafe {
            let mut stream = self
                .device
                .cuda_device()
                .create_stream(cuda_rs::CUstream_flags::CU_STREAM_DEFAULT)?;

            self.pipeline
                .launch(&stream, params_buf.ptr(), params.len(), size as u32)?;

            let event = Arc::new(Event::create(&self.device.cuda_device())?);
            stream.record_event(&event)?;
            Ok(Arc::new(DeviceFuture {
                event,
                params: params_buf,
                stream,
            }))
        }
    }

    fn assembly(&self) -> &str {
        &self.asm
    }

    fn backend_ident(&self) -> &'static str {
        "OptiX"
    }
}

unsafe impl Sync for Accel {}
unsafe impl Send for Accel {}

#[derive(Debug)]
pub struct Accel {
    device: Device,
    tlas: (u64, cuda_core::Buffer),
    blaccels: Vec<(u64, cuda_core::Buffer)>,
    buffers: Vec<Arc<dyn backend::Buffer>>, // Keep buffers arround

    sbt_info: SBTInfo,
    sbt_hash: u64,
}

impl Accel {
    pub fn ptr(&self) -> u64 {
        self.tlas.0
    }
    pub fn create(device: &Device, desc: backend::AccelDesc) -> Result<Self, Error> {
        let sbt_info: SBTInfo = desc.sbt.into();

        let mut hasher = DefaultHasher::new();
        sbt_info.hash(&mut hasher);
        let sbt_hash = hasher.finish();

        let stream = Stream::create(
            &device.cuda_device(),
            cuda_rs::CUstream_flags::CU_STREAM_DEFAULT,
        )?;
        let build_options = OptixAccelBuildOptions {
            // buildFlags: OptixBuildFlags::OPTIX_BUILD_FLAG_NONE as u32,
            buildFlags: OptixBuildFlags::OPTIX_BUILD_FLAG_PREFER_FAST_TRACE as u32
                | OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION as u32,
            operation: OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD,
            ..Default::default()
        };

        let build_accel =
            |build_inputs: &[OptixBuildInput]| -> Result<(u64, cuda_core::Buffer), Error> {
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
                // let gas_tmp = Buffer::uninit(device.cuda_device(), buffer_size.tempSizeInBytes)?;
                // let gas = Buffer::uninit(device.cuda_device(), buffer_size.outputSizeInBytes)?;
                let gas_tmp = device
                    .cuda_device()
                    .buffer_uninit(buffer_size.tempSizeInBytes)?;
                let gas = device
                    .cuda_device()
                    .buffer_uninit(buffer_size.outputSizeInBytes)?;
                // dbg!(gas.ptr());
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
                            gas_tmp.ptr(),
                            buffer_size.tempSizeInBytes,
                            gas.ptr(),
                            buffer_size.outputSizeInBytes,
                            &mut accel,
                            std::ptr::null(),
                            0,
                        )
                        .check()
                        .unwrap();
                    // TODO: Async construction needs to keep gas_tmp and gas buffers arround
                    stream.synchronize().unwrap();
                    Ok((accel, gas))
                }
            };

        let mut blaccels = vec![];
        let mut buffers: Vec<Arc<dyn backend::Buffer>> = vec![];
        for geometry in desc.geometries.iter() {
            let build_inputs = match geometry {
                backend::GeometryDesc::Triangles { vertices, indices } => {
                    buffers.push((*vertices).clone()); // Keep references to buffers
                    buffers.push((*indices).clone());

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
                    vec![build_input]
                }
            };

            blaccels.push(build_accel(&build_inputs)?);
        }

        let instances = desc
            .instances
            .iter()
            .enumerate()
            .map(|(i, inst)| OptixInstance {
                transform: inst.transform,
                instanceId: i as _,
                sbtOffset: inst.hit_goup,
                visibilityMask: 255,
                flags: OptixInstanceFlags::OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING as _,
                traversableHandle: blaccels[inst.geometry].0,
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let instance_buf = unsafe {
            device
                .cuda_device()
                .buffer_from_slice(std::slice::from_raw_parts(
                    instances.as_ptr() as *const _,
                    std::mem::size_of::<OptixInstance>() * instances.len(),
                ))
        }?;

        let mut build_input = OptixBuildInput {
            type_: OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_INSTANCES,
            __bindgen_anon_1: Default::default(),
        };
        unsafe {
            *build_input.__bindgen_anon_1.instanceArray.as_mut() = OptixBuildInputInstanceArray {
                instances: instance_buf.ptr(),
                numInstances: instances.len() as _,
            }
        };

        let tlas = build_accel(&[build_input])?;

        Ok(Self {
            device: device.clone(),
            blaccels,
            tlas,
            buffers,
            sbt_info,
            sbt_hash,
        })
    }
}

impl backend::Accel for Accel {
    fn sbt_hash(&self) -> u64 {
        self.sbt_hash
    }

    fn sbt_info(&self) -> &backend::SBTInfo {
        &self.sbt_info
    }
}

impl backend::CompileOptions {
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

unsafe impl Sync for DeviceFuture {}
unsafe impl Send for DeviceFuture {}
#[derive(Debug)]
pub struct DeviceFuture {
    event: Arc<Event>,
    params: Lease<cuda_core::Buffer>,
    stream: Stream,
}

impl backend::DeviceFuture for DeviceFuture {
    fn wait(&self) {
        self.event.synchronize().unwrap();
    }
}

impl Drop for DeviceFuture {
    fn drop(&mut self) {
        self.event.synchronize().unwrap();
    }
}

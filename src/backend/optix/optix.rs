use std::ffi::CString;
use std::fmt::Debug;
use std::sync::Arc;

use crate::backend;
use crate::backend::cuda::cuda_core;
use optix_rs::{
    OptixApi, OptixDeviceContext, OptixDeviceContextOptions, OptixExceptionFlags,
    OptixModuleCompileOptions, OptixPipelineCompileOptions,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    CoreError(#[from] cuda_core::Error),
    #[error("{}", .0)]
    OptixError(#[from] optix_rs::OptixError),
}

pub struct BackendInternal {
    device: cuda_core::Device,
    ctx: OptixDeviceContext,
    api: OptixApi,
    stream: Arc<cuda_core::Stream>,
}

#[derive(Clone)]
pub struct Backend(Arc<BackendInternal>);
impl Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backend")
            .field("device", &self.0.device)
            .field("ctx", &self.0.ctx)
            .field("stream", &self.0.stream)
            .finish()
    }
}

impl Backend {
    pub fn new() -> Result<Self, Error> {
        let instance = Arc::new(cuda_core::Instance::new()?);
        let device = cuda_core::Device::create(&instance, 0)?;
        let stream =
            Arc::new(device.create_stream(cuda_rs::CUstream_flags_enum::CU_STREAM_DEFAULT)?);
        unsafe {
            let optix = optix_rs::OptixApi::find_and_load()?;
            let mut optix_ctx = std::ptr::null_mut();
            optix
                .optixDeviceContextCreate(
                    device.ctx().raw(),
                    &OptixDeviceContextOptions {
                        ..Default::default()
                    },
                    &mut optix_ctx,
                )
                .check()?;
            Ok(Self(Arc::new(BackendInternal {
                device,
                stream,
                ctx: optix_ctx,
                api: optix,
            })))
        }
    }
}

impl backend::Backend for Backend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn new_kernel(&self) -> Box<dyn backend::Kernel> {
        unsafe {}
        todo!()
    }

    fn create_texture(&self, shape: &[usize], n_channels: usize) -> Arc<dyn backend::Texture> {
        todo!()
    }

    fn buffer_uninit(&self, size: usize) -> Arc<dyn backend::Buffer> {
        todo!()
    }

    fn buffer_from_slice(&self, slice: &[u8]) -> Arc<dyn backend::Buffer> {
        todo!()
    }

    fn first_register(&self) -> usize {
        todo!()
    }

    fn synchronize(&self) {
        todo!()
    }

    fn compress(&self, src: &dyn backend::Buffer, dst: &dyn backend::Buffer) -> usize {
        todo!()
    }
}

impl Drop for BackendInternal {
    fn drop(&mut self) {
        unsafe {
            self.api
                .optixDeviceContextDestroy(self.ctx)
                .check()
                .unwrap();
        }
    }
}

#[derive(Debug)]
pub struct Kernel {
    backend: Backend,
    pub asm: String,
}

impl backend::Kernel for Kernel {
    fn as_any(&self) -> &dyn std::any::Any {
        todo!()
    }

    fn assemble(&mut self, ir: &crate::schedule::ScheduleIr, env: &crate::schedule::Env) {
        todo!()
    }

    fn compile(&mut self) {
        unsafe {
            let mco = OptixModuleCompileOptions {
                optLevel:
                    optix_rs::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
                debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
                ..Default::default()
            };
            let pco = OptixPipelineCompileOptions {
                numAttributeValues: 2,
                pipelineLaunchParamsVariableName: b"params" as *const _ as *const _,
                exceptionFlags: optix_rs::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE as _,
                ..Default::default()
            };

            let ptx_cstring = CString::new(self.asm.as_str()).unwrap();

            let mut log = [0i8; 128];
            let mut log_size = log.len();

            let mut module = std::ptr::null_mut();

            self.backend
                .0
                .api
                .optixModuleCreateFromPTX(
                    self.backend.0.ctx,
                    &mco,
                    &pco,
                    ptx_cstring.as_ptr() as *mut _,
                    self.asm.len(),
                    log.as_mut_ptr(),
                    &mut log_size,
                    &mut module,
                )
                .check()
                .unwrap();
        }
    }

    fn execute_async(&mut self, ir: &mut crate::schedule::Env, size: usize) {
        todo!()
    }

    fn assembly(&self) -> &str {
        todo!()
    }
}

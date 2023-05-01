use std::sync::Arc;

use crate::backend::cuda::cuda_core;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    CoreError(#[from] cuda_core::Error),
    #[error("{}", .0)]
    OptixError(#[from] optix_rs::OptixError),
}

pub struct Backend {
    device: cuda_core::Device,
    stream: Arc<cuda_core::Stream>,
}

impl Backend {
    pub fn new() -> Result<Self, Error> {
        let instance = Arc::new(cuda_core::Instance::new()?);
        let device = cuda_core::Device::create(&instance, 0)?;
        let stream =
            Arc::new(device.create_stream(cuda_rs::CUstream_flags_enum::CU_STREAM_DEFAULT)?);
        unsafe {
            optix_rs::OptixApi::find_and_load()?;
        }

        Ok(Self { device, stream })
    }
}

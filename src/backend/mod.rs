pub mod cuda;
pub mod optix;

use anyhow::Result;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use downcast_rs::{impl_downcast, DowncastSync};

use crate::schedule::{Env, ScheduleIr};

pub trait Texture: Debug + Sync + Send + DowncastSync {
    fn channels(&self) -> usize;
    fn dimensions(&self) -> usize;
    fn shape(&self) -> &[usize];
    fn copy_from_buffer(&self, buf: &dyn Buffer);
    fn copy_to_buffer(&self, buf: &dyn Buffer);
}
impl_downcast!(sync Texture);

pub trait Kernel: Debug + Sync + Send + DowncastSync {
    fn into_any(self: Arc<Self>) -> Arc<dyn Any>;
    fn execute_async(&self, env: &mut Env, size: usize) -> Result<Arc<dyn DeviceFuture>>;
    fn assembly(&self) -> &str;
    fn backend_ident(&self) -> &'static str;
}
impl_downcast!(sync Kernel);

pub trait DeviceFuture: Debug + Sync + Send {
    fn wait(&self);
}

pub trait Buffer: Debug + Sync + Send + DowncastSync {
    fn copy_to_host(&self, dst: &mut [u8]);
    fn ptr(&self) -> Option<u64>;
    fn size(&self) -> usize;
}
impl_downcast!(sync Buffer);

pub trait Backend: Debug + Sync + Send + DowncastSync {
    fn compile_kernel(&self, ir: &ScheduleIr, env: &Env) -> Arc<dyn Kernel>;
    fn assemble_kernel(&self, asm: &str, entry_point: &str) -> Arc<dyn Kernel>;
    fn create_texture(&self, shape: &[usize], n_channels: usize) -> Arc<dyn Texture>;
    fn buffer_uninit(&self, size: usize) -> Arc<dyn Buffer>;
    fn buffer_from_slice(&self, slice: &[u8]) -> Arc<dyn Buffer>;
    fn first_register(&self) -> usize;
    fn synchronize(&self);
    fn create_accel(&self, desc: AccelDesc) -> Arc<dyn Accel>;
    fn set_compile_options(&mut self, compile_options: &CompileOptions);
    fn set_miss_from_str(&mut self, entry_point: &str, source: &str);
    fn push_hit_from_str(&mut self, entry_point: &str, source: &str);
    fn ident(&self) -> &'static str;

    fn compress(&self, mask: &dyn Buffer) -> Arc<dyn Buffer>;
}
impl_downcast!(sync Backend);

pub trait Accel: Debug + Sync + Send + DowncastSync {}
impl_downcast!(sync Accel);

#[derive(Clone, Default)]
pub struct CompileOptions {
    pub num_payload_values: i32,
}

pub enum GeometryDesc {
    Triangles {
        vertices: Arc<dyn Buffer>,
        indices: Arc<dyn Buffer>,
    },
}
pub struct InstanceDesc {
    pub geometry: usize,
    pub transform: [f32; 12],
}
pub struct AccelDesc<'a> {
    pub geometries: &'a [GeometryDesc],
    pub instances: &'a [InstanceDesc],
}

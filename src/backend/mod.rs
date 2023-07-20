pub mod cuda;
pub mod optix;

pub mod info;

use anyhow::Result;
use std::fmt::Debug;
use std::sync::Arc;

use downcast_rs::{impl_downcast, DowncastSync};

use crate::schedule::{Env, ScheduleIr};

pub use self::info::*;

pub trait Texture: Debug + Sync + Send + DowncastSync {
    fn channels(&self) -> usize;
    fn dimensions(&self) -> usize;
    fn shape(&self) -> &[usize];
    fn copy_from_buffer(&self, buf: &dyn Buffer) -> Result<()>;
    fn copy_to_buffer(&self, buf: &dyn Buffer) -> Result<()>;
}
impl_downcast!(sync Texture);

pub trait Kernel: Debug + Sync + Send + DowncastSync {
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

    fn compress(&self) -> Result<Arc<dyn Buffer>>;
}
impl_downcast!(sync Buffer);

pub trait Backend: Debug + Sync + Send + DowncastSync {
    fn compile_kernel(&self, ir: &ScheduleIr, env: &Env) -> Result<Arc<dyn Kernel>>;
    // fn assemble_kernel(&self, asm: &str, entry_point: &str) -> Result<Arc<dyn Kernel>>;
    fn create_texture(&self, shape: &[usize], n_channels: usize) -> Result<Arc<dyn Texture>>;
    fn buffer_uninit(&self, size: usize) -> Result<Arc<dyn Buffer>>;
    fn buffer_from_slice(&self, slice: &[u8]) -> Result<Arc<dyn Buffer>>;
    fn first_register(&self) -> usize;
    fn create_accel(&self, desc: AccelDesc) -> Result<Arc<dyn Accel>>;
    fn ident(&self) -> &'static str;
}
impl_downcast!(sync Backend);

pub trait Accel: Debug + Sync + Send + DowncastSync {
    fn sbt_hash(&self) -> u64;
    fn sbt_info(&self) -> &SBTInfo;
}
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
    pub hit_goup: u32,
    pub transform: [f32; 12],
}
pub struct AccelDesc<'a> {
    pub sbt: SBTDesc<'a>,
    pub geometries: &'a [GeometryDesc],
    pub instances: &'a [InstanceDesc],
}

#[derive(Default)]
pub struct HitGroupDesc<'a> {
    closest_hit: ModuleDesc<'a>,
    any_hit: Option<ModuleDesc<'a>>,
    intersection: Option<ModuleDesc<'a>>,
}
pub struct MissGroupDesc<'a> {
    miss: ModuleDesc<'a>,
}

#[derive(Default)]
pub struct ModuleDesc<'a> {
    pub asm: &'a str,
    pub entry_point: &'a str,
}

pub struct SBTDesc<'a> {
    pub hit_groups: &'a [HitGroupDesc<'a>],
    pub miss_groups: &'a [MissGroupDesc<'a>],
}

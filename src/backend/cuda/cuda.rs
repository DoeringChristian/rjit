use anyhow::{anyhow, bail, ensure, Result};
use itertools::Itertools;
use resource_pool::hashpool::Lease;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;
use thiserror::Error;

use super::cuda_core::{self, Device, Event, Function, Instance, Module, Stream};
use super::params;
use crate::backend::{self};
use crate::schedule::{Env, ScheduleIr};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    CoreError(#[from] super::cuda_core::Error),
}

#[derive(Debug)]
pub struct InternalBackend {
    pub device: Device,
    pub kernels: Arc<Module>, // Default kernels
}
#[derive(Debug, Clone)]
pub struct Backend(pub(crate) Arc<InternalBackend>);

impl Deref for Backend {
    type Target = InternalBackend;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Backend {
    pub fn new() -> Result<Self, Error> {
        let instance = Arc::new(Instance::new()?);
        let device = Device::create(&instance, 0)?;

        let kernels =
            Arc::new(Module::from_ptx(&device, include_str!("./kernels/kernels_70.ptx")).unwrap());

        Ok(Self(Arc::new(InternalBackend { kernels, device })))
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
        Ok(Arc::new(Texture::create(&self.device, shape, n_channels)?))
    }
    fn buffer_uninit(&self, size: usize) -> Result<Arc<dyn crate::backend::Buffer>> {
        Ok(Arc::new(Buffer::uninit(&self, size)?))
    }
    fn buffer_from_slice(&self, slice: &[u8]) -> Result<Arc<dyn crate::backend::Buffer>> {
        Ok(Arc::new(Buffer::from_slice(&self, slice)?))
    }

    // fn synchronize(&self) -> Result<()> {
    //     self.stream.synchronize()?;
    //     Ok(())
    // }

    fn create_accel(&self, desc: backend::AccelDesc) -> Result<Arc<dyn backend::Accel>> {
        bail!("Not implemented for CUDA backend!");
    }

    // fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    //     self
    // }

    fn compile_kernel(&self, ir: &ScheduleIr, env: &Env) -> Result<Arc<dyn backend::Kernel>> {
        Ok(Arc::new(Kernel::compile(&self.device, ir, env)?))
    }

    fn ident(&self) -> &'static str {
        "CUDA"
    }

    // fn assemble_kernel(&self, asm: &str, entry_point: &str) -> Result<Arc<dyn backend::Kernel>> {
    //     Ok(Arc::new(Kernel::assemble(&self.device, asm, entry_point)?))
    // }

    // fn compress(&self, mask: &dyn backend::Buffer) -> Result<Arc<dyn backend::Buffer>> {
    //     Ok(super::compress::compress(mask, &self.kernels)?)
    // }
}

impl Drop for Backend {
    fn drop(&mut self) {}
}

#[derive(Debug)]
pub struct Buffer {
    buf: Lease<cuda_core::Buffer>,
    pub(super) size: usize,
    backend: Backend,
}
impl Buffer {
    pub fn ptr(&self) -> u64 {
        self.buf.ptr()
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn uninit(backend: &Backend, size: usize) -> Result<Self> {
        Ok(Self {
            buf: backend.device.lease_buffer(size)?,
            size,
            backend: backend.clone(),
        })
    }
    pub fn from_slice(backend: &Backend, slice: &[u8]) -> Result<Self> {
        let buf = backend.device.lease_buffer(slice.len())?;
        buf.copy_from_slice(slice)?;
        Ok(Self {
            size: slice.len(),
            buf,
            backend: backend.clone(),
        })
    }
    pub fn backend(&self) -> &Backend {
        &self.backend
    }
}
unsafe impl Sync for Buffer {}
unsafe impl Send for Buffer {}
impl backend::Buffer for Buffer {
    fn copy_to_host(&self, dst: &mut [u8]) {
        self.buf.copy_to_host(dst);
    }

    fn ptr(&self) -> Option<u64> {
        Some(self.buf.ptr())
    }

    fn size(&self) -> usize {
        self.size
    }

    fn compress(&self) -> Result<Arc<dyn backend::Buffer>> {
        Ok(super::compress::compress(&self, &self.backend.kernels)?)
    }
}

#[derive(Debug)]
pub struct Texture {
    // tex: cuda_core::Texture,
    textures: Vec<cuda_core::Texture>,
    device: Device,
    n_channels: usize,
}

impl Texture {
    pub fn ptrs<'a>(&'a self) -> impl Iterator<Item = u64> + 'a {
        self.textures.iter().map(|t| t.ptr())
    }
    fn shape(&self) -> &[usize] {
        self.textures[0].shape()
    }
    pub fn n_texels(&self) -> usize {
        self.shape().iter().fold(1, |a, b| a * b) * self.n_channels
    }
    pub fn create(device: &Device, shape: &[usize], n_channels: usize) -> Result<Self> {
        ensure!(
            shape.len() >= 1,
            "A zero dimensional texture is not supported!"
        );
        ensure!(
            shape.len() <= 3,
            "Only textures of dimension less than 3 are supported!"
        );

        let textures = (0..n_channels)
            .step_by(4)
            .map(|i| {
                // let channels_per_texture = (n_channels - i).min(4);
                let tex = device.create_texture(&cuda_core::TexutreDesc {
                    shape,
                    n_channels: 4,
                })?;
                Ok(tex)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            textures,
            device: device.clone(),
            n_channels,
        })
    }
}

unsafe impl Sync for Texture {}
unsafe impl Send for Texture {}
impl backend::Texture for Texture {
    fn n_channels(&self) -> usize {
        self.n_channels
    }

    fn dimensions(&self) -> usize {
        self.textures[0].dim()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn copy_from_buffer(&self, buf: &dyn backend::Buffer) -> Result<()> {
        let buf = buf.as_any().downcast_ref::<Buffer>().unwrap();
        let stream = self
            .device
            .create_stream(cuda_rs::CUstream_flags::CU_STREAM_DEFAULT)?;
        if self.textures.len() > 1 {
            let staging = self.device.lease_buffer(
                self.textures[0].n_texels()
                    * self.textures[0].n_channels()
                    * std::mem::size_of::<f32>(),
            )?;
            let texel_size = self.n_channels * std::mem::size_of::<f32>();
            for (i, tex) in self.textures.iter().enumerate() {
                let texel_offset = i * 4 * std::mem::size_of::<f32>();
                let op = cuda_rs::CUDA_MEMCPY2D {
                    srcXInBytes: texel_offset as _,
                    srcMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcDevice: buf.ptr(),
                    srcPitch: texel_size,

                    dstXInBytes: 0,
                    dstMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstDevice: staging.ptr(),
                    dstPitch: tex.n_channels() * std::mem::size_of::<f32>(),

                    WidthInBytes: tex.n_channels() * std::mem::size_of::<f32>(),
                    Height: tex.n_texels(),
                    ..Default::default()
                };
                dbg!(&op);

                unsafe {
                    self.device
                        .ctx()
                        .cuMemcpy2DAsync_v2(&op, stream.raw())
                        .check()?;
                }

                tex.copy_form_buffer(&staging, &stream)?;
            }
        } else {
            self.textures[0].copy_form_buffer(&buf.buf, &stream)?;
        }
        stream.synchronize()?;
        Ok(())
    }

    fn copy_to_buffer(&self, buf: &dyn backend::Buffer) -> Result<()> {
        let buf = buf.as_any().downcast_ref::<Buffer>().unwrap();
        let stream = self
            .device
            .create_stream(cuda_rs::CUstream_flags::CU_STREAM_DEFAULT)?;
        if self.textures.len() > 1 {
            let staging = self
                .device
                .lease_buffer(self.textures[0].n_texels() * std::mem::size_of::<f32>())?;
            let texel_size = self.n_channels * std::mem::size_of::<f32>();
            for (i, tex) in self.textures.iter().enumerate() {
                tex.copy_to_buffer(&staging, &stream)?;

                let texel_offset = i * 4 * std::mem::size_of::<f32>();

                let op = cuda_rs::CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcDevice: staging.ptr(),
                    srcPitch: tex.n_channels() * std::mem::size_of::<f32>(),

                    dstXInBytes: texel_offset as _,
                    dstMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstDevice: buf.ptr(),
                    dstPitch: texel_size,

                    WidthInBytes: tex.n_channels() * std::mem::size_of::<f32>(),
                    Height: tex.n_texels(),
                    ..Default::default()
                };

                unsafe {
                    self.device
                        .ctx()
                        .cuMemcpy2DAsync_v2(&op, stream.raw())
                        .check()?;
                }
            }
        } else {
            self.textures[0].copy_form_buffer(&buf.buf, &stream)?;
        }
        stream.synchronize()?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct Kernel {
    pub asm: String,
    module: Module,
    func: Function,
    device: Device,
    // stream: Arc<Stream>,
}

impl Kernel {
    pub const ENTRY_POINT: &str = "cujit";
}

impl Kernel {
    pub fn assemble(device: &Device, asm: &str, entry_point: &str) -> Result<Self> {
        let module = Module::from_ptx(&device, &asm)?;
        let func = module.function(entry_point)?;

        Ok(Self {
            asm: String::from(asm),
            module,
            func,
            device: device.clone(),
        })
    }
    pub fn compile(device: &Device, ir: &ScheduleIr, env: &Env) -> Result<Self> {
        assert!(env.accels().is_empty());
        // Assemble:

        let mut asm = String::new();

        super::codegen::assemble_entry(&mut asm, ir, env, Kernel::ENTRY_POINT)?;

        std::fs::write("/tmp/tmp.ptx", &asm).ok();

        log::trace!("{}", asm);

        Self::assemble(device, &asm, Kernel::ENTRY_POINT)
    }
    pub fn launch_size(&self, size: usize) -> (u32, u32) {
        let ctx = self.device.ctx();
        unsafe {
            let mut unused = 0;
            let mut block_size = 0;
            ctx.cuOccupancyMaxPotentialBlockSize(
                &mut unused,
                &mut block_size,
                self.func.raw(),
                None,
                0,
                0,
            )
            .check()
            .unwrap();
            let block_size = block_size as u32;

            let grid_size = (size as u32 + block_size - 1) / block_size;

            (grid_size, block_size)
        }
    }
    pub fn launch_with_size(
        &self,
        params: &mut [*mut std::ffi::c_void],
        size: usize,
    ) -> Result<Arc<DeviceFuture>, Error> {
        let ctx = self.device.ctx();
        unsafe {
            let mut stream = self
                .device
                .create_stream(cuda_rs::CUstream_flags::CU_STREAM_DEFAULT)
                .unwrap();

            let mut unused = 0;
            let mut block_size = 0;
            ctx.cuOccupancyMaxPotentialBlockSize(
                &mut unused,
                &mut block_size,
                self.func.raw(),
                None,
                0,
                0,
            )
            .check()
            .unwrap();
            let block_size = block_size as u32;

            let grid_size = (size as u32 + block_size - 1) / block_size;

            ctx.cuLaunchKernel(
                self.func.raw(),
                grid_size,
                1,
                1,
                block_size as _,
                1,
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .check()
            .unwrap();

            let event = Arc::new(Event::create(&self.device).unwrap());
            stream.record_event(&event).unwrap();
            Ok(Arc::new(DeviceFuture { event, stream }))
        }
    }
    pub fn launch(
        &self,
        params: &mut [*mut std::ffi::c_void],
        block_count: u32,
        thread_count: u32,
        shared_mem_bytes: u32,
    ) -> Arc<DeviceFuture> {
        let ctx = self.device.ctx();
        unsafe {
            let mut stream = self
                .device
                .create_stream(cuda_rs::CUstream_flags::CU_STREAM_DEFAULT)
                .unwrap();

            ctx.cuLaunchKernel(
                self.func.raw(),
                block_count,
                1,
                1,
                thread_count,
                1,
                1,
                shared_mem_bytes,
                stream.raw(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .check()
            .unwrap();

            let event = Arc::new(Event::create(&self.device).unwrap());
            stream.record_event(&event).unwrap();
            Arc::new(DeviceFuture { event, stream })
        }
    }
}

unsafe impl Sync for Kernel {}
unsafe impl Send for Kernel {}
impl backend::Kernel for Kernel {
    fn execute_async(
        &self,
        env: &mut crate::schedule::Env,
        size: usize,
    ) -> anyhow::Result<Arc<dyn backend::DeviceFuture>> {
        let mut params = params::params_cuda(size, &env)?;

        Ok(self.launch_with_size(&mut [params.as_mut_ptr() as *mut _], size)?)
    }

    fn assembly(&self) -> &str {
        &self.asm
    }

    fn backend_ident(&self) -> &'static str {
        "CUDA"
    }
}

unsafe impl Sync for DeviceFuture {}
unsafe impl Send for DeviceFuture {}
#[derive(Debug)]
pub struct DeviceFuture {
    event: Arc<Event>,
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

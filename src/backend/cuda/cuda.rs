use anyhow::{anyhow, bail, Result};
use resource_pool::hashpool::Lease;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;
use thiserror::Error;

use super::cuda_core::{self, Device, Event, Function, Instance, Module, Stream};
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
    tex: u64,
    array: cuda_rs::CUarray,
    n_channels: usize,
    shape: smallvec::SmallVec<[usize; 4]>,
    device: Device,
}

impl Drop for Texture {
    fn drop(&mut self) {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuArrayDestroy(self.array).check().unwrap();
            ctx.cuTexObjectDestroy(self.tex).check().unwrap();
        }
    }
}
impl Texture {
    pub fn ptr(&self) -> u64 {
        self.tex
    }
    pub fn create(device: &Device, shape: &[usize], n_channels: usize) -> Result<Self> {
        let ctx = device.ctx();
        unsafe {
            let mut tex = 0;
            let mut array = std::ptr::null_mut();
            if shape.len() == 1 || shape.len() == 2 {
                let array_desc = cuda_rs::CUDA_ARRAY_DESCRIPTOR {
                    Width: shape[0],
                    Height: if shape.len() == 1 { 1 } else { shape[1] },
                    Format: cuda_rs::CUarray_format::CU_AD_FORMAT_FLOAT,
                    NumChannels: n_channels as _,
                };
                ctx.cuArrayCreate_v2(&mut array, &array_desc).check()?;
            } else if shape.len() == 3 {
                let array_desc = cuda_rs::CUDA_ARRAY3D_DESCRIPTOR {
                    Width: shape[0],
                    Height: shape[1],
                    Depth: shape[2],
                    Format: cuda_rs::CUarray_format::CU_AD_FORMAT_FLOAT,
                    Flags: 0,
                    NumChannels: n_channels as _,
                };
                let mut array = std::ptr::null_mut();
                ctx.cuArray3DCreate_v2(&mut array, &array_desc).check()?;
            } else {
                bail!("Shape not supported!");
            };

            let res_desc = cuda_rs::CUDA_RESOURCE_DESC {
                resType: cuda_rs::CUresourcetype::CU_RESOURCE_TYPE_ARRAY,
                res: cuda_rs::CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                    array: cuda_rs::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                        hArray: array,
                    },
                },
                flags: 0,
            };
            let tex_desc = cuda_rs::CUDA_TEXTURE_DESC {
                addressMode: [cuda_rs::CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP; 3],
                filterMode: cuda_rs::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR,
                flags: 1,
                maxAnisotropy: 1,
                mipmapFilterMode: cuda_rs::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR,
                ..Default::default()
            };
            let view_desc = cuda_rs::CUDA_RESOURCE_VIEW_DESC {
                format: if n_channels == 1 {
                    cuda_rs::CUresourceViewFormat::CU_RES_VIEW_FORMAT_FLOAT_1X32
                } else if n_channels == 2 {
                    cuda_rs::CUresourceViewFormat::CU_RES_VIEW_FORMAT_FLOAT_2X32
                } else if n_channels == 4 {
                    cuda_rs::CUresourceViewFormat::CU_RES_VIEW_FORMAT_FLOAT_4X32
                } else {
                    panic!("{n_channels} number of channels is not supported!");
                },
                width: shape[0],
                height: if shape.len() >= 2 { shape[1] } else { 1 },
                depth: if shape.len() == 3 { shape[2] } else { 0 },
                ..Default::default()
            };
            ctx.cuTexObjectCreate(&mut tex, &res_desc, &tex_desc, &view_desc)
                .check()?;
            Ok(Self {
                n_channels,
                shape: smallvec::SmallVec::from(shape),
                array,
                tex,
                device: device.clone(),
            })
        }
    }
}

unsafe impl Sync for Texture {}
unsafe impl Send for Texture {}
impl backend::Texture for Texture {
    fn channels(&self) -> usize {
        self.n_channels
    }

    fn dimensions(&self) -> usize {
        self.shape.len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn copy_from_buffer(&self, buf: &dyn backend::Buffer) -> Result<()> {
        let buf = buf.as_any().downcast_ref::<Buffer>().unwrap();
        let ctx = self.device.ctx();
        unsafe {
            if self.shape.len() == 1 || self.shape.len() == 2 {
                let pitch = self.shape[0] * self.n_channels * std::mem::size_of::<f32>();
                let op = cuda_rs::CUDA_MEMCPY2D {
                    srcMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcDevice: buf.ptr(),
                    srcPitch: pitch,
                    dstMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstArray: self.array,
                    WidthInBytes: pitch,
                    Height: if self.shape.len() == 2 {
                        self.shape[1]
                    } else {
                        1
                    },
                    ..Default::default()
                };
                ctx.cuMemcpy2D_v2(&op).check()?;
                Ok(())
            } else {
                let pitch = self.shape[0] * self.n_channels * std::mem::size_of::<f32>();
                let op = cuda_rs::CUDA_MEMCPY3D {
                    srcMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcDevice: buf.ptr(),
                    srcHeight: self.shape[1],
                    dstMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstArray: self.array,
                    WidthInBytes: pitch,
                    Height: self.shape[1],
                    Depth: self.shape[2],
                    ..Default::default()
                };
                ctx.cuMemcpy3D_v2(&op).check()?;
                Ok(())
            }
        }
    }

    fn copy_to_buffer(&self, buf: &dyn backend::Buffer) -> Result<()> {
        let buf = buf.as_any().downcast_ref::<Buffer>().unwrap();
        let ctx = self.device.ctx();
        unsafe {
            if self.shape.len() == 1 || self.shape.len() == 2 {
                let pitch = self.shape[0] * self.n_channels * std::mem::size_of::<f32>();
                let op = cuda_rs::CUDA_MEMCPY2D {
                    srcMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    srcArray: self.array,
                    dstMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstDevice: buf.ptr(),
                    dstPitch: pitch,
                    WidthInBytes: pitch,
                    Height: if self.shape.len() == 2 {
                        self.shape[1]
                    } else {
                        1
                    },
                    ..Default::default()
                };
                ctx.cuMemcpy2D_v2(&op).check()?;
                Ok(())
            } else {
                let pitch = self.shape[0] * self.n_channels * std::mem::size_of::<f32>();
                let op = cuda_rs::CUDA_MEMCPY3D {
                    srcMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    srcHeight: self.shape[1],
                    dstMemoryType: cuda_rs::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcArray: self.array,
                    dstDevice: buf.ptr(),
                    WidthInBytes: pitch,
                    Height: self.shape[1],
                    Depth: self.shape[2],
                    ..Default::default()
                };
                ctx.cuMemcpy3D_v2(&op).check()?;
                Ok(())
            }
        }
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
        let mut params = [Ok(size as u64)]
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
            .collect::<anyhow::Result<Vec<_>>>()?;

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

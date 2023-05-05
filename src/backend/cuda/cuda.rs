use std::ffi::c_void;
use std::fmt::{Debug, Write};
use std::sync::Arc;
use thiserror::Error;

use super::cuda_core::{Device, Event, Function, Instance, Module, Stream};
use crate::backend;
use crate::schedule::{Env, SVarId, ScheduleIr};
use crate::trace::VarType;
use crate::var::ParamType;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    CoreError(#[from] super::cuda_core::Error),
}

#[derive(Debug)]
pub struct Backend {
    device: Device,
    stream: Arc<Stream>,
    kernels: Module, // Default kernels
}
impl Backend {
    pub fn new() -> Result<Self, Error> {
        let instance = Arc::new(Instance::new()?);
        let device = Device::create(&instance, 0)?;
        let stream =
            Arc::new(device.create_stream(cuda_rs::CUstream_flags_enum::CU_STREAM_DEFAULT)?);

        let kernels = Module::from_ptx(&device, include_str!("./kernels/kernels_70.ptx")).unwrap();

        Ok(Self {
            device,
            stream,
            kernels,
        })
    }
}

unsafe impl Sync for Backend {}
unsafe impl Send for Backend {}
impl backend::Backend for Backend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn create_texture(&self, shape: &[usize], n_channels: usize) -> Arc<dyn backend::Texture> {
        Arc::new(Texture::create(&self.device, shape, n_channels))
    }
    fn buffer_uninit(&self, size: usize) -> Arc<dyn crate::backend::Buffer> {
        Arc::new(Buffer::uninit(&self.device, size))
    }
    fn buffer_from_slice(&self, slice: &[u8]) -> Arc<dyn crate::backend::Buffer> {
        Arc::new(Buffer::from_slice(&self.device, slice))
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
        todo!()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn set_compile_options(&mut self, compile_options: &backend::CompileOptions) {
        todo!()
    }

    fn set_miss_from_str(&mut self, entry_point: &str, source: &str) {
        todo!()
    }

    fn push_hit_from_str(&mut self, entry_point: &str, source: &str) {
        todo!()
    }

    fn compile_kernel(&self, ir: &ScheduleIr, env: &Env) -> Box<dyn backend::Kernel> {
        Box::new(Kernel::compile(&self.device, &self.stream, ir, env))
    }
}

impl Drop for Backend {
    fn drop(&mut self) {}
}

#[derive(Debug)]
pub struct Buffer {
    device: Device,
    dptr: u64,
    size: usize,
}
impl Buffer {
    pub fn ptr(&self) -> u64 {
        self.dptr
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn uninit(device: &Device, size: usize) -> Self {
        unsafe {
            let ctx = device.ctx();
            let mut dptr = 0;
            ctx.cuMemAlloc_v2(&mut dptr, size).check().unwrap();
            Self {
                device: device.clone(),
                dptr,
                size,
            }
        }
    }
    pub fn from_slice(device: &Device, slice: &[u8]) -> Self {
        unsafe {
            let size = slice.len();

            let ctx = device.ctx();

            let mut dptr = 0;
            ctx.cuMemAlloc_v2(&mut dptr, size).check().unwrap();
            ctx.cuMemcpyHtoD_v2(dptr, slice.as_ptr() as _, size)
                .check()
                .unwrap();
            Self {
                device: device.clone(),
                dptr,
                size,
            }
        }
    }
}
impl backend::Buffer for Buffer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn copy_to_host(&self, dst: &mut [u8]) {
        unsafe {
            let ctx = self.device.ctx();
            assert!(dst.len() <= self.size);

            ctx.cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut _, self.dptr, self.size)
                .check()
                .unwrap();
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.ctx().cuMemFree_v2(self.dptr).check().unwrap();
        }
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
    pub fn create(device: &Device, shape: &[usize], n_channels: usize) -> Self {
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
                ctx.cuArrayCreate_v2(&mut array, &array_desc)
                    .check()
                    .unwrap();
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
                ctx.cuArray3DCreate_v2(&mut array, &array_desc)
                    .check()
                    .unwrap();
            } else {
                panic!("Shape not supported!");
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
                .check()
                .unwrap();
            Self {
                n_channels,
                shape: smallvec::SmallVec::from(shape),
                array,
                tex,
                device: device.clone(),
            }
        }
    }
}

unsafe impl Sync for Texture {}
unsafe impl Send for Texture {}
impl backend::Texture for Texture {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn channels(&self) -> usize {
        self.n_channels
    }

    fn dimensions(&self) -> usize {
        self.shape.len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn copy_from_buffer(&self, buf: &dyn backend::Buffer) {
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
                ctx.cuMemcpy2D_v2(&op).check().unwrap();
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
                ctx.cuMemcpy3D_v2(&op).check().unwrap();
            }
        }
    }

    fn copy_to_buffer(&self, buf: &dyn backend::Buffer) {
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
                ctx.cuMemcpy2D_v2(&op).check().unwrap();
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
                ctx.cuMemcpy3D_v2(&op).check().unwrap();
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
    stream: Arc<Stream>,
}

impl Kernel {
    pub const ENTRY_POINT: &str = "cujit";
    pub const FIRST_REGISTER: usize = 4;
}

impl Kernel {
    pub fn compile(device: &Device, stream: &Arc<Stream>, ir: &ScheduleIr, env: &Env) -> Self {
        assert!(env.accels().is_empty());
        // Assemble:

        let mut asm = String::new();

        super::codegen::assemble_entry(&mut asm, ir, env, Kernel::ENTRY_POINT).unwrap();

        std::fs::write("/tmp/tmp.ptx", &asm).unwrap();

        log::trace!("{}", asm);

        // Compile module:
        let module = Module::from_ptx(&device, &asm).unwrap();
        let func = module.function(Kernel::ENTRY_POINT).unwrap();

        Kernel {
            asm,
            module,
            func,
            device: device.clone(),
            stream: stream.clone(),
        }
    }
}

unsafe impl Sync for Kernel {}
unsafe impl Send for Kernel {}
impl backend::Kernel for Kernel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn execute_async(
        &mut self,
        env: &mut crate::schedule::Env,
        size: usize,
    ) -> Arc<dyn backend::DeviceFuture> {
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

            let mut params = vec![size as u64];
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

            ctx.cuLaunchKernel(
                self.func.raw(),
                grid_size,
                1,
                1,
                block_size as _,
                1,
                1,
                0,
                **self.stream,
                [params.as_mut_ptr() as *mut std::ffi::c_void].as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .check()
            .unwrap();

            let mut event = Event::create(&self.device).unwrap();
            event.record(&self.stream).unwrap();
            Arc::new(DeviceFuture { event })
        }
    }

    fn assembly(&self) -> &str {
        &self.asm
    }
}

unsafe impl Sync for DeviceFuture {}
unsafe impl Send for DeviceFuture {}
#[derive(Debug)]
pub struct DeviceFuture {
    event: Event,
}

impl backend::DeviceFuture for DeviceFuture {
    fn wait(&self) {
        self.event.synchronize();
    }
}
impl Drop for DeviceFuture {
    fn drop(&mut self) {
        self.event.synchronize();
    }
}

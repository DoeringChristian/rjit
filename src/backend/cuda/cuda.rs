use std::ffi::c_void;
use std::fmt::{Debug, Write};
use std::mem::size_of;
use std::sync::Arc;
use thiserror::Error;

use super::cuda_core::{Device, Event, Function, Instance, Module, Stream};
use crate::backend::{self};
use crate::schedule::{Env, SVarId, ScheduleIr};
use crate::trace::VarType;
use crate::var::ParamType;

fn round_pow2(mut x: u32) -> u32 {
    x = x.wrapping_sub(1);
    x |= x.wrapping_shr(1);
    x |= x.wrapping_shr(2);
    x |= x.wrapping_shr(4);
    x |= x.wrapping_shr(8);
    x |= x.wrapping_shr(16);
    x = x.wrapping_add(1);
    x
}
pub fn align(n: usize, alignment: usize) -> usize {
    ((n + (alignment - 1)) / alignment) * alignment
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    CoreError(#[from] super::cuda_core::Error),
}

#[derive(Debug)]
pub struct Backend {
    device: Device,
    stream: Arc<Stream>,
    kernels: Arc<Module>, // Default kernels
}
impl Backend {
    pub fn new() -> Result<Self, Error> {
        let instance = Arc::new(Instance::new()?);
        let device = Device::create(&instance, 0)?;
        let stream =
            Arc::new(device.create_stream(cuda_rs::CUstream_flags_enum::CU_STREAM_DEFAULT)?);

        let kernels =
            Arc::new(Module::from_ptx(&device, include_str!("./kernels/kernels_70.ptx")).unwrap());

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
    // fn as_any(&self) -> &dyn std::any::Any {
    //     self
    // }

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

    fn create_accel(&self, desc: backend::AccelDesc) -> Arc<dyn backend::Accel> {
        todo!()
    }

    // fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    //     self
    // }

    fn set_compile_options(&mut self, compile_options: &backend::CompileOptions) {
        todo!()
    }

    fn set_miss_from_str(&mut self, entry_point: &str, source: &str) {
        todo!()
    }

    fn push_hit_from_str(&mut self, entry_point: &str, source: &str) {
        todo!()
    }

    fn compile_kernel(&self, ir: &ScheduleIr, env: &Env) -> Arc<dyn backend::Kernel> {
        Arc::new(Kernel::compile(&self.device, ir, env))
    }

    fn ident(&self) -> &'static str {
        "CUDA"
    }

    fn assemble_kernel(&self, asm: &str, entry_point: &str) -> Arc<dyn backend::Kernel> {
        Arc::new(Kernel::assemble(&self.device, asm, entry_point))
    }

    fn compress(&self, mask: &dyn backend::Buffer) -> (Arc<dyn backend::Buffer>, usize) {
        let mask = mask.downcast_ref::<Buffer>().unwrap();
        let size = mask.size();
        if size <= 4096 {
            let out = self.buffer_uninit(4 * size);

            let count_out = self
                .buffer_uninit(size_of::<u32>())
                .downcast_arc::<Buffer>()
                .unwrap();

            let mut in_ptr = mask.ptr();
            let mut out_ptr = out.ptr().unwrap();
            let mut count_out_ptr = count_out.ptr();
            let mut size = size as u32;

            let items_per_thread = 4;
            let thread_count = round_pow2((size + items_per_thread - 1) / items_per_thread);
            let shared_size = thread_count * 2 * std::mem::size_of::<u32>() as u32;
            let trailer = thread_count * items_per_thread - size;

            // std::fs::write("/tmp/dbg.txt", std::format!("in_ptr = {in_ptr:#018x}, out_ptr = {out_ptr:#018x}, count_out_ptr = {count_out_ptr:#018x}")).unwrap();

            let mut params = [
                &mut in_ptr as *mut _ as *mut _,
                &mut out_ptr as *mut _ as *mut _,
                &mut size as *mut _ as *mut _,
                &mut count_out_ptr as *mut _ as *mut _,
            ];

            let stream = self
                .device
                .create_stream(cuda_rs::CUstream_flags::CU_STREAM_DEFAULT)
                .unwrap();

            unsafe {
                self.kernels
                    .function("compress_small")
                    .unwrap()
                    .launch(&stream, &mut params, 1, thread_count, shared_size)
                    .unwrap();
            }
            stream.synchronize().unwrap();

            let mut size = 0u32;
            backend::Buffer::copy_to_host(
                &*count_out,
                bytemuck::cast_slice_mut(std::slice::from_mut(&mut size)),
            );
            (out, size as _)
        } else {
            let stream = self
                .device
                .create_stream(cuda_rs::CUstream_flags::CU_STREAM_DEFAULT)
                .unwrap();

            let compress_large = self.kernels.function("compress_large").unwrap();
            let scan_large_u32_int = self.kernels.function("scan_large_u32_init").unwrap();

            let size = size as u32;
            let items_per_thread = 16u32;
            let thread_count = 128u32;
            let items_per_block = items_per_thread * thread_count;
            let block_count = (size + items_per_block - 1) / items_per_block;
            let shared_size = items_per_block * std::mem::size_of::<u32>() as u32;
            let mut scratch_items = block_count + 32;
            let trailer = items_per_block * block_count - size;

            let scratch = self.buffer_uninit(scratch_items as usize * size_of::<u64>());

            let (block_count_init, thread_count_init) =
                unsafe { scan_large_u32_int.launch_size(scratch_items as _).unwrap() };

            let mut params = [
                &mut scratch.ptr().unwrap() as *mut _ as *mut _,
                &mut scratch_items as *mut _ as *mut _,
            ];

            unsafe {
                scan_large_u32_int
                    .launch(&stream, &mut params, block_count_init, thread_count_init, 0)
                    .unwrap();
            }

            let out = self.buffer_uninit(size_of::<u32>() * size as usize);
            let count_out = self.buffer_uninit(size_of::<u32>());

            let mut in_ptr = mask.ptr();
            let mut out_ptr = out.ptr().unwrap();
            let mut count_out_ptr = count_out.ptr().unwrap();

            let mut scratch_ptr = scratch.ptr().unwrap() + 32 * std::mem::size_of::<u64>() as u64;

            let mut params = [
                &mut in_ptr as *mut _ as *mut _,
                &mut out_ptr as *mut _ as *mut _,
                &mut scratch_ptr as *mut _ as *mut _,
                &mut count_out_ptr as *mut _ as *mut _,
            ];

            unsafe {
                compress_large
                    .launch(&stream, &mut params, block_count, thread_count, shared_size)
                    .unwrap();
            }

            stream.synchronize().unwrap();

            let mut size = 0u32;
            backend::Buffer::copy_to_host(
                &*count_out,
                bytemuck::cast_slice_mut(std::slice::from_mut(&mut size)),
            );
            dbg!(size);
            (out, size as _)
        }
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
            let size = align(size, 8);

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
    fn copy_to_host(&self, dst: &mut [u8]) {
        unsafe {
            let ctx = self.device.ctx();
            // assert!(self.size <= dst.len());

            ctx.cuMemcpyDtoH_v2(
                dst.as_mut_ptr() as *mut _,
                self.dptr,
                dst.len().min(self.size),
            )
            .check()
            .unwrap();
        }
    }

    fn ptr(&self) -> Option<u64> {
        Some(self.dptr)
    }

    fn size(&self) -> usize {
        self.size
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
    // stream: Arc<Stream>,
}

impl Kernel {
    pub const ENTRY_POINT: &str = "cujit";
    pub const FIRST_REGISTER: usize = 4;
}

impl Kernel {
    pub fn assemble(device: &Device, asm: &str, entry_point: &str) -> Self {
        let module = Module::from_ptx(&device, &asm).unwrap();
        let func = module.function(entry_point).unwrap();

        Self {
            asm: String::from(asm),
            module,
            func,
            device: device.clone(),
        }
    }
    pub fn compile(device: &Device, ir: &ScheduleIr, env: &Env) -> Self {
        assert!(env.accels().is_empty());
        // Assemble:

        let mut asm = String::new();

        super::codegen::assemble_entry(&mut asm, ir, env, Kernel::ENTRY_POINT).unwrap();

        std::fs::write("/tmp/tmp.ptx", &asm).unwrap();

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
    ) -> Arc<DeviceFuture> {
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
            Arc::new(DeviceFuture { event, stream })
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
    ) -> Arc<dyn backend::DeviceFuture> {
        let mut params = vec![size as u64];
        params.extend(env.opaques());
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
        self.launch_with_size(&mut [params.as_mut_ptr() as *mut _], size)
    }

    fn assembly(&self) -> &str {
        &self.asm
    }

    fn backend_ident(&self) -> &'static str {
        "CUDA"
    }

    fn into_any(self: Arc<Self>) -> Arc<dyn std::any::Any> {
        self
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

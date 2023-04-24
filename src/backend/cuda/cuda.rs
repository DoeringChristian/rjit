use std::ffi::c_void;
use std::fmt::{Debug, Write};
use std::sync::Arc;
use thiserror::Error;

use super::cuda_core::{Device, Function, Instance, Module, Stream};
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

impl backend::Backend for Backend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn new_kernel(&self) -> Box<dyn backend::Kernel> {
        Box::new(Kernel {
            asm: Default::default(),
            module: None,
            func: None,
            device: self.device.clone(),
            stream: self.stream.clone(),
        })
    }

    fn create_texture(&self, shape: &[usize], n_channels: usize) -> Arc<dyn backend::Texture> {
        let ctx = self.device.ctx();
        dbg!(shape);
        dbg!(n_channels);
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
            dbg!(&tex_desc);
            dbg!(&view_desc);
            ctx.cuTexObjectCreate(&mut tex, &res_desc, &tex_desc, &view_desc)
                .check()
                .unwrap();
            Arc::new(Texture {
                n_channels,
                shape: smallvec::SmallVec::from(shape),
                array,
                tex,
                device: self.device.clone(),
            })
        }
    }
    fn buffer_uninit(&self, size: usize) -> Arc<dyn crate::backend::Buffer> {
        unsafe {
            let ctx = self.device.ctx();
            let mut dptr = 0;
            ctx.cuMemAlloc_v2(&mut dptr, size).check().unwrap();
            Arc::new(Buffer {
                device: self.device.clone(),
                dptr,
                size,
            })
        }
    }
    fn buffer_from_slice(&self, slice: &[u8]) -> Arc<dyn crate::backend::Buffer> {
        unsafe {
            let size = slice.len();

            let ctx = self.device.ctx();

            let mut dptr = 0;
            ctx.cuMemAlloc_v2(&mut dptr, size).check().unwrap();
            ctx.cuMemcpyHtoD_v2(dptr, slice.as_ptr() as _, size)
                .check()
                .unwrap();
            Arc::new(Buffer {
                device: self.device.clone(),
                dptr,
                size,
            })
        }
    }

    fn first_register(&self) -> usize {
        Kernel::FIRST_REGISTER
    }

    fn synchronize(&self) {
        self.stream.synchronize().unwrap();
    }

    fn compress(&self, src: &dyn backend::Buffer, dst: &dyn backend::Buffer) -> usize {
        let src = src.as_any().downcast_ref::<Buffer>().unwrap();
        let dst = dst.as_any().downcast_ref::<Buffer>().unwrap();
        let ctx = self.device.ctx();
        unsafe {
            fn round_pow2(mut x: u32) -> u32 {
                x -= 1;
                x |= x.wrapping_shr(1);
                x |= x.wrapping_shr(2);
                x |= x.wrapping_shr(4);
                x |= x.wrapping_shr(8);
                x |= x.wrapping_shr(16);
                x + 1
            }

            let func = self.kernels.function("compress_small").unwrap();
            let mut size = src.size as u32;
            let mut count_dptr = 0;
            let mut src_dptr = src.ptr();
            let mut dst_dptr = dst.ptr();

            let items_per_thread = 4;
            let thread_count = round_pow2((size + items_per_thread - 1) / items_per_thread);
            let shared_size = thread_count * 2 * std::mem::size_of::<u32>() as u32;

            let trailer = thread_count * items_per_thread - size;

            dbg!(thread_count);
            dbg!(shared_size);
            dbg!(size);

            ctx.cuMemAlloc_v2(&mut count_dptr, 4).check().unwrap();

            if trailer > 0 {
                dbg!(size);
                ctx.cuMemsetD8Async(src_dptr + size as u64, 0, trailer as _, self.stream.raw())
                    .check()
                    .unwrap();
            }

            func.launch(
                &self.stream,
                &mut [
                    &mut src_dptr as *mut _ as *mut c_void,
                    &mut dst_dptr as *mut _ as *mut c_void,
                    &mut size as *mut _ as *mut c_void,
                    &mut count_dptr as *mut _ as *mut c_void,
                ],
                (1, 1, 1),
                (thread_count, 1, 1),
                shared_size,
            )
            .unwrap();
            self.stream.synchronize().unwrap();
            let mut out_size: u32 = 0;
            ctx.cuMemcpyDtoH_v2(&mut out_size as *mut _ as *mut c_void, count_dptr, 4)
                .check()
                .unwrap();
            ctx.cuMemFree_v2(count_dptr).check().unwrap();
            dbg!(out_size);
            out_size as usize
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
    fn ptr(&self) -> u64 {
        self.dptr
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
    fn ptr(&self) -> u64 {
        self.tex
    }
}

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
    module: Option<Module>,
    func: Option<Function>,
    device: Device,
    stream: Arc<Stream>,
}

impl Kernel {
    const ENTRY_POINT: &str = "cujit";
    const FIRST_REGISTER: usize = 4;
    #[allow(unused_must_use)]
    fn assemble_var(&mut self, ir: &ScheduleIr, env: &Env, id: SVarId) {
        super::codegen::assemble_var(&mut self.asm, ir, id, 1, 1 + env.buffers().len());
    }
}

impl backend::Kernel for Kernel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    #[allow(unused_must_use)]
    fn assemble(&mut self, ir: &ScheduleIr, env: &Env) {
        self.asm.clear();
        let n_params = 1 + env.buffers().len() + env.textures().len(); // Add 1 for size
        let n_regs = ir.n_regs();

        /* Special registers:
             %r0   :  Index
             %r1   :  Step
             %r2   :  Size
             %p0   :  Stopping predicate
             %rd0  :  Temporary for parameter pointers
             %rd1  :  Pointer to parameter table in global memory if too big
             %b3, %w3, %r3, %rd3, %f3, %d3, %p3: reserved for use in compound
             statements that must write a temporary result to a register.
        */

        writeln!(self.asm, ".version {}.{}", 8, 0);
        writeln!(self.asm, ".target {}", "sm_86");
        writeln!(self.asm, ".address_size 64");

        writeln!(self.asm, "");

        writeln!(self.asm, ".entry {}(", Self::ENTRY_POINT);
        writeln!(
            self.asm,
            "\t.param .align 8 .b8 params[{}]) {{",
            ((n_params + 1) * std::mem::size_of::<u64>())
        );
        writeln!(self.asm, "");

        writeln!(
            self.asm,
            "\t.reg.b8   %b <{n_regs}>; .reg.b16 %w<{n_regs}>; .reg.b32 %r<{n_regs}>;"
        );
        writeln!(
            self.asm,
            "\t.reg.b64  %rd<{n_regs}>; .reg.f32 %f<{n_regs}>; .reg.f64 %d<{n_regs}>;"
        );
        writeln!(self.asm, "\t.reg.pred %p <{n_regs}>;");
        writeln!(self.asm, "");

        write!(
            self.asm,
            "\tmov.u32 %r0, %ctaid.x;\n\
            \tmov.u32 %r1, %ntid.x;\n\
            \tmov.u32 %r2, %tid.x;\n\
            \tmad.lo.u32 %r0, %r0, %r1, %r2; // r0 <- Index\n"
        );

        writeln!(self.asm, "");

        writeln!(
            self.asm,
            "\t// Index Conditional (jump to done if Index >= Size)."
        );
        writeln!(
            self.asm,
            "\tld.param.u32 %r2, [params]; // r2 <- params[0] (Size)"
        );

        write!(
            self.asm,
            "\tsetp.ge.u32 %p0, %r0, %r2; // p0 <- r0 >= r2\n\
           \t@%p0 bra done; // if p0 => done\n\
           \t\n\
           \tmov.u32 %r3, %nctaid.x; // r3 <- nctaid.x\n\
           \tmul.lo.u32 %r1, %r3, %r1; // r1 <- r3 * r1\n\
           \t\n"
        );

        write!(self.asm, "body: // sm_{}\n", 86); // TODO: compute capability from device

        for id in ir.ids() {
            let var = ir.var(id);
            match var.param_ty {
                ParamType::None => self.assemble_var(ir, env, id),
                ParamType::Input => {
                    let param_offset = (var.buf.unwrap() + 1) * 8;
                    // Load from params
                    writeln!(self.asm, "");
                    writeln!(self.asm, "\t// [{}]: {:?} =>", id, var);
                    if var.is_literal() {
                        writeln!(
                            self.asm,
                            "\tld.param.{} {}, [params+{}];",
                            var.ty.name_cuda(),
                            var.reg(),
                            param_offset
                        );
                        continue;
                    } else {
                        writeln!(self.asm, "\tld.param.u64 %rd0, [params+{}];", param_offset);
                    }
                    if var.size > 1 {
                        writeln!(
                            self.asm,
                            "\tmad.wide.u32 %rd0, %r0, {}, %rd0;",
                            var.ty.size()
                        );
                    }

                    if var.ty == VarType::Bool {
                        writeln!(self.asm, "\tld.global.cs.u8 %w0, [%rd0];");
                        writeln!(self.asm, "\tsetp.ne.u16 {}, %w0, 0;", var.reg());
                    } else {
                        writeln!(
                            self.asm,
                            "\tld.global.cs.{} {}, [%rd0];",
                            var.ty.name_cuda(),
                            var.reg(),
                        );
                    }
                }
                ParamType::Output => {
                    let param_offst = (var.buf.unwrap() + 1) * 8;
                    self.assemble_var(ir, env, id);
                    // let offset = param_idx * 8;
                    write!(
                        self.asm,
                        "\n\t// Store:\n\
                       \tld.param.u64 %rd0, [params + {}]; // rd0 <- params[offset]\n\
                           \tmad.wide.u32 %rd0, %r0, {}, %rd0; // rd0 <- Index * ty.size() + \
                           params[offset]\n",
                        param_offst,
                        var.ty.size(),
                    );

                    if var.ty == VarType::Bool {
                        writeln!(self.asm, "\tselp.u16 %w0, 1, 0, {};", var.reg());
                        writeln!(self.asm, "\tst.global.cs.u8 [%rd0], %w0;");
                    } else {
                        writeln!(
                               self.asm,
                               "\tst.global.cs.{} [%rd0], {}; // (Index * ty.size() + params[offset])[Index] <- var",
                               var.ty.name_cuda(),
                               var.reg(),
                           );
                    }
                }
            }
        }

        // End of kernel:

        writeln!(self.asm, "\n\t//End of Kernel:");
        writeln!(
            self.asm,
            "\n\tadd.u32 %r0, %r0, %r1; // r0 <- r0 + r1\n\
           \tsetp.ge.u32 %p0, %r0, %r2; // p0 <- r1 >= r2\n\
           \t@!%p0 bra body; // if p0 => body\n\
           \n"
        );
        writeln!(self.asm, "done:");
        write!(
            self.asm,
            "\n\tret;\n\
       }}\n"
        );

        std::fs::write("/tmp/tmp.ptx", &self.asm).unwrap();

        log::trace!("{}", self.asm);
    }

    fn compile(&mut self) {
        self.module = Some(Module::from_ptx(&self.device, &self.asm).unwrap());
        self.func = Some(
            self.module
                .as_ref()
                .unwrap()
                .function(Self::ENTRY_POINT)
                .unwrap(),
        );
    }

    fn execute_async(&mut self, env: &mut crate::schedule::Env, size: usize) {
        let ctx = self.device.ctx();
        unsafe {
            let mut unused = 0;
            let mut block_size = 0;
            ctx.cuOccupancyMaxPotentialBlockSize(
                &mut unused,
                &mut block_size,
                self.func.as_ref().unwrap().raw(),
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
                self.func.as_ref().unwrap().raw(),
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
        }
    }

    fn assembly(&self) -> &str {
        &self.asm
    }
}

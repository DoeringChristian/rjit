use std::ffi::{c_void, CStr, CString};
use std::fmt::{Debug, Write};
use std::sync::Arc;
use thiserror::Error;

use super::cuda_core::{Device, Instance, Stream};
use crate::backend;
use crate::schedule::{SVarId, ScheduleIr};
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
}
impl Backend {
    pub fn new() -> Result<Self, Error> {
        let instance = Arc::new(Instance::new()?);
        let device = Device::create(&instance, 0)?;
        let stream =
            Arc::new(device.create_stream(cuda_rs::CUstream_flags_enum::CU_STREAM_DEFAULT)?);
        Ok(Self { device, stream })
    }
}

impl backend::Backend for Backend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn new_kernel(&self) -> Box<dyn backend::Kernel> {
        Box::new(Kernel {
            asm: Default::default(),
            data: Default::default(),
            module: std::ptr::null_mut(),
            func: std::ptr::null_mut(),
            device: self.device.clone(),
            stream: self.stream.clone(),
        })
    }

    fn create_texture(&self, shape: &[usize], n_channels: usize) -> Box<dyn backend::Texture> {
        let ctx = self.device.ctx();
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
                addressMode: todo!(),
                filterMode: todo!(),
                flags: todo!(),
                maxAnisotropy: todo!(),
                mipmapFilterMode: todo!(),
                mipmapLevelBias: todo!(),
                minMipmapLevelClamp: todo!(),
                maxMipmapLevelClamp: todo!(),
                borderColor: todo!(),
                reserved: todo!(),
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
                firstMipmapLevel: Default::default(),
                lastMipmapLevel: Default::default(),
                firstLayer: Default::default(),
                lastLayer: Default::default(),
                reserved: Default::default(),
            };
            ctx.cuTexObjectCreate(&mut tex, &res_desc, &tex_desc, &view_desc)
                .check()
                .unwrap();
            Box::new(Texture {
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
    fn as_ptr(&self) -> u64 {
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
}

#[derive(Debug)]
pub struct Kernel {
    pub asm: String,
    pub data: Vec<u8>,
    pub module: cuda_rs::CUmodule,
    pub func: cuda_rs::CUfunction,
    device: Device,
    stream: Arc<Stream>,
}

impl Kernel {
    const ENTRY_POINT: &str = "cujit";
    const FIRST_REGISTER: usize = 4;
    #[allow(unused_must_use)]
    fn assemble_var(&mut self, ir: &ScheduleIr, id: SVarId) {
        super::codegen::assemble_var(&mut self.asm, ir, id);
    }
}

impl backend::Kernel for Kernel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    #[allow(unused_must_use)]
    fn assemble(&mut self, ir: &ScheduleIr) {
        self.asm.clear();
        let n_params = 1 + ir.buffers().len(); // Add 1 for size
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
                ParamType::None => self.assemble_var(ir, id),
                ParamType::Input => {
                    let param_offset = (var.param.unwrap() + 1) * 8;
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
                    let param_offst = (var.param.unwrap() + 1) * 8;
                    self.assemble_var(ir, id);
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
        let ctx = self.device.ctx();
        unsafe {
            let asm = CString::new(self.asm.as_str()).unwrap();

            const log_size: usize = 16384;
            let mut error_log = [0u8; log_size];
            let mut info_log = [0u8; log_size];

            let mut options = [
                cuda_rs::CUjit_option_enum::CU_JIT_OPTIMIZATION_LEVEL,
                cuda_rs::CUjit_option_enum::CU_JIT_LOG_VERBOSE,
                cuda_rs::CUjit_option_enum::CU_JIT_INFO_LOG_BUFFER,
                cuda_rs::CUjit_option_enum::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                cuda_rs::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER,
                cuda_rs::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                cuda_rs::CUjit_option_enum::CU_JIT_GENERATE_LINE_INFO,
                cuda_rs::CUjit_option_enum::CU_JIT_GENERATE_DEBUG_INFO,
            ];

            let mut option_values = [
                4 as *mut c_void,
                1 as *mut c_void,
                info_log.as_mut_ptr() as *mut c_void,
                log_size as *mut c_void,
                error_log.as_mut_ptr() as *mut c_void,
                log_size as *mut c_void,
                0 as *mut c_void,
                0 as *mut c_void,
            ];

            let mut linkstate = std::ptr::null_mut();
            ctx.cuLinkCreate_v2(
                options.len() as _,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
                &mut linkstate,
            )
            .check()
            .unwrap();

            ctx.cuLinkAddData_v2(
                linkstate,
                cuda_rs::CUjitInputType::CU_JIT_INPUT_PTX,
                asm.as_ptr() as *mut c_void,
                self.asm.len(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
            .check()
            .or_else(|err| {
                let error_log = CStr::from_bytes_until_nul(&error_log).unwrap().to_str().unwrap();
                log::error!("Compilation failed. Please see the PTX listing and error message below:\n{}\n{}", error_log, err);
                Err(err)
            }).unwrap();

            let mut link_out = std::ptr::null_mut();
            let mut link_out_size = 0;
            ctx.cuLinkComplete(linkstate, &mut link_out, &mut link_out_size)
                .check()
                .or_else(|err| {
                    let error_log = CStr::from_bytes_until_nul(&error_log).unwrap().to_str().unwrap();
                    log::error!("Compilation failed. Please see the PTX listing and error message below:\n{}\n{}", error_log, err);
                    Err(err)
                }).unwrap();

            log::trace!(
                "Detailed linker output: {}",
                CStr::from_bytes_until_nul(&info_log)
                    .unwrap()
                    .to_str()
                    .unwrap()
            );

            let mut out: Vec<u8> = Vec::with_capacity(link_out_size);
            std::ptr::copy_nonoverlapping(link_out as *mut u8, out.as_mut_ptr(), link_out_size);
            out.set_len(link_out_size);

            ctx.cuLinkDestroy(linkstate).check().unwrap();

            self.data = out;
        }
        unsafe {
            let mut module = std::ptr::null_mut();
            ctx.cuModuleLoadData(&mut module, self.data.as_ptr() as *const c_void)
                .check()
                .unwrap();
            self.module = module;

            let fname = CString::new(Self::ENTRY_POINT).unwrap();
            let mut func = std::ptr::null_mut();
            ctx.cuModuleGetFunction(&mut func, self.module, fname.as_ptr() as *const i8)
                .check()
                .unwrap();
            self.func = func;
        }
    }

    fn execute_async(&mut self, ir: &mut crate::schedule::ScheduleIr) {
        let ctx = self.device.ctx();
        unsafe {
            let mut unused = 0;
            let mut block_size = 0;
            ctx.cuOccupancyMaxPotentialBlockSize(
                &mut unused,
                &mut block_size,
                self.func,
                None,
                0,
                0,
            )
            .check()
            .unwrap();
            let block_size = block_size as u32;

            let grid_size = (ir.size() as u32 + block_size - 1) / block_size;

            let mut params = vec![ir.size() as u64];
            params.extend(
                ir.buffers()
                    .iter()
                    .map(|b| b.as_any().downcast_ref::<Buffer>().unwrap().as_ptr()),
            );

            ctx.cuLaunchKernel(
                self.func,
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

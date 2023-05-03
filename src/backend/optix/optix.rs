use std::ffi::{CStr, CString};
use std::fmt::{Debug, Write};
use std::sync::Arc;

use crate::backend;
use crate::backend::cuda::cuda_core;
use crate::schedule::{Env, SVarId, ScheduleIr};
use crate::trace::VarType;
use crate::var::ParamType;
use optix_rs::{
    OptixApi, OptixDeviceContext, OptixDeviceContextOptions, OptixExceptionFlags,
    OptixModuleCompileOptions, OptixPipelineCompileOptions, OptixProgramGroup,
    OptixProgramGroupDesc,
};
use thiserror::Error;

use super::optix_core::{self, Device};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    OptixError(#[from] optix_core::Error),
}

pub struct Backend {
    instance: Arc<optix_core::Instance>,
    device: optix_core::Device,
    stream: Arc<cuda_core::Stream>,
}
impl Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backend").finish()
    }
}

impl Backend {
    pub fn new() -> Result<Self, Error> {
        let instance = Arc::new(optix_core::Instance::new()?);
        let device = optix_core::Device::create(&instance, 0)?;
        let stream = Arc::new(
            device
                .cuda_device()
                .create_stream(cuda_rs::CUstream_flags_enum::CU_STREAM_DEFAULT)
                .unwrap(),
        );
        Ok(Self {
            device,
            instance,
            stream,
        })
    }
}

impl backend::Backend for Backend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn new_kernel(&self) -> Box<dyn backend::Kernel> {
        Box::new(Kernel {
            device: self.device.clone(),
            stream: self.stream.clone(),
            asm: "".into(),
            entry_point: "__raygen__cujit".into(),
            pipeline: None,
        })
    }

    fn create_texture(&self, shape: &[usize], n_channels: usize) -> Arc<dyn backend::Texture> {
        todo!()
    }

    fn buffer_uninit(&self, size: usize) -> Arc<dyn backend::Buffer> {
        unsafe {
            let ctx = self.device.cuda_ctx();
            let mut dptr = 0;
            ctx.cuMemAlloc_v2(&mut dptr, size).check().unwrap();
            Arc::new(Buffer {
                device: self.device.clone(),
                dptr,
                size,
            })
        }
    }

    fn buffer_from_slice(&self, slice: &[u8]) -> Arc<dyn backend::Buffer> {
        unsafe {
            let size = slice.len();

            let ctx = self.device.cuda_ctx();

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
        todo!()
    }
}

pub struct Kernel {
    device: Device,
    pub asm: String,
    entry_point: String,
    pipeline: Option<optix_core::Pipeline>,
    stream: Arc<cuda_core::Stream>,
}
impl Kernel {
    const FIRST_REGISTER: usize = 4;
    fn assemble_var(&mut self, ir: &ScheduleIr, env: &Env, id: SVarId) {
        crate::backend::cuda::codegen::assemble_var(
            &mut self.asm,
            ir,
            id,
            1,
            1 + env.buffers().len(),
            "const",
        );
    }
}
impl Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("device", &self.device)
            .field("asm", &self.asm)
            .field("kernel_name", &self.entry_point)
            .finish()
    }
}

impl backend::Kernel for Kernel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[allow(unused_must_use)]
    fn assemble(&mut self, ir: &crate::schedule::ScheduleIr, env: &crate::schedule::Env) {
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

        writeln!(self.asm, ".const .align 8 .b8 params[{}];", 8 * n_params);
        writeln!(self.asm, ".entry {}(){{", self.entry_point);
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
            "\tcall (%r0), _optix_get_launch_index_x, ();\n\
            \tld.const.u32 %r1, [params + 4];\n\
            \tadd.u32 %r0, %r0, %r1;\n\n\
            body:\n"
        );

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
                        writeln!(self.asm, "\tld.const.u64 %rd0, [params+{}];", param_offset);
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
                    let param_offset = (var.buf.unwrap() + 1) * 8;
                    self.assemble_var(ir, env, id);
                    // let offset = param_idx * 8;
                    write!(
                        self.asm,
                        "\n\t// Store:\n\
                           \tld.const.u64 %rd0, [params + {}]; // rd0 <- params[offset]\n\
                           \tmad.wide.u32 %rd0, %r0, {}, %rd0; // rd0 <- Index * ty.size() + \
                           params[offset]\n",
                        param_offset,
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

        write!(
            self.asm,
            "\n\tret;\n\
       }}\n"
        );

        std::fs::write("/tmp/tmp.ptx", &self.asm).unwrap();

        log::trace!("{}", self.asm);
    }

    fn compile(&mut self) {
        let mco = OptixModuleCompileOptions {
            optLevel: optix_rs::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
            debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            ..Default::default()
        };
        let pco = OptixPipelineCompileOptions {
            numAttributeValues: 2,
            pipelineLaunchParamsVariableName: b"params\0" as *const _ as *const _,
            exceptionFlags: optix_rs::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE as _,
            ..Default::default()
        };

        let miss_minimal = ".version 6.0 .target sm_50 .address_size 64\n\
                                    .entry __miss__dr() { ret; }";

        let rgen = optix_core::Module::create(&self.device, &self.asm, mco, pco).unwrap();
        let miss = optix_core::Module::create(&self.device, miss_minimal, mco, pco).unwrap();
        let rgen_pg = optix_core::ProgramGroup::create(
            &self.device,
            optix_core::ProgramGroupDesc::RayGen {
                module: &rgen,
                entry_point: &self.entry_point,
            },
        )
        .unwrap();
        let miss_pg = optix_core::ProgramGroup::create(
            &self.device,
            optix_core::ProgramGroupDesc::Miss {
                module: &miss,
                entry_point: "__miss__dr",
            },
        )
        .unwrap();
        let pipeline = optix_core::Pipeline::create(
            &self.device,
            &pco,
            &optix_rs::OptixPipelineLinkOptions {
                maxTraceDepth: 1,
                debugLevel: optix_rs::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            },
            rgen_pg,
            None,
            [miss_pg],
        )
        .unwrap();
        self.pipeline = Some(pipeline);
    }

    fn execute_async(&mut self, env: &mut crate::schedule::Env, size: usize) {
        let params = [size as u32, 0u32];
        let mut params = Vec::from(bytemuck::cast_slice(&params));
        params.extend(
            env.buffers()
                .iter()
                .map(|b| b.as_any().downcast_ref::<Buffer>().unwrap().ptr()),
        );
        // params.extend(
        //     env.textures()
        //         .iter()
        //         .map(|b| b.as_any().downcast_ref::<Texture>().unwrap().ptr()),
        // );

        log::trace!("params: {:02x?}", bytemuck::cast_slice::<_, u8>(&params));
        log::trace!("Optix Kernel Launch with {size} threads.");

        unsafe {
            let mut d_params = 0;
            let ctx = self.device.cuda_ctx();
            ctx.cuMemAlloc_v2(&mut d_params, 8 * params.len())
                .check()
                .unwrap();
            ctx.cuMemcpyHtoD_v2(d_params, params.as_ptr() as *const _, params.len() * 8)
                .check()
                .unwrap(); // TODO: Free somehow...

            self.pipeline
                .as_ref()
                .unwrap()
                .launch(&self.stream, d_params, params.len() * 8, size as u32)
                .unwrap();
            self.stream.synchronize().unwrap();

            ctx.cuMemFree_v2(d_params).check().unwrap();
        }
    }

    fn assembly(&self) -> &str {
        &self.asm
    }
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
            let ctx = self.device.cuda_ctx();
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
            self.device
                .cuda_ctx()
                .cuMemFree_v2(self.dptr)
                .check()
                .unwrap();
        }
    }
}

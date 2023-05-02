use std::ffi::CString;
use std::ops::Deref;
use std::sync::Arc;

use cuda_rs::CudaError;
use optix_rs::{
    OptixApi, OptixDeviceContext, OptixDeviceContextOptions, OptixDeviceProperty, OptixError,
    OptixModule, OptixModuleCompileOptions, OptixPipeline, OptixPipelineCompileOptions,
    OptixPipelineLinkOptions, OptixProgramGroup, OptixProgramGroupDesc, OptixProgramGroupFlags,
    OptixProgramGroupKind, OptixProgramGroupOptions, OptixShaderBindingTable,
    OPTIX_SBT_RECORD_HEADER_SIZE,
};
use smallvec::{smallvec, SmallVec};
use thiserror::Error;

use crate::backend::cuda::cuda_core;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    CudaError(#[from] CudaError),
    #[error("{}", .0)]
    OptixError(#[from] OptixError),
    #[error("Loading Error {}", .0)]
    Loading(#[from] libloading::Error),
    #[error("The CUDA verion {}.{} is not supported!", .0, .1)]
    VersionError(i32, i32),
}

impl From<cuda_core::Error> for Error {
    fn from(value: cuda_core::Error) -> Self {
        match value {
            cuda_core::Error::CudaError(err) => Self::CudaError(err),
            cuda_core::Error::Loading(err) => Self::Loading(err),
            cuda_core::Error::VersionError(major, minor) => Self::VersionError(major, minor),
        }
    }
}

pub struct Instance {
    cuda: Arc<cuda_core::Instance>,
    optix: OptixApi,
}

impl Instance {
    pub fn new() -> Result<Self, Error> {
        unsafe {
            let cuda = Arc::new(cuda_core::Instance::new()?);
            let optix = OptixApi::find_and_load()?;
            Ok(Self { cuda, optix })
        }
    }
}

pub struct InternalDevice {
    cuda_device: cuda_core::Device,
    ctx: OptixDeviceContext,
    instance: Arc<Instance>,
}

#[derive(Clone)]
pub struct Device(Arc<InternalDevice>);
impl Deref for Device {
    type Target = Arc<InternalDevice>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Device {
    pub fn create(instance: &Arc<Instance>, id: i32) -> Result<Self, Error> {
        unsafe {
            let cuda_device = cuda_core::Device::create(&instance.cuda, id)?;
            let mut ctx = std::ptr::null_mut();
            instance
                .optix
                .optixDeviceContextCreate(
                    cuda_device.ctx().raw(),
                    &OptixDeviceContextOptions {
                        ..Default::default()
                    },
                    &mut ctx,
                )
                .check()?;

            Ok(Self(Arc::new(InternalDevice {
                cuda_device,
                ctx,
                instance: instance.clone(),
            })))
        }
    }
}

impl Drop for InternalDevice {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .optix
                .optixDeviceContextDestroy(self.ctx)
                .check()
                .unwrap();
        }
    }
}

pub struct Module {
    device: Device,
    module: OptixModule,
}

impl Module {
    pub fn create(
        device: &Device,
        ptx: &str,
        module_compile_options: OptixModuleCompileOptions,
        pipeline_compile_options: OptixPipelineCompileOptions,
    ) -> Result<Self, Error> {
        unsafe {
            let ptx_cstring = CString::new(ptx).unwrap();
            let mut log = [0i8; 128];
            let mut log_size = log.len();
            let mut module = std::ptr::null_mut();

            device
                .instance
                .optix
                .optixModuleCreateFromPTX(
                    device.ctx,
                    &module_compile_options,
                    &pipeline_compile_options,
                    ptx_cstring.as_ptr() as *const _,
                    ptx.len(),
                    log.as_mut_ptr(),
                    &mut log_size,
                    &mut module,
                )
                .check()?;
            Ok(Self {
                device: device.clone(),
                module,
            })
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            self.device
                .instance
                .optix
                .optixModuleDestroy(self.module)
                .check()
                .unwrap();
        }
    }
}

pub enum ProgramGroupDesc<'a> {
    Miss {
        module: &'a Module,
        entry_point: &'a str,
    },
    RayGen {
        module: &'a Module,
        entry_point: &'a str,
    },
    HitGroup {
        module_ch: Option<&'a Module>,
        entry_point_ch: Option<&'a str>,
        module_ah: Option<&'a Module>,
        entry_point_ah: Option<&'a str>,
        module_is: Option<&'a Module>,
        entry_point_is: Option<&'a str>,
    },
}

impl<'a> From<ProgramGroupDesc<'a>> for OptixProgramGroupDesc {
    fn from(value: ProgramGroupDesc<'a>) -> Self {
        Self {
            kind: match value {
                ProgramGroupDesc::Miss { .. } => {
                    OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS
                }
                ProgramGroupDesc::RayGen { .. } => {
                    OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN
                }
                ProgramGroupDesc::HitGroup { .. } => {
                    OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP
                }
            },
            flags: todo!(),
            __bindgen_anon_1: todo!(),
        }
    }
}

pub struct ProgramGroup {
    group: OptixProgramGroup,
    device: Device,
}

impl ProgramGroup {
    pub fn create<'a>(device: &Device, desc: ProgramGroupDesc<'a>) -> Result<Self, Error> {
        match desc {
            ProgramGroupDesc::Miss {
                module,
                entry_point,
            } => {
                let desc = OptixProgramGroupDesc {
                    kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
                    flags: OptixProgramGroupFlags::OPTIX_PROGRAM_GROUP_FLAGS_NONE as _,
                    __bindgen_anon_1: optix_rs::OptixProgramGroupDesc__bindgen_ty_1 {
                        miss: optix_rs::OptixProgramGroupSingleModule {
                            module: module.module,
                            entryFunctionName: entry_point.as_ptr() as *const _,
                        },
                    },
                };
                let mut log = [0i8; 128];
                let mut log_size = log.len();

                let mut group = std::ptr::null_mut();

                unsafe {
                    module
                        .device
                        .instance
                        .optix
                        .optixProgramGroupCreate(
                            module.device.ctx,
                            &desc,
                            1,
                            &OptixProgramGroupOptions {
                                ..Default::default()
                            },
                            log.as_mut_ptr() as *mut _,
                            &mut log_size,
                            &mut group,
                        )
                        .check()?;
                }
                Ok(Self {
                    device: device.clone(),
                    group,
                })
            }
            ProgramGroupDesc::RayGen {
                module,
                entry_point,
            } => {
                let desc = OptixProgramGroupDesc {
                    kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                    flags: OptixProgramGroupFlags::OPTIX_PROGRAM_GROUP_FLAGS_NONE as _,
                    __bindgen_anon_1: optix_rs::OptixProgramGroupDesc__bindgen_ty_1 {
                        raygen: optix_rs::OptixProgramGroupSingleModule {
                            module: module.module,
                            entryFunctionName: entry_point.as_ptr() as *const _,
                        },
                    },
                };
                let mut log = [0i8; 128];
                let mut log_size = log.len();

                let mut group = std::ptr::null_mut();

                unsafe {
                    module
                        .device
                        .instance
                        .optix
                        .optixProgramGroupCreate(
                            module.device.ctx,
                            &desc,
                            1,
                            &OptixProgramGroupOptions {
                                ..Default::default()
                            },
                            log.as_mut_ptr() as *mut _,
                            &mut log_size,
                            &mut group,
                        )
                        .check()?;
                }
                Ok(Self {
                    // modules: smallvec![module],
                    device: device.clone(),
                    group,
                })
            }
            ProgramGroupDesc::HitGroup {
                mut module_ch,
                entry_point_ch,
                mut module_ah,
                entry_point_ah,
                mut module_is,
                entry_point_is,
            } => {
                let desc = OptixProgramGroupDesc {
                    kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
                    flags: OptixProgramGroupFlags::OPTIX_PROGRAM_GROUP_FLAGS_NONE as _,
                    __bindgen_anon_1: optix_rs::OptixProgramGroupDesc__bindgen_ty_1 {
                        hitgroup: optix_rs::OptixProgramGroupHitgroup {
                            moduleCH: module_ch
                                .as_mut()
                                .map(|module| module.module)
                                .unwrap_or(std::ptr::null_mut()),
                            entryFunctionNameCH: entry_point_ch
                                .as_ref()
                                .map(|entry_point| entry_point.as_ptr() as *const _)
                                .unwrap_or(std::ptr::null()),
                            moduleAH: module_ah
                                .as_mut()
                                .map(|module| module.module)
                                .unwrap_or(std::ptr::null_mut()),
                            entryFunctionNameAH: entry_point_ah
                                .as_ref()
                                .map(|entry_point| entry_point.as_ptr() as *const _)
                                .unwrap_or(std::ptr::null()),
                            moduleIS: module_is
                                .as_mut()
                                .map(|module| module.module)
                                .unwrap_or(std::ptr::null_mut()),
                            entryFunctionNameIS: entry_point_is
                                .as_ref()
                                .map(|entry_point| entry_point.as_ptr() as *const _)
                                .unwrap_or(std::ptr::null()),
                        },
                    },
                };
                let mut log = [0i8; 128];
                let mut log_size = log.len();

                let mut group = std::ptr::null_mut();

                unsafe {
                    device
                        .instance
                        .optix
                        .optixProgramGroupCreate(
                            device.ctx,
                            &desc,
                            1,
                            &OptixProgramGroupOptions {
                                ..Default::default()
                            },
                            log.as_mut_ptr() as *mut _,
                            &mut log_size,
                            &mut group,
                        )
                        .check()?;
                }
                Ok(Self {
                    group,
                    device: device.clone(),
                })
            }
        }
    }
}

impl Drop for ProgramGroup {
    fn drop(&mut self) {
        unsafe {
            self.device
                .instance
                .optix
                .optixProgramGroupDestroy(self.group)
                .check()
                .unwrap();
        }
    }
}

pub struct Pipeline {
    pipeline: OptixPipeline,
    sbt: OptixShaderBindingTable,
    rgen_group: ProgramGroup,
    hit_groups: SmallVec<[ProgramGroup; 4]>,
    miss_groups: SmallVec<[ProgramGroup; 4]>,
    device: Device,
}

impl Pipeline {
    pub fn create(
        device: &Device,
        pipeline_compile_options: &OptixPipelineCompileOptions,
        pipeline_link_options: &OptixPipelineLinkOptions,
        rgen: ProgramGroup,
        hit: impl IntoIterator<Item = ProgramGroup>,
        miss: impl IntoIterator<Item = ProgramGroup>,
        // groups: &[ProgramGroup],
    ) -> Result<Self, Error> {
        let hit = hit.into_iter().collect::<SmallVec<[ProgramGroup; 4]>>();
        let miss = miss.into_iter().collect::<SmallVec<[ProgramGroup; 4]>>();
        unsafe {
            let new_header = |groups: &[ProgramGroup]| -> Result<u64, Error> {
                if groups.len() == 0 {
                    return Ok(0);
                }
                let mut header = vec![0u8; OPTIX_SBT_RECORD_HEADER_SIZE * groups.len()];
                let mut record = 0;
                device
                    .cuda_device
                    .ctx()
                    .cuMemAlloc_v2(&mut record, OPTIX_SBT_RECORD_HEADER_SIZE)
                    .check()?;

                for (i, group) in groups.iter().enumerate() {
                    device
                        .instance
                        .optix
                        .optixSbtRecordPackHeader(
                            group.group,
                            &mut header[OPTIX_SBT_RECORD_HEADER_SIZE * i] as *mut _ as *mut _,
                        )
                        .check()?
                }
                device
                    .cuda_device
                    .ctx()
                    .cuMemcpyHtoD_v2(
                        record,
                        header.as_ptr() as *const _,
                        OPTIX_SBT_RECORD_HEADER_SIZE,
                    )
                    .check()?;
                drop(header);
                Ok(record)
            };
            let rgen_record = new_header(std::slice::from_ref(&rgen))?;
            let hit_record = new_header(&hit)?;
            let miss_record = new_header(&miss)?;

            let sbt = OptixShaderBindingTable {
                raygenRecord: rgen_record,
                missRecordBase: miss_record,
                missRecordStrideInBytes: OPTIX_SBT_RECORD_HEADER_SIZE as _,
                missRecordCount: miss.len() as _,
                hitgroupRecordBase: hit_record,
                hitgroupRecordStrideInBytes: OPTIX_SBT_RECORD_HEADER_SIZE as _,
                hitgroupRecordCount: hit.len() as _,
                ..Default::default()
            };

            let mut log = [0i8; 128];
            let mut log_size = log.len();

            let groups = std::slice::from_ref(&rgen)
                .iter()
                .chain(hit.iter())
                .chain(miss.iter())
                .map(|group| group.group)
                .collect::<SmallVec<[_; 10]>>();

            let mut pipeline = std::ptr::null_mut();
            device
                .instance
                .optix
                .optixPipelineCreate(
                    device.ctx,
                    pipeline_compile_options,
                    pipeline_link_options,
                    groups.as_ptr(),
                    groups.len() as _,
                    log.as_mut_ptr(),
                    &mut log_size,
                    &mut pipeline,
                )
                .check()?;
            Ok(Self {
                pipeline,
                sbt,
                rgen_group: rgen,
                hit_groups: hit,
                miss_groups: miss,
                device: device.clone(),
            })
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            let ctx = self.device.cuda_device.ctx();
            if self.sbt.raygenRecord != 0 {
                ctx.cuMemFree_v2(self.sbt.raygenRecord).check().unwrap();
            }
            if self.sbt.hitgroupRecordBase != 0 {
                ctx.cuMemFree_v2(self.sbt.hitgroupRecordBase)
                    .check()
                    .unwrap();
            }
            if self.sbt.missRecordBase != 0 {
                ctx.cuMemFree_v2(self.sbt.missRecordBase).check().unwrap();
            }
            self.device
                .instance
                .optix
                .optixPipelineDestroy(self.pipeline)
                .check()
                .unwrap();
        }
    }
}

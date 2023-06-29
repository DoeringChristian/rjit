use std::ffi::{c_void, CStr, CString};
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use cuda_rs::{
    CUcontext, CUdevice_attribute, CUevent, CUevent_flags, CUstream, CudaApi, CudaError,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0)]
    CudaError(#[from] CudaError),
    #[error("Loading Error {}", .0)]
    Loading(#[from] libloading::Error),
    #[error("The CUDA verion {}.{} is not supported!", .0, .1)]
    VersionError(i32, i32),
}

pub struct CtxRef {
    device: Arc<InternalDevice>,
}

impl CtxRef {
    pub fn create(device: &Arc<InternalDevice>) -> Self {
        unsafe { device.instance.api.cuCtxPushCurrent_v2(device.ctx) };
        Self {
            device: device.clone(),
        }
    }
    pub fn raw(&self) -> CUcontext {
        self.device.ctx
    }
}
impl Deref for CtxRef {
    type Target = CudaApi;

    fn deref(&self) -> &Self::Target {
        &self.device.instance.api
    }
}
impl Drop for CtxRef {
    fn drop(&mut self) {
        unsafe {
            let mut old_ctx = std::ptr::null_mut();
            self.device.instance.api.cuCtxPopCurrent_v2(&mut old_ctx);
        }
    }
}

#[derive(Debug)]
pub struct DeviceInfo {
    pub pci_bus_id: i32,
    pub pci_dom_id: i32,
    pub pci_dev_id: i32,
    pub num_sm: i32,
    pub unified_addr: i32,
    pub shared_memory_bytes: i32,
    pub cc_minor: i32,
    pub cc_major: i32,
    pub memory_pool: i32,
    pub mem_total: usize,
    pub name: String,
}

pub struct InternalDevice {
    id: i32,
    ctx: CUcontext,
    instance: Arc<Instance>,
    info: DeviceInfo,
}

impl Debug for InternalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("id", &self.id)
            // .field("ctx", &self.ctx)
            .field("instance", &self.instance)
            .field("pci_bus_id", &self.info.pci_bus_id)
            .field("pci_dom_id", &self.info.pci_dom_id)
            .field("pci_dev_id", &self.info.pci_dev_id)
            .field("num_sm", &self.info.num_sm)
            .field("unified_addr", &self.info.unified_addr)
            .field("shared_memory_bytes", &self.info.shared_memory_bytes)
            .field("cc_minor", &self.info.cc_minor)
            .field("cc_major", &self.info.cc_major)
            .field("memory_pool", &self.info.memory_pool)
            .field("mem_total", &self.info.mem_total)
            .field("name", &self.info.name)
            .finish()
    }
}

impl InternalDevice {
    pub fn new(instance: &Arc<Instance>, id: i32) -> Result<Self, Error> {
        unsafe {
            let api = &instance.api;
            let mut ctx = std::ptr::null_mut();
            api.cuDevicePrimaryCtxRetain(&mut ctx, id).check()?;
            api.cuCtxPushCurrent_v2(ctx).check()?;

            let mut pci_bus_id = 0;
            let mut pci_dom_id = 0;
            let mut pci_dev_id = 0;
            let mut num_sm = 0;
            let mut unified_addr = 0;
            let mut shared_memory_bytes = 0;
            let mut cc_minor = 0;
            let mut cc_major = 0;
            let mut memory_pool = 0;

            let mut mem_total = 0;

            let mut name = [0u8; 256];

            api.cuDeviceGetName(name.as_mut_ptr() as *mut _, name.len() as _, id)
                .check()?;
            api.cuDeviceGetAttribute(
                &mut pci_bus_id,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut pci_dev_id,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut pci_dom_id,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut num_sm,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut unified_addr,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut shared_memory_bytes,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut cc_minor,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                id,
            )
            .check()?;
            api.cuDeviceGetAttribute(
                &mut cc_major,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                id,
            )
            .check()?;
            api.cuDeviceTotalMem_v2(&mut mem_total, id).check()?;

            if instance.cuda_version_major > 11
                || (instance.cuda_version_major == 11 && instance.cuda_version_minor >= 2)
            {
                api.cuDeviceGetAttribute(
                    &mut memory_pool,
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                    id,
                )
                .check()?;
            }

            let name = CStr::from_bytes_until_nul(&name).unwrap();
            let name = String::from_utf8_lossy(name.to_bytes()).to_string();

            log::trace!(
                "Found CUDA device {id}: \"{name}\" (PCI ID\
                    {pci_bus_id:#04x}:{pci_dev_id:#04x}.{pci_dom_id}, compute cap.\
                    {cc_major}.{cc_minor}, {num_sm} SMs w/{shared_memory_bytes} shared mem., \
                    {mem_total} global mem.)",
                shared_memory_bytes = bytesize::ByteSize(shared_memory_bytes as _),
                mem_total = bytesize::ByteSize(mem_total as _)
            );

            let mut old_ctx = std::ptr::null_mut();
            api.cuCtxPopCurrent_v2(&mut old_ctx).check()?;

            Ok(Self {
                ctx,
                instance: instance.clone(),
                id,
                info: DeviceInfo {
                    pci_bus_id,
                    pci_dom_id,
                    pci_dev_id,
                    num_sm,
                    unified_addr,
                    shared_memory_bytes,
                    cc_minor,
                    cc_major,
                    memory_pool,
                    mem_total,
                    name,
                },
            })
        }
    }
}
impl Drop for InternalDevice {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .api
                .cuDevicePrimaryCtxRelease_v2(self.id)
                .check()
                .unwrap()
        }
    }
}

#[derive(Clone)]
pub struct Device {
    internal: Arc<InternalDevice>,
}
unsafe impl Sync for Device {}
unsafe impl Send for Device {}
impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.internal)
    }
}

impl Device {
    pub fn create(instance: &Arc<Instance>, id: i32) -> Result<Self, Error> {
        let internal = Arc::new(InternalDevice::new(&instance, id)?);

        Ok(Self { internal })
    }
    // TODO: better context switch operation
    pub fn ctx(&self) -> CtxRef {
        CtxRef::create(&self.internal)
    }
    pub fn create_stream(&self, flags: cuda_rs::CUstream_flags_enum) -> Result<Stream, Error> {
        Stream::create(self, flags)
    }
    pub fn info(&self) -> &DeviceInfo {
        &self.internal.info
    }
}

pub struct Instance {
    api: CudaApi,
    device_count: i32,
    cuda_version_major: i32,
    cuda_version_minor: i32,
}
unsafe impl Sync for Instance {}
unsafe impl Send for Instance {}
impl Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            // .field("api", &self.api)
            .field("device_count", &self.device_count)
            .field("cuda_version_major", &self.cuda_version_major)
            .field("cuda_version_minor", &self.cuda_version_minor)
            .finish()
    }
}
impl Instance {
    pub fn new() -> Result<Self, Error> {
        unsafe {
            let api = CudaApi::find_and_load()?;
            api.cuInit(0).check()?;

            let mut device_count = 0;
            api.cuDeviceGetCount(&mut device_count).check()?;

            let mut cuda_version = 0;
            api.cuDriverGetVersion(&mut cuda_version).check()?;

            let cuda_version_major = cuda_version / 1000;
            let cuda_version_minor = (cuda_version % 1000) / 10;

            log::trace!(
                "Found CUDA driver with version {}.{}",
                cuda_version_major,
                cuda_version_minor
            );

            if cuda_version_major < 10 {
                log::error!(
                    "CUDA version {}.{} is to old an not supported. The minimum supported\
                    version is 10.x",
                    cuda_version_major,
                    cuda_version_minor
                );
                return Err(Error::VersionError(cuda_version_major, cuda_version_minor));
            }
            Ok(Self {
                api,
                device_count,
                cuda_version_major,
                cuda_version_minor,
            })
        }
    }
    pub fn device_count(&self) -> usize {
        self.device_count as _
    }
}

impl Drop for Instance {
    fn drop(&mut self) {}
}

#[derive(Debug)]
pub struct Stream {
    events: Vec<Arc<Event>>,
    raw: CUstream,
    device: Device,
}

impl Stream {
    pub fn create(device: &Device, flags: cuda_rs::CUstream_flags_enum) -> Result<Self, Error> {
        let ctx = device.ctx();
        unsafe {
            let mut stream = std::ptr::null_mut();
            ctx.cuStreamCreate(&mut stream, flags as _).check()?;

            Ok(Self {
                raw: stream,
                device: device.clone(),
                events: vec![],
            })
        }
    }
    #[must_use]
    pub fn synchronize(&self) -> Result<(), Error> {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuStreamSynchronize(self.raw).check()?;
        }
        Ok(())
    }
    pub unsafe fn raw(&self) -> CUstream {
        self.raw
    }
    pub fn record_event(&mut self, event: &Arc<Event>) -> Result<(), Error> {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuEventRecord(event.raw(), self.raw()).check()?;
        }
        self.events.push(event.clone());
        Ok(())
    }
}

impl Deref for Stream {
    type Target = CUstream;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuStreamSynchronize(self.raw).check().unwrap();
            ctx.cuStreamDestroy_v2(self.raw).check().unwrap();
        }
    }
}

#[derive(Debug)]
pub struct InternalModule {
    module: cuda_rs::CUmodule,
    device: Device,
}

impl Drop for InternalModule {
    fn drop(&mut self) {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuModuleUnload(self.module);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Module(Arc<InternalModule>);

impl Module {
    pub fn from_ptx(device: &Device, ptx: &str) -> Result<Module, Error> {
        let ctx = device.ctx();
        unsafe {
            let ptx_cstring = CString::new(ptx).unwrap();

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
            .check()?;

            ctx.cuLinkAddData_v2(
                linkstate,
                cuda_rs::CUjitInputType::CU_JIT_INPUT_PTX,
                ptx_cstring.as_ptr() as *mut c_void,
                ptx.len(),
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
            })?;

            let mut link_out = std::ptr::null_mut();
            let mut link_out_size = 0;
            ctx.cuLinkComplete(linkstate, &mut link_out, &mut link_out_size)
                .check()
                .or_else(|err| {
                    let error_log = CStr::from_bytes_until_nul(&error_log).unwrap().to_str().unwrap();
                    log::error!("Compilation failed. Please see the PTX listing and error message below:\n{}\n{}", error_log, err);
                    Err(err)
                })?;

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

            ctx.cuLinkDestroy(linkstate).check()?;

            let mut module = std::ptr::null_mut();
            ctx.cuModuleLoadData(&mut module, out.as_ptr() as *const c_void)
                .check()?;

            Ok(Self(Arc::new(InternalModule {
                module,
                device: device.clone(),
            })))
        }
    }
    pub fn function(&self, name: &str) -> Result<Function, Error> {
        let ctx = self.0.device.ctx();
        unsafe {
            let fname = CString::new(name).unwrap();
            let mut func = std::ptr::null_mut();
            ctx.cuModuleGetFunction(&mut func, self.0.module, fname.as_ptr() as *const i8)
                .check()?;
            Ok(Function {
                func,
                module: self.clone(),
            })
        }
    }
}

#[derive(Debug)]
pub struct Function {
    func: cuda_rs::CUfunction,
    module: Module,
}

impl Function {
    pub unsafe fn raw(&self) -> cuda_rs::CUfunction {
        self.func
    }
    pub unsafe fn launch_size(&self, size: usize) -> Result<(usize, usize), Error> {
        let ctx = self.module.0.device.ctx();
        unsafe {
            let mut unused = 0;
            let mut block_size = 0;
            ctx.cuOccupancyMaxPotentialBlockSize(
                &mut unused,
                &mut block_size,
                self.raw(),
                None,
                0,
                0,
            )
            .check()?;

            let block_size = block_size as usize;
            let grid_size = (size + block_size - 1) / block_size;

            Ok((grid_size, block_size))
        }
    }
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        args: &mut [*mut c_void],
        grid_size: impl Into<KernelSize>,
        block_size: impl Into<KernelSize>,
        shared_size: u32,
    ) -> Result<(), Error> {
        let ctx = self.module.0.device.ctx();
        // let mut unused = 0;
        // let mut block_size = 0;
        // ctx.cuOccupancyMaxPotentialBlockSize(&mut unused, &mut block_size, self.raw(), None, 0, 0)
        //     .check()?;
        // let block_size = block_size as u32;
        //
        // let grid_size = (size as u32 + block_size - 1) / block_size;
        let block_size = block_size.into();
        let grid_size = grid_size.into();

        ctx.cuLaunchKernel(
            self.raw(),
            grid_size.0,
            grid_size.1,
            grid_size.2,
            block_size.0,
            block_size.1,
            block_size.2,
            shared_size,
            **stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
        .check()?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct KernelSize(pub u32, pub u32, pub u32);

impl From<u32> for KernelSize {
    fn from(value: u32) -> Self {
        Self(value, 1, 1)
    }
}
impl From<(u32, u32)> for KernelSize {
    fn from(value: (u32, u32)) -> Self {
        Self(value.0, value.1, 1)
    }
}
impl From<(u32, u32, u32)> for KernelSize {
    fn from(value: (u32, u32, u32)) -> Self {
        Self(value.0, value.1, value.2)
    }
}
impl From<usize> for KernelSize {
    fn from(value: usize) -> Self {
        Self(value as _, 1, 1)
    }
}
impl From<(usize, usize)> for KernelSize {
    fn from(value: (usize, usize)) -> Self {
        Self(value.0 as _, value.1 as _, 1)
    }
}
impl From<(usize, usize, usize)> for KernelSize {
    fn from(value: (usize, usize, usize)) -> Self {
        Self(value.0 as _, value.1 as _, value.2 as _)
    }
}
impl From<i32> for KernelSize {
    fn from(value: i32) -> Self {
        Self(value as _, 1, 1)
    }
}
impl From<(i32, i32)> for KernelSize {
    fn from(value: (i32, i32)) -> Self {
        Self(value.0 as _, value.1 as _, 1)
    }
}
impl From<(i32, i32, i32)> for KernelSize {
    fn from(value: (i32, i32, i32)) -> Self {
        Self(value.0 as _, value.1 as _, value.2 as _)
    }
}

#[derive(Debug)]
pub struct Event {
    // stream: Option<Arc<Stream>>,
    device: Device,
    event: CUevent,
}

impl Event {
    pub fn create(device: &Device) -> Result<Self, Error> {
        let ctx = device.ctx();
        unsafe {
            let mut event = std::ptr::null_mut();
            ctx.cuEventCreate(&mut event, CUevent_flags::CU_EVENT_DEFAULT as _)
                .check()?;
            Ok(Self {
                device: device.clone(),
                // stream: None,
                event,
            })
        }
    }
    pub fn raw(&self) -> CUevent {
        self.event
    }
    pub fn synchronize(&self) -> Result<(), Error> {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuEventSynchronize(self.event).check()?;
        }
        Ok(())
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let ctx = self.device.ctx();
        unsafe {
            ctx.cuEventDestroy_v2(self.event).check().unwrap();
        }
    }
}

// TODO: Buffer implementation
//
// pub struct Buffer {
//     device: Device,
//     ptr: u64,
//     size: usize,
// }
//
// impl Buffer{
//     pub fn uninit(device: &Device) -> Result<Self, Error>{
//         let ctx = device.ctx();
//         unsafe{
//         }
//     }
// }

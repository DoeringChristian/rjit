pub mod cuda;

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use crate::schedule::ScheduleIr;

pub trait Texture: Debug {
    fn as_any(&self) -> &dyn Any;
}

pub trait Kernel: Debug {
    fn as_any(&self) -> &dyn Any;
    fn assemble(&mut self, ir: &ScheduleIr);
    fn compile(&mut self);
    fn execute_async(&mut self, ir: &mut ScheduleIr);
    fn assembly(&self) -> &str;
}

pub trait Buffer: Debug {
    fn as_any(&self) -> &dyn Any;
    fn copy_to_host(&self, dst: &mut [u8]);
}

pub trait Backend: Debug {
    fn as_any(&self) -> &dyn Any;
    fn new_kernel(&self) -> Box<dyn Kernel>;
    fn create_texture(&self, shape: &[usize], n_channels: usize) -> Box<dyn Texture>;
    fn buffer_uninit(&self, size: usize) -> Arc<dyn Buffer>;
    fn buffer_from_slice(&self, slice: &[u8]) -> Arc<dyn Buffer>;
    fn first_register(&self) -> usize;
    fn synchronize(&self);
}

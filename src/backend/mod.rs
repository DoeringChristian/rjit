pub mod cuda;

use std::fmt::Debug;
use std::sync::Arc;

use crate::schedule::ScheduleIr;

pub trait Texture: Debug + Send + Sync {}

pub trait Kernel: Debug {
    fn assemble(&mut self, ir: &ScheduleIr);
    fn compile(&mut self);
    fn execute_async(&mut self, ir: &mut ScheduleIr);
    fn assembly(&self) -> &str;
}

pub trait Buffer: Send + Sync + Debug {
    fn as_ptr(&self) -> u64;
    fn as_vec(&self) -> Vec<u8>;
}

pub trait Backend: Debug + Send + Sync {
    // type Kernel: Kernel;
    // type Buffer: Buffer;
    fn new_kernel(&self) -> Box<dyn Kernel>;
    fn buffer_uninit(&self, size: usize) -> Arc<dyn Buffer>;
    fn buffer_from_slice(&self, slice: &[u8]) -> Arc<dyn Buffer>;
    fn first_register(&self) -> usize;
    fn synchronize(&self);
}

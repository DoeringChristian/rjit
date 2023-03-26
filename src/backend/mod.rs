pub mod cuda;

use std::fmt::Debug;

use crate::schedule::ScheduleIr;

pub trait Kernel: Debug + Send + Sync {
    fn assemble(&mut self, ir: &ScheduleIr);
    fn compile(&mut self);
    fn execute_async(&mut self, ir: &mut ScheduleIr);
    fn assembly(&self) -> &str;
}

pub trait Buffer: Send + Sync {
    fn as_ptr(&self) -> u64;
    fn as_vec(&self) -> Vec<u8>;
}

pub trait Backend: Debug + Send + Sync {
    // type Kernel: Kernel;
    // type Buffer: Buffer;
    fn new_kernel(&self) -> Box<dyn Kernel>;
    fn buffer_uninit(&self, size: usize) -> Box<dyn Buffer>;
    fn buffer_from_slice(&self, slice: &[u8]) -> Box<dyn Buffer>;
    fn first_register(&self) -> usize;
    fn synchronize(&self);
}

pub mod cuda;

use crate::schedule::ScheduleIr;

pub trait Kernel {
    fn assemble(&mut self, ir: &ScheduleIr);
    fn compile(&mut self);
    fn execute(&mut self, ir: &mut ScheduleIr);
}

pub trait Buffer {
    fn as_ptr(&self) -> u64;
    fn as_vec(&self) -> Vec<u8>;
}

pub trait Backend {
    // type Kernel: Kernel;
    // type Buffer: Buffer;
    fn new_kernel(&self) -> Box<dyn Kernel>;
    fn buffer_uninit(&self, size: usize) -> Box<dyn Buffer>;
    fn buffer_from_slice(&self, slice: &[u8]) -> Box<dyn Buffer>;
    fn first_register(&self) -> usize;
}

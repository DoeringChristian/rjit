pub mod cuda;

use std::fmt::Debug;

use crate::schedule::ScheduleIr;

pub trait Kernel: Debug {
    fn assemble(&mut self, ir: &ScheduleIr<impl Backend>);
    fn compile(&mut self);
    fn execute(&mut self, ir: &mut ScheduleIr<impl Backend>);
    fn assembly(&self) -> &str;
}

pub trait Buffer {
    fn as_ptr(&self) -> u64;
    fn as_vec(&self) -> Vec<u8>;
}

pub trait Backend: Debug {
    type Kernel: Kernel;
    type Buffer: Buffer;
    fn new_kernel(&self) -> Self::Kernel;
    fn buffer_uninit(&self, size: usize) -> Self::Buffer;
    fn buffer_from_slice(&self, slice: &[u8]) -> Self::Buffer;
    fn first_register(&self) -> usize;
}

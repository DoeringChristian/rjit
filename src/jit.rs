use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::fmt::Write;
use std::sync::Arc;

use rayon::prelude::*;

use crate::backend::{Backend, Kernel};
use crate::ir::{self, Op, ParamType, Ref, BACKEND, IR};
use crate::schedule::ScheduleIr;

thread_local! {pub static JIT: RefCell<Jit> = RefCell::new(Jit::default())}

// TODO: pooling for paralel exectution
///
/// The Jit Compiler first generates schedules (Intermediate Representation) from a Trace.
/// It then assembles and compiles a Kernel depending on the Backend.
///
/// Ir -> [Schedule; N] -> [Kernel; N]
///
/// Where N is the number of schedule groups.
/// These are extracted from the scheduled variables in the Ir.
///
#[derive(Debug, Default)]
pub struct Jit {
    pub schedules: Vec<ScheduleIr>,
    pub kernels: Vec<Box<dyn Kernel>>,
    pub scheduled: Vec<Ref>,
}

///
/// Compiles the computation graph of all scheduled variables in a Ir.
///
/// First, all scheduled variables with the same size are grouped.
/// Then, a Schedule Intermediate Representation is constructed from the groups.
/// In the end a set of Kernels is assembled and compiled.
///
fn compile() {
    BACKEND.with(|backend| {
        JIT.with(|jit| {
            IR.with(|ir| {
                let mut jit = jit.borrow_mut();

                jit.schedules.clear();
                jit.kernels.clear();
                let mut scheduled = jit.scheduled.clone();
                scheduled.sort_by(|r0, r1| ir.borrow().var(r0).size.cmp(&ir.borrow().var(r1).size));

                // For every scheduled variable (destination) we have to create a new buffer
                for id in scheduled.iter() {
                    let mut ir = ir.borrow_mut();
                    let var = ir.var_mut(id);
                    var.param_ty = ParamType::Output;
                    var.buffer = Some(
                        backend
                            .borrow()
                            .as_ref()
                            .unwrap()
                            .buffer_uninit(var.size * var.ty.size()),
                    );
                }

                let cur = 0;
                let mut size;
                for i in 1..scheduled.len() {
                    let ir = ir.borrow();
                    let var0 = ir.var(&scheduled[i - 1]);
                    let var1 = ir.var(&scheduled[i]);
                    size = var0.size;
                    if var0.size != var1.size {
                        let mut tmp = ScheduleIr::new(
                            &backend.borrow().as_ref().unwrap(),
                            backend.borrow().as_ref().unwrap().first_register(),
                            size,
                        );
                        tmp.collect_vars(&scheduled[cur..i]);
                        jit.borrow_mut().schedules.push(tmp);
                    }
                }
                size = ir.borrow().var(scheduled.last().unwrap()).size;
                let mut tmp = ScheduleIr::new(
                    &backend.borrow().as_ref().unwrap(),
                    backend.borrow().as_ref().unwrap().first_register(),
                    size,
                );

                tmp.collect_vars(&scheduled[cur..scheduled.len()]);
                jit.schedules.push(tmp);

                jit.kernels = jit
                    .schedules
                    .iter()
                    .map(|mut s| {
                        let mut kernel = backend.borrow().as_ref().unwrap().new_kernel();
                        kernel.assemble(&mut s);
                        // kernel.compile();
                        kernel
                    })
                    .collect::<Vec<_>>();
                for kernel in jit.kernels.iter_mut() {
                    kernel.compile();
                }
            })
        })
    })
}
///
/// Evaluates a Ir by first constructing Schedules, which are then compiled and assembled
/// into kernels.
///
/// A the end, all scheduled variables are overwritten with the calculated data.
///
pub fn eval() {
    BACKEND.with(|backend| {
        JIT.with(|jit| {
            compile();
            let n_kernels = jit.borrow().kernels.len();
            for i in 0..n_kernels {
                let mut jit = jit.borrow_mut();
                let (kernel, schedule) = jit.schedule_kernel(i);
                kernel.execute_async(schedule);
            }

            backend.borrow().as_ref().unwrap().synchronize();

            // After executing the kernels, the Ir is cleaned up.
            // To do so, we first decrement the refcount and then set the ParamType to Input and op to
            // Data
            jit.borrow_mut().scheduled.clear();
        })
    })
}

pub fn scedule(refs: &[&Ref]) {
    JIT.with(|jit| {
        for r in refs {
            jit.borrow_mut().scheduled.push((*r).clone());
        }
    })
}

impl Jit {
    pub fn schedule_kernel(&mut self, i: usize) -> (&mut Box<dyn Kernel>, &mut ScheduleIr) {
        (&mut self.kernels[i], &mut self.schedules[i])
    }
    ///
    /// Writes the kernel assemblies into a string which can then be checked by snapshot testing
    /// tools such as insta.
    ///
    pub fn kernel_debug(&self) -> String {
        let mut string = String::new();
        for (i, k) in self.kernels.iter().enumerate() {
            writeln!(string, "===============================================").unwrap();
            writeln!(string, "Kernel {}:", i).unwrap();
            writeln!(string, "").unwrap();
            write!(string, "{}", k.assembly()).unwrap();
        }
        string
    }
}

#[cfg(test)]
mod test {}

use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::fmt::Write;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::backend::{Backend, Kernel};
use crate::ir::{self, Ir, Op, ParamType, Ref, VarId, BACKEND, IR};
use crate::schedule::ScheduleIr;

pub static JIT: Lazy<Mutex<Jit>> = Lazy::new(|| Mutex::new(Jit::default()));

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
    pub scheduled: Vec<VarId>,
}

///
/// Evaluates a Ir by first constructing Schedules, which are then compiled and assembled
/// into kernels.
///
/// A the end, all scheduled variables are overwritten with the calculated data.
///
pub fn eval() {
    JIT.lock().eval();
}

pub fn schedule(refs: &[&Ref]) {
    dbg!();
    let mut jit = JIT.lock();
    for r in refs {
        IR.lock().inc_rc(r.id());
        jit.scheduled.push(r.id());
    }
}
///
/// Writes the kernel assemblies into a string which can then be checked by snapshot testing
/// tools such as insta.
///
pub fn kernel_debug() -> String {
    let jit = JIT.lock();

    let mut string = String::new();
    for (i, k) in jit.borrow().kernels.iter().enumerate() {
        writeln!(string, "===============================================").unwrap();
        writeln!(string, "Kernel {}:", i).unwrap();
        writeln!(string, "").unwrap();
        write!(string, "{}", k.assembly()).unwrap();
    }
    string
}

impl Jit {
    ///
    /// Evaluates a Ir by first constructing Schedules, which are then compiled and assembled
    /// into kernels.
    ///
    /// A the end, all scheduled variables are overwritten with the calculated data.
    ///
    pub fn eval(&mut self) {
        self.compile(&mut IR.lock());
        let n_kernels = self.kernels.len();
        for i in 0..n_kernels {
            let (kernel, schedule) = self.schedule_kernel(i);
            kernel.execute_async(schedule);
        }

        BACKEND.get().unwrap().lock().synchronize();

        // After executing the kernels, the Ir is cleaned up.
        // To do so, we first decrement the refcount and then set the ParamType to Input and op to
        // Data
        self.scheduled.clear();
    }
    ///
    /// Compiles the computation graph of all scheduled variables in a Ir.
    ///
    /// First, all scheduled variables with the same size are grouped.
    /// Then, a Schedule Intermediate Representation is constructed from the groups.
    /// In the end a set of Kernels is assembled and compiled.
    ///
    fn compile(&mut self, ir: &mut Ir) {
        self.schedules.clear();
        self.kernels.clear();
        let mut scheduled = self.scheduled.clone();
        scheduled.sort_by(|id0, id1| ir.var(*id0).size.cmp(&ir.var(*id1).size));

        // For every scheduled variable (destination) we have to create a new buffer
        for id in scheduled.iter_mut() {
            let mut var = ir.var_mut(*id);
            var.param_ty = ParamType::Output;
            var.buffer = Some(
                BACKEND
                    .get()
                    .unwrap()
                    .lock()
                    .buffer_uninit(var.size * var.ty.size()),
            );
        }

        let cur = 0;
        let mut size;
        for i in 1..scheduled.len() {
            let var0 = ir.var(scheduled[i - 1]);
            let var1 = ir.var(scheduled[i]);
            size = var0.size;
            if var0.size != var1.size {
                let mut tmp = ScheduleIr::new(BACKEND.get().unwrap().lock().first_register(), size);
                tmp.collect_vars(ir, &scheduled[cur..i]);
                self.borrow_mut().schedules.push(tmp);
            }
        }
        size = ir.var(*scheduled.last().unwrap()).size;
        let mut tmp = ScheduleIr::new(BACKEND.get().unwrap().lock().first_register(), size);

        tmp.collect_vars(ir, &scheduled[cur..scheduled.len()]);
        self.schedules.push(tmp);

        // TODO: paralelize by populating kernels first.
        self.kernels = self
            .schedules
            .iter()
            .map(|mut s| {
                let mut kernel = BACKEND.get().unwrap().lock().new_kernel();
                kernel.assemble(&mut s);
                // kernel.compile();
                kernel
            })
            .collect::<Vec<_>>();
        for kernel in self.kernels.iter_mut() {
            kernel.compile();
        }
    }
    pub fn schedule_kernel(&mut self, i: usize) -> (&mut Box<dyn Kernel>, &mut ScheduleIr) {
        (&mut self.kernels[i], &mut self.schedules[i])
    }
}

#[cfg(test)]
mod test {}

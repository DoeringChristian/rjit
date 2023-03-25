use std::fmt::Write;
use std::sync::Arc;

use rayon::prelude::*;

use crate::backend::{Backend, Kernel};
use crate::schedule::ScheduleIr;
use crate::trace::{Ir, Op, ParamType};

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
#[derive(Debug)]
pub struct Jit {
    pub backend: Arc<dyn Backend>,
    pub schedules: Vec<ScheduleIr>,
    pub kernels: Vec<Box<dyn Kernel>>,
}

impl Jit {
    ///
    /// Construct a new Jit Compiler from a backend.
    ///
    pub fn new(backend: &Arc<dyn Backend>) -> Self {
        Self {
            backend: backend.clone(),
            schedules: vec![],
            kernels: vec![],
        }
    }
    ///
    /// Compiles the computation graph of all scheduled variables in a Ir.
    ///
    /// First, all scheduled variables with the same size are grouped.
    /// Then, a Schedule Intermediate Representation is constructed from the groups.
    /// In the end a set of Kernels is assembled and compiled.
    ///
    pub fn compile(&mut self, ir: &mut Ir) {
        self.schedules.clear();
        self.kernels.clear();
        let mut scheduled = ir.scheduled.clone();
        scheduled.sort_by(|id0, id1| ir.var(*id0).size.cmp(&ir.var(*id1).size));

        // For every scheduled variable (destination) we have to create a new buffer
        for id in scheduled.iter() {
            let var = ir.var_mut(*id);
            var.param_ty = ParamType::Output;
            var.buffer = Some(self.backend.buffer_uninit(var.size * var.ty.size()));
        }

        let cur = 0;
        let mut size;
        for i in 1..scheduled.len() {
            let var0 = ir.var(scheduled[i - 1]);
            let var1 = ir.var(scheduled[i]);
            size = var0.size;
            if var0.size != var1.size {
                let mut tmp = ScheduleIr::new(&self.backend, self.backend.first_register(), size);
                tmp.collect_vars(ir, &scheduled[cur..i]);
                self.schedules.push(tmp);
            }
        }
        size = ir.var(*scheduled.last().unwrap()).size;
        let mut tmp = ScheduleIr::new(&self.backend, self.backend.first_register(), size);

        tmp.collect_vars(ir, &scheduled[cur..scheduled.len()]);
        self.schedules.push(tmp);

        self.kernels = self
            .schedules
            .par_iter()
            .map(|mut s| {
                let mut kernel = self.backend.new_kernel();
                kernel.assemble(&mut s);
                // kernel.compile();
                kernel
            })
            .collect::<Vec<_>>();
        for kernel in self.kernels.iter_mut() {
            kernel.compile();
        }
    }
    ///
    /// Evaluates a Ir by first constructing Schedules, which are then compiled and assembled
    /// into kernels.
    ///
    /// A the end, all scheduled variables are overwritten with the calculated data.
    ///
    pub fn eval(&mut self, ir: &mut Ir) {
        self.compile(ir);
        for i in 0..self.kernels.len() {
            self.kernels[i].execute_async(&mut self.schedules[i]);
        }

        self.backend.synchronize();

        // After executing the kernels, the Ir is cleaned up.
        // To do so, we first decrement the refcount and then set the ParamType to Input and op to
        // Data
        for id in ir.scheduled.clone() {
            ir.clear_deps(id);

            let var = ir.var_mut(id);

            var.param_ty = ParamType::Input;
            var.op = Op::Data;
        }
        ir.clear_schedule();
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

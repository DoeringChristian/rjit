use std::borrow::BorrowMut;
use std::fmt::Write;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::backend::Kernel;
use crate::schedule::ScheduleIr;
use crate::trace::{Internal, VarRef, IR};
use crate::var::{Op, ParamType, VarId};

///
/// This is the default Just In Time Compiler (JIT).
///
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
    let mut jit = JIT.lock(); // always lock JIT before IR
    let mut ir = IR.lock();
    jit.eval(&mut ir);
}

pub fn schedule(refs: &[&VarRef]) {
    let mut jit = JIT.lock(); // always lock JIT before IR
    let mut ir = IR.lock();
    for r in refs {
        ir.inc_rc(r.id());
        jit.scheduled.push(r.id());
    }
}

impl Jit {
    pub fn schedule(&mut self, refs: &[&VarRef]) {
        let mut ir = IR.lock(); // IR is always locked later.
        for r in refs {
            ir.inc_rc(r.id());
            self.scheduled.push(r.id());
        }
    }
    ///
    /// Evaluates a Ir by first constructing Schedules, which are then compiled and assembled
    /// into kernels.
    ///
    /// A the end, all scheduled variables are overwritten with the calculated data.
    ///
    pub fn eval(&mut self, ir: &mut Internal) {
        self.compile(ir);
        let n_kernels = self.kernels.len();
        for i in 0..n_kernels {
            let (kernel, schedule) = self.schedule_kernel(i);
            kernel.execute_async(schedule);
        }

        ir.backend.as_ref().unwrap().synchronize();

        // After executing the kernels, the Ir is cleaned up.
        // To do so, we first decrement the refcount and then set the ParamType to Input and op to
        // Data
        for id in self.scheduled.iter() {
            let var = ir.var_mut(*id);

            // Set op and type for next kernel:
            var.param_ty = ParamType::Input;
            var.op = Op::Data;

            // Clear dependecies:
            let deps = var.deps.clone();
            var.deps.clear();

            for dep in deps {
                ir.dec_rc(dep);
            }

            ir.dec_rc(*id);
        }
        self.scheduled.clear();
    }
    ///
    /// Compiles the computation graph of all scheduled variables in a Ir.
    ///
    /// First, all scheduled variables with the same size are grouped.
    /// Then, a Schedule Intermediate Representation is constructed from the groups.
    /// In the end a set of Kernels is assembled and compiled.
    ///
    fn compile(&mut self, ir: &mut Internal) {
        if self.scheduled.len() == 0 {
            return;
        }
        self.schedules.clear();
        self.kernels.clear();
        let mut scheduled = self.scheduled.clone();
        scheduled.sort_by(|id0, id1| ir.var(*id0).size.cmp(&ir.var(*id1).size));

        // For every scheduled variable (destination) we have to create a new buffer
        for id in scheduled.iter_mut() {
            let size = ir.var(*id).size;
            let ty_size = ir.var(*id).ty.size();
            let buffer = ir.backend.as_ref().unwrap().buffer_uninit(size * ty_size);

            let mut var = ir.var_mut(*id);
            var.param_ty = ParamType::Output;
            var.buffer = Some(buffer);
        }

        let first_register = ir.backend.as_ref().unwrap().first_register();
        let cur = 0;
        let mut size;
        for i in 1..scheduled.len() {
            let var0 = ir.var(scheduled[i - 1]);
            let var1 = ir.var(scheduled[i]);
            size = var0.size;
            if var0.size != var1.size {
                let mut tmp = ScheduleIr::new(first_register, size);
                tmp.collect_vars(ir, &scheduled[cur..i]);
                self.borrow_mut().schedules.push(tmp);
            }
        }
        size = ir.var(*scheduled.last().unwrap()).size;
        let mut tmp = ScheduleIr::new(first_register, size);

        tmp.collect_vars(ir, &scheduled[cur..scheduled.len()]);
        self.schedules.push(tmp);

        // TODO: paralelize by populating kernels first.
        self.kernels = self
            .schedules
            .iter()
            .map(|mut s| {
                let mut kernel = ir.backend.as_ref().unwrap().new_kernel();
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

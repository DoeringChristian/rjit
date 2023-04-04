use std::borrow::BorrowMut;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::backend::Kernel;
use crate::schedule::ScheduleIr;
use crate::trace::{Internal, IR};
use crate::var::{Op, VarId};

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
}

#[derive(Debug)]
struct Pass {
    ids: Vec<VarId>,
    deps: Vec<VarId>,
    // access: HashMap<VarId, Access>,
    size: usize,
}

///
///  Evaluates a Ir by first constructing Schedules, which are then compiled and assembled
///  into kernels.
///
///  A the end, all scheduled variables are overwritten with the calculated data.
///
pub fn eval() {
    let mut jit = JIT.lock(); // always lock JIT before IR
    let mut ir = IR.lock();
    jit.eval(&mut ir);
}

impl Jit {
    ///
    /// Evaluates a Ir by first constructing Schedules, which are then compiled and assembled
    /// into kernels.
    ///
    /// A the end, all scheduled variables are overwritten with the calculated data.
    ///
    pub fn eval(&mut self, ir: &mut Internal) {
        // For every scheduled variable (destination) we have to create a new buffer (except if it
        // is void)
        for id in ir.scheduled.clone() {
            let var = ir.var(id);
            // Do not reallocate on scheduled variables (don't yet know if this is right)
            if var.buffer.is_none() {
                let size = ir.var(id).size;
                let ty_size = ir.var(id).ty.size();
                let buffer = ir.backend.as_ref().unwrap().buffer_uninit(size * ty_size);

                let mut var = ir.var_mut(id);
                var.buffer = Some(buffer);
            }
        }

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
        for id in ir.scheduled.clone() {
            let var = ir.var_mut(id);

            // Set op and type for next kernel:
            // var.param_ty = ParamType::Input;
            var.op = Op::Data;

            // Clear dependencies:
            let deps = var.deps.clone();
            var.deps.clear();

            for dep in deps {
                ir.dec_rc(dep);
            }

            ir.dec_rc(id);
        }
        ir.scheduled.clear();
    }
    ///
    /// Collect neccesary passes from `ir`.
    /// NOTE: We would need to correctly assign dependencies on scatter
    ///
    pub fn passes(&self, ir: &Internal) -> Vec<Pass> {
        ///
        /// Gets the dependencies of `id` in `scheduled`
        ///
        fn dependencies_in(
            ir: &Internal,
            id: VarId,
            scheduled: &HashSet<VarId>,
            deps: &mut HashSet<VarId>,
        ) {
            if scheduled.contains(&id) {
                deps.insert(id);
                return;
            }
            let var = ir.var(id);
            for dep in var.deps.iter() {
                dependencies_in(ir, *dep, scheduled, deps);
            }
        }
        let scheduled = ir.scheduled.iter().cloned().collect::<HashSet<_>>();
        let passes = ir
            .scheduled
            .iter()
            .map(|id| {
                let mut deps = HashSet::new();
                dependencies_in(ir, *id, &scheduled, &mut deps);
                let deps = deps.difference(&HashSet::from([*id])).cloned().collect();
                Pass {
                    ids: vec![*id],
                    deps,
                    size: ir.var(*id).size,
                }
            })
            .collect::<Vec<_>>();

        passes
    }
    ///
    /// Compiles the computation graph of all scheduled variables in a Ir.
    ///
    /// First, all scheduled variables with the same size are grouped.
    /// Then, a Schedule Intermediate Representation is constructed from the groups.
    /// In the end a set of Kernels is assembled and compiled.
    ///
    fn compile(&mut self, ir: &Internal) {
        if ir.scheduled.len() == 0 {
            return;
        }
        self.schedules.clear();
        self.kernels.clear();

        let passses = self.passes(&ir);
        dbg!(&passses);

        let mut scheduled = ir.scheduled.clone();
        scheduled.sort_by(|id0, id1| ir.var(*id0).size.cmp(&ir.var(*id1).size));

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

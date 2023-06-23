use std::borrow::BorrowMut;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::ops::Range;
use std::sync::Arc;

use itertools::Itertools;
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::backend::Kernel;
use crate::schedule::{Env, ScheduleIr};
use crate::trace::Internal;
use crate::var::{Data, Op};

// TODO: pooling for paralel execution
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
    pub kernels: HashMap<KernelKey, Arc<dyn Kernel>>,
    pub kernel_history: Vec<KernelHistroyEntry>,
}

#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum KernelKey {
    Hash(u128),
    Name(String),
}
impl std::fmt::Display for KernelKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelKey::Hash(hash) => write!(f, "{hash}"),
            KernelKey::Name(name) => write!(f, "{name}"),
        }
    }
}

#[derive(Debug)]
pub struct KernelHistroyEntry {
    key: KernelKey,
    size: usize,
    backend: &'static str,
}

#[derive(Debug)]
struct ScheduleGroup {
    size: usize,
    range: Range<usize>,
}

impl Jit {
    ///
    /// Evaluates a Ir by first constructing Schedules, which are then compiled and assembled
    /// into kernels.
    ///
    /// A the end, all scheduled variables are overwritten with the calculated data.
    ///
    pub fn eval(&mut self, ir: &mut Internal) {
        if ir.scheduled.len() == 0 {
            return;
        }
        let mut schedule = ir.scheduled.clone();

        // For every scheduled variable (destination) we have to create a new buffer (except if it
        // is void)
        for id in ir.scheduled.clone() {
            let var = ir.var(id);
            // Do not reallocate on scheduled variables (don't yet know if this is right)
            // TODO: better test
            if !var.data.is_buffer() && var.ty.size() > 0 {
                let size = ir.var(id).size;
                let ty_size = ir.var(id).ty.size();
                let buffer = ir.backend.as_ref().unwrap().buffer_uninit(size * ty_size);

                let mut var = ir.var_mut(id);
                var.data = Data::Buffer(buffer);
            }
        }

        schedule.sort_by(|id0, id1| ir.var(*id0).size.cmp(&ir.var(*id1).size));
        let mut schedule_groups = vec![];

        let mut current = 0;
        for i in 1..schedule.len() {
            let size = ir.var(schedule[i - 1]).size;
            if size != ir.var(schedule[i]).size {
                schedule_groups.push(ScheduleGroup {
                    size,
                    range: current..i,
                });
                current = i;
            }
        }
        schedule_groups.push(ScheduleGroup {
            size: ir.var(schedule[current]).size,
            range: current..schedule.len(),
        });

        let first_register = ir.backend.as_ref().unwrap().first_register();

        let futures = schedule_groups
            .into_iter()
            .map(|sg| {
                let mut s = ScheduleIr::new(first_register);
                let mut env = Env::default();
                s.collect_vars(&mut env, ir, &schedule[sg.range]);
                let hash = s.internal_hash();
                let key = KernelKey::Hash(hash);

                let kernel = self
                    .kernels
                    .entry(key.clone())
                    .or_insert(ir.backend.as_ref().unwrap().compile_kernel(&s, &env));

                self.kernel_history.push(KernelHistroyEntry {
                    key,
                    size: sg.size,
                    backend: kernel.backend_ident(),
                });

                kernel.execute_async(&mut env, sg.size)
            })
            .collect::<Vec<_>>();

        // TODO: synchronisation here
        for future in futures {
            future.wait();
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
            var.dirty = false;

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
    pub fn launch_kernel(&mut self, name: &str, env: &mut Env, size: usize) {
        self.kernels
            .get_mut(&KernelKey::Name(String::from(name)))
            .unwrap()
            .execute_async(env, size)
            .wait();
    }
    // pub fn launch_kernel(&self, ir: &mut Internal) {}
    ///
    /// Writes the kernel assemblies into a string which can then be checked by snapshot testing
    /// tools such as insta.
    ///
    pub fn kernel_history(&self) -> String {
        let mut s = String::new();
        writeln!(s, "Kernel History:").unwrap();
        for entry in self.kernel_history.iter() {
            writeln!(
                s,
                "Launched {} Kernel {} with {} elements",
                entry.backend, entry.key, entry.size
            )
            .unwrap();
        }

        for (hash, kernel) in self
            .kernels
            .iter()
            .sorted_by(|(hash1, _), (hash2, _)| hash1.cmp(hash2))
        {
            writeln!(s, "===============================================").unwrap();
            writeln!(s, "Kernel {}:", hash).unwrap();
            writeln!(s, "").unwrap();
            write!(s, "{}", kernel.assembly()).unwrap();
        }
        s
    }
}

#[cfg(test)]
mod test {}

use std::borrow::BorrowMut;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::ops::Range;

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
    // schedules: Vec<ScheduleIr>,
    // hashes: Vec<u128>,
    // passes: Vec<Pass>,
    pub kernels: HashMap<u128, Box<dyn Kernel>>,
}

#[derive(Debug)]
struct ScheduleGroup {
    size: usize,
    range: Range<usize>,
}
#[derive(Debug)]
struct KernelLaunch {
    size: usize,
    hash: u128,
    env: Env,
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

        let launches = schedule_groups
            .into_iter()
            .map(|sg| {
                let mut s = ScheduleIr::new(first_register);
                let mut env = Env::default();
                s.collect_vars(&mut env, ir, &schedule[sg.range]);
                let hash = s.internal_hash();
                if !self.kernels.contains_key(&hash) {
                    self.kernels.insert(hash, {
                        ir.backend.as_ref().unwrap().compile_kernel(&s, &env)
                    });
                };
                KernelLaunch {
                    size: sg.size,
                    hash,
                    env,
                }
            })
            .collect::<Vec<_>>();

        let futures = launches.into_iter().map(|mut launch| {
            self.kernels
                .get_mut(&launch.hash)
                .unwrap()
                .execute_async(&mut launch.env, launch.size)
        });

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
    /// Writes the kernel assemblies into a string which can then be checked by snapshot testing
    /// tools such as insta.
    ///
    pub fn kernel_debug(&self) -> String {
        let mut kernel_strings = self
            .kernels
            .iter()
            .map(|(hash, k)| {
                let mut string = String::new();
                writeln!(string, "===============================================").unwrap();
                writeln!(string, "Kernel {}:", hash).unwrap();
                writeln!(string, "").unwrap();
                write!(string, "{}", k.assembly()).unwrap();
                (string, hash)
            })
            .collect::<Vec<_>>();

        kernel_strings.sort_by(|(_, hash0), (_, hash1)| hash0.cmp(hash1));

        let string = kernel_strings.into_iter().map(|(string, _)| string).fold(
            String::new(),
            |mut s0, s1| {
                s0.push_str(&s1);
                s0
            },
        );

        string
    }
}

#[cfg(test)]
mod test {}

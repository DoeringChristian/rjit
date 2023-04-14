use std::borrow::BorrowMut;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::backend::Kernel;
use crate::schedule::ScheduleIr;
use crate::trace::Internal;
use crate::var::{Op, VarId};

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
    // pub schedules: Vec<ScheduleIr>,
    pub kernels: HashMap<u128, Box<dyn Kernel>>,
    // pub hashes: Vec<u128>,
}

///
/// This represents the Evaluation of one or many variables.
/// With dependencies on other variables which have been scheduled.
/// TODO: Better naming
///
/// Dependencies of a scatter operation have to be set up in the following way:
///  a  <---------- b
/// ^ ^------ s1 <--|
/// '-- s0 <--|
///
/// A is an arbitrary operation. If s0 and s1 are scattering operations, a has to be scheduled as
/// do s0 and s1. In order for s0 and s1 to be evaluated before b, b needs to depend on s1 and s1
/// on s0.
/// This can be done by adding extra dependencies to b and s1.
///
///
#[derive(Debug)]
struct Pass {
    ids: Vec<VarId>,
    deps: Vec<usize>,
    size: usize,
}
///
/// Does a depend on b?
///
fn depends_on(passes: &[Pass], a: usize, b: usize) -> bool {
    if a == b {
        return true;
    }
    for dep in passes[b].deps.iter() {
        if depends_on(passes, a, *dep) {
            return true;
        }
    }
    return false;
}
///
/// Try to merge src into dst.
///
fn try_merge(passes: &mut Vec<Pass>, dst: usize, src: usize) -> bool {
    if passes[dst].size != passes[src].size {
        return false;
    }
    if depends_on(&passes, dst, src) {
        return false;
    }

    let src_pass = passes.remove(src);

    for pass in passes[src..].iter_mut() {
        for dep in pass.deps.iter_mut() {
            if *dep == src {
                *dep = dst;
            } else if *dep > src {
                *dep -= 1;
            }
        }
    }

    passes[dst].deps.extend_from_slice(&src_pass.deps);
    passes[dst].ids.extend_from_slice(&src_pass.ids);

    true
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
            if var.buffer.is_none() && var.ty.size() > 0 {
                let size = ir.var(id).size;
                let ty_size = ir.var(id).ty.size();
                let buffer = ir.backend.as_ref().unwrap().buffer_uninit(size * ty_size);

                let mut var = ir.var_mut(id);
                var.buffer = Some(buffer);
            }
        }

        let schedules = self.compile(ir);
        for (hash, pass, mut schedule) in schedules {
            self.kernels
                .get_mut(&hash)
                .unwrap()
                .execute_async(&mut schedule);
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
    /// NOTE: Passes are in ordered (DAG ordering)
    ///
    fn passes(&self, ir: &Internal) -> Vec<Pass> {
        ///
        /// Gets the dependencies of `id` in `scheduled`
        ///
        fn dependencies_in(
            ir: &Internal,
            id: VarId,
            scheduled: &HashSet<VarId>,
            deps: &mut Vec<VarId>,
        ) {
            if scheduled.contains(&id) {
                deps.push(id);
            }
            let var = ir.var(id);
            for dep in var.deps.iter() {
                dependencies_in(ir, *dep, scheduled, deps);
            }
        }
        let scheduled = ir.scheduled.iter().cloned().collect::<HashSet<_>>();
        let mut id2pass = HashMap::new();
        let passes = ir
            .scheduled
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let mut deps = Vec::new();
                dependencies_in(ir, *id, &scheduled, &mut deps);
                let deps = deps
                    .into_iter()
                    .filter(|dep| dep != id)
                    .map(|dep| id2pass[&dep])
                    .collect();
                id2pass.insert(*id, i);
                Pass {
                    ids: vec![*id],
                    deps,
                    size: ir.var(*id).size,
                }
            })
            .collect::<Vec<_>>();

        // TODO: use passes to flatten dependencies and stop traversal eraly.

        passes
    }
    ///
    /// Compiles the computation graph of all scheduled variables in a Ir.
    ///
    /// First, all scheduled variables with the same size are grouped.
    /// Then, a Schedule Intermediate Representation is constructed from the groups.
    /// In the end a set of Kernels is assembled and compiled.
    ///
    fn compile(&mut self, ir: &Internal) -> Vec<(u128, Pass, ScheduleIr)> {
        if ir.scheduled.len() == 0 {
            return vec![];
        }
        // self.schedules.clear();
        // self.kernels.clear();

        // The problem we try to solve is the following:
        //
        // The nodes in `ir` are represent DAG, so do the `passes`.
        // They are also stored in a topological ordering.
        // We try to merge as many passes as possible.
        // Passes can only be merged if tey have the same size.
        //
        // This algorithm is not perfect and results in a potentially sub optimal result.
        // Also, using hash sets might not be the best approach.
        let mut passes = self.passes(&ir);

        for i in (0..passes.len()).rev() {
            for j in (0..i).rev() {
                // Try to merge i into j
                if try_merge(&mut passes, j, i) {
                    break;
                }
            }
        }
        let first_register = ir.backend.as_ref().unwrap().first_register();
        let schedules = passes
            .into_iter()
            .map(|pass| {
                let mut s = ScheduleIr::new(first_register, pass.size);
                s.collect_vars(ir, &pass.ids.iter().cloned().collect::<Vec<_>>());

                let hash = s.internal_hash();
                let kernel = self.kernels.entry(hash).or_insert({
                    let mut kernel = ir.backend.as_ref().unwrap().new_kernel();
                    kernel.assemble(&mut s);
                    kernel.compile();
                    kernel
                });
                (hash, pass, s)
            })
            .collect::<Vec<_>>();

        schedules
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

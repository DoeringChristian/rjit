use std::borrow::BorrowMut;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::backend::Kernel;
use crate::schedule::{Env, ScheduleIr};
use crate::trace::Internal;
use crate::var::{Data, Op, VarId};

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

///
/// An `ExecutionGraph` records The high level operations that need to be executed.
/// TODO: Since this is a per eval structure, it would be preferable to pool the passes.
///
struct ExecutionGraph {
    passes: Vec<Pass>,
}

impl ExecutionGraph {
    ///
    /// Collect neccesary passes from `ir`.
    /// NOTE: We would need to correctly assign dependencies on scatter
    /// NOTE: Passes are ordered (DAG ordering)
    ///
    fn new(ir: &Internal) -> Self {
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
                let op = if ir.var(*id).op == Op::TexUpload {
                    PassOp::TexUpload
                } else {
                    PassOp::KernelLaunch(0, Env::default())
                };
                id2pass.insert(*id, i);
                Pass {
                    ids: vec![*id],
                    deps,
                    size: ir.var(*id).size,
                    op,
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>();
        Self { passes }
    }

    /// Merge passes if possible
    ///
    /// The problem we try to solve is the following:
    ///
    /// The nodes in `ir` are represent DAG, so do the `passes`.
    /// They are also stored in a topological ordering.
    /// We try to merge as many passes as possible.
    /// Passes can only be merged if tey have the same size.
    ///
    /// This algorithm is not perfect and results in a potentially sub optimal result.
    /// Also, using hash sets might not be the best approach.
    fn simplify(&mut self) {
        // TODO: use passes to flatten dependencies and stop traversal eraly.
        for i in (0..self.passes.len()).rev() {
            for j in (0..i).rev() {
                // Try to merge i into j
                if self.try_merge(j, i) {
                    break;
                }
            }
        }
    }
    ///
    /// Does a depend on b?
    ///
    fn depends_on(&self, a: usize, b: usize) -> bool {
        if a == b {
            return true;
        }
        for dep in self.passes[b].deps.iter() {
            if self.depends_on(a, *dep) {
                return true;
            }
        }
        return false;
    }
    ///
    /// Try to merge src into dst.
    ///
    fn try_merge(&mut self, dst: usize, src: usize) -> bool {
        if self.passes[dst].size != self.passes[src].size {
            return false;
        }
        if self.depends_on(dst, src) {
            return false;
        }
        if !self.same_op(dst, src) {
            return false;
        }

        let src_pass = self.passes.remove(src);

        for pass in self.passes[src..].iter_mut() {
            for dep in pass.deps.iter_mut() {
                if *dep == src {
                    *dep = dst;
                } else if *dep > src {
                    *dep -= 1;
                }
            }
        }

        self.passes[dst].deps.extend_from_slice(&src_pass.deps);
        self.passes[dst].ids.extend_from_slice(&src_pass.ids);

        true
    }

    fn same_op(&self, a: usize, b: usize) -> bool {
        match self.passes[a].op {
            PassOp::None => match self.passes[b].op {
                PassOp::None => true,
                _ => false,
            },
            PassOp::KernelLaunch(..) => match self.passes[b].op {
                PassOp::KernelLaunch(..) => true,
                _ => false,
            },
            PassOp::TexUpload => match self.passes[b].op {
                PassOp::TexUpload => true,
                _ => false,
            },
        }
    }
}

#[derive(Debug, Default)]
enum PassOp {
    #[default]
    None,
    KernelLaunch(u128, Env),
    TexUpload,
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
#[derive(Debug, Default)]
struct Pass {
    ids: Vec<VarId>,  // Variables to evaluate
    deps: Vec<usize>, // Passes on which this depends on
    size: usize,      // Number of threads for this execution
    op: PassOp,       // Operation that needs to be performed
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
            // TODO: better test
            if !var.data.is_buffer() && var.ty.size() > 0 {
                let size = ir.var(id).size;
                let ty_size = ir.var(id).ty.size();
                let buffer = ir.backend.as_ref().unwrap().buffer_uninit(size * ty_size);

                let mut var = ir.var_mut(id);
                var.data = Data::Buffer(buffer);
            }
        }

        let graph = if let Some(graph) = self.compile(ir) {
            graph
        } else {
            return;
        };

        for mut pass in graph.passes {
            match &mut pass.op {
                PassOp::KernelLaunch(hash, env) => {
                    self.kernels
                        .get_mut(&hash)
                        .unwrap()
                        .execute_async(env, pass.size);
                }
                PassOp::TexUpload => {
                    for id in pass.ids.iter() {
                        let dep = ir.var(*id).deps[0];
                        let buf = ir.var(dep).data.buffer().unwrap().clone();
                        ir.var(*id)
                            .data
                            .texture()
                            .unwrap()
                            .copy_from_buffer(buf.as_ref());
                    }
                }
                _ => {
                    todo!()
                }
            }
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
    /// Compiles the computation graph of all scheduled variables in a Ir.
    ///
    /// First, all scheduled variables with the same size are grouped.
    /// Then, a Schedule Intermediate Representation is constructed from the groups.
    /// In the end a set of Kernels is assembled and compiled.
    ///
    fn compile(&mut self, ir: &Internal) -> Option<ExecutionGraph> {
        if ir.scheduled.len() == 0 {
            return None;
        }

        let mut graph = ExecutionGraph::new(&ir);
        graph.simplify();

        let first_register = ir.backend.as_ref().unwrap().first_register();
        for pass in graph.passes.iter_mut() {
            match &mut pass.op {
                PassOp::KernelLaunch(ref mut hash, env) => {
                    let mut s = ScheduleIr::new(first_register);
                    s.collect_vars(env, ir, &pass.ids.iter().cloned().collect::<Vec<_>>());

                    *hash = s.internal_hash();
                    if !self.kernels.contains_key(hash) {
                        self.kernels.insert(*hash, {
                            let mut kernel = ir.backend.as_ref().unwrap().new_kernel();
                            kernel.assemble(&s, env);
                            kernel.compile();
                            kernel
                        });
                    }
                }
                PassOp::TexUpload => {}
                _ => {
                    todo!()
                }
            }
        }
        Some(graph)
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

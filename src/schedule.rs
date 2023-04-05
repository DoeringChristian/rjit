use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use smallvec::{smallvec, SmallVec};

use crate::backend::Buffer;
use crate::trace::Internal;
use crate::var::{Op, ParamType, VarId, VarType};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SVarId(pub usize);
impl std::fmt::Display for SVarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

///
/// A Representation of a variable used in the ScheduleIr.
/// This only holds data that is needed to compile the Kernel.
///
/// Variables are densly stored in the ScheduleIr, simplifying the compilation.
///
#[derive(Debug, Default)]
pub struct ScheduleVar {
    pub op: Op,
    pub deps: SmallVec<[SVarId; 4]>,
    pub ty: VarType,
    pub param_ty: ParamType,
    pub reg: usize,
    pub param: usize,    // Index into literal/buffer/texture vec
    pub gs_param: usize, // Parameter offset for gather/scatter operation
    pub literal: u64,
    pub size: usize,
}

impl ScheduleVar {
    pub fn is_literal(&self) -> bool {
        self.op == Op::Literal
    }
}

///
/// Helper struct for printing register names.
/// <prefix><register_index>
///
pub struct Reg<'a>(pub &'a ScheduleVar);
impl<'a> std::fmt::Display for Reg<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.0.ty.prefix(), self.0.reg)
    }
}
impl ScheduleVar {
    ///
    /// Returns a helper struct (Reg) that can be displayed with the correct prefix for the
    /// variable type.
    ///
    pub fn reg(&self) -> Reg {
        Reg(self)
    }
    ///
    /// Returns the raw index of the register for this variable.
    ///
    pub fn reg_idx(&self) -> usize {
        self.reg
    }
}

///
/// Intermediate representation for scheduled variables
///
#[derive(Debug, Default)]
pub struct ScheduleIr {
    vars: Vec<ScheduleVar>,
    size: usize,
    literals: Vec<u64>,            // Literals (directly passed to the kernel)
    buffers: Vec<Arc<dyn Buffer>>, // Buffers referenced in the kernel
    n_regs: usize,
    visited: HashMap<VarId, SVarId>,
}

impl ScheduleIr {
    pub fn new(first_register: usize, size: usize) -> Self {
        Self {
            n_regs: first_register,
            size,
            ..Default::default()
        }
    }
    pub fn ids(&self) -> impl Iterator<Item = SVarId> {
        (0..self.vars.len()).map(|i| SVarId(i))
    }
    pub fn var(&self, id: SVarId) -> &ScheduleVar {
        &self.vars[id.0]
    }
    pub fn var_mut(&mut self, id: SVarId) -> &mut ScheduleVar {
        &mut self.vars[id.0]
    }
    pub fn reg(&self, id: SVarId) -> Reg {
        self.var(id).reg()
    }
    pub fn buffers(&self) -> &[Arc<dyn Buffer>] {
        &self.buffers
    }
    pub fn n_regs(&self) -> usize {
        self.n_regs
    }
    fn next_reg(&mut self) -> usize {
        let reg = self.n_regs;
        self.n_regs += 1;
        reg
    }
    pub fn size(&self) -> usize {
        self.size
    }
    fn push_var(&mut self, var: ScheduleVar) -> SVarId {
        let id = SVarId(self.vars.len());
        self.vars.push(var);
        id
    }
    /// TODO: optimization to not push buffer twice
    fn push_buffer(&mut self, buf: &Arc<dyn Buffer>) -> usize {
        let idx = self.buffers.len();
        self.buffers.push(buf.clone());
        idx
    }
    pub fn collect_vars(&mut self, ir: &Internal, schedule: &[VarId]) {
        for id in schedule {
            let sv_id = self.collect(ir, *id);

            let var = ir.var(*id);
            if var.ty.size() == 0 {
                continue;
            }
            let param = self.push_buffer(var.buffer.as_ref().unwrap());

            let mut sv = self.var_mut(sv_id);

            sv.param_ty = ParamType::Output;
            sv.param = param;
        }
    }
    ///
    /// Traverse computation graph and collect variables into Schedule.
    ///
    /// If a gather operation is encountered, that only depends on trivial operations we can
    /// reindex it using the parameter idx.
    ///
    pub fn collect(&mut self, ir: &Internal, id: VarId) -> SVarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = ir.var(id);

        let mut sv = ScheduleVar {
            op: var.op,
            ty: var.ty.clone(),
            deps: smallvec![],
            reg: self.next_reg(),
            param_ty: ParamType::None,
            param: 0,
            gs_param: 0,
            literal: var.literal,
            size: var.size,
        };

        // Collect dependencies

        match var.op {
            Op::Data => {
                sv.param = self.push_buffer(var.buffer.as_ref().unwrap());
                sv.param_ty = ParamType::Input;
            }
            Op::Literal => {
                // sv.param_offset = self.push_param(var.literal);
                sv.literal = var.literal;
            }
            Op::Gather => {
                let src = ir.var(var.deps[0]);

                sv.gs_param = self.push_buffer(src.buffer.as_ref().unwrap());

                // Then: collect index and mask.
                sv.deps = smallvec![
                    // src
                    self.collect(ir, var.deps[1]), // index
                    self.collect(ir, var.deps[2])  // mask
                ];
            }
            Op::Scatter => {
                let dst = ir.var(var.deps[1]);
                dbg!(&ir);
                dbg!(var.deps[1]);
                sv.gs_param = self.push_buffer(dst.buffer.as_ref().unwrap());

                sv.deps = smallvec![
                    self.collect(ir, var.deps[0]), // src
                    // dst
                    self.collect(ir, var.deps[2]), // index
                    self.collect(ir, var.deps[3])  // mask
                ];
            }
            _ => {
                sv.deps = var
                    .deps
                    .iter()
                    .map(|id| self.collect(ir, *id))
                    .collect::<SmallVec<[_; 4]>>();
            }
        }

        let svid = self.push_var(sv);

        self.visited.insert(id, svid);

        svid
    }
}

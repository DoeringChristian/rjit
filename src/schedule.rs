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
    pub param_offset: usize,
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
    params: Vec<u64>,
    literals: Vec<u64>,            // Literals (directly passed to the kernel)
    buffers: Vec<Arc<dyn Buffer>>, // Buffers referenced in the kernel
    n_regs: usize,
    visited: HashMap<VarId, SVarId>,
}

impl ScheduleIr {
    pub fn new(first_register: usize, size: usize) -> Self {
        Self {
            n_regs: first_register,
            params: vec![size as _],
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
    // pub fn lit(&self, id: SVarId) -> Literal {
    //     self.var(id).lit()
    // }
    pub fn n_params(&self) -> usize {
        self.params.len()
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
        self.params[0] as usize
    }
    pub fn params(&self) -> &[u64] {
        &self.params
    }
    pub fn params_mut(&mut self) -> &mut [u64] {
        &mut self.params
    }
    fn push_var(&mut self, var: ScheduleVar) -> SVarId {
        let id = SVarId(self.vars.len());
        self.vars.push(var);
        id
    }
    fn push_param(&mut self, param: u64) -> usize {
        let idx = self.params.len();
        self.params.push(param);
        idx
    }
    pub fn collect_vars(&mut self, ir: &Internal, schedule: &[VarId]) {
        for id in schedule {
            let sv_id = self.collect(ir, *id);

            let var = ir.var(*id);
            let buffer_ref = var.buffer.as_ref().unwrap().as_ptr();
            let param_offset = self.push_param(buffer_ref);

            let mut sv = self.var_mut(sv_id);

            sv.param_ty = ParamType::Output;
            sv.param_offset = param_offset;
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
            param_offset: 0,
            literal: var.literal,
            size: var.size,
        };

        // Collect dependencies

        match var.op {
            Op::Data => {
                sv.param_offset = self.push_param(var.buffer.as_ref().unwrap().as_ptr());
                sv.param_ty = ParamType::Input;
            }
            Op::Literal => {
                // sv.param_offset = self.push_param(var.literal);
                sv.literal = var.literal;
            }
            Op::Gather => {
                // Fisrt: push source ptr literal.
                let d0 = ir.var(var.deps[0]);
                assert_eq!(d0.op, Op::Data);

                let param_offset = self.push_param(d0.buffer.as_ref().unwrap().as_ptr());
                let reg = self.next_reg();

                let d0 = self.push_var(ScheduleVar {
                    op: Op::Literal,
                    ty: VarType::U64,
                    deps: smallvec![],
                    reg,
                    param_ty: ParamType::Input,
                    literal: 0,
                    param_offset,
                    size: var.size,
                });
                // Then: collect index and mask.
                sv.deps = smallvec![
                    d0,
                    self.collect(ir, var.deps[1]),
                    self.collect(ir, var.deps[2])
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

        let id = self.push_var(sv);

        id
    }
}

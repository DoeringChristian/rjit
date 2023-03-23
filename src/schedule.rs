use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use cust::util::SliceExt;
use smallvec::SmallVec;

use crate::backend::Backend;
use crate::compiler::CUDAKernel;
use crate::trace::{Ir, Op, ParamType, VarId, VarType};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SVarId(pub usize);
impl std::fmt::Display for SVarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Default)]
pub struct ScheduleVar {
    pub op: Op,
    pub deps: SmallVec<[SVarId; 4]>,
    pub ty: VarType,
    pub param_ty: ParamType,
    pub reg: usize,
    pub param_offset: usize,
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
    pub fn reg(&self) -> Reg {
        Reg(self)
    }
}

///
/// Intermediate representation for scheduled variables
///
// #[derive(Default)]
pub struct ScheduleIr {
    vars: Vec<ScheduleVar>,
    params: Vec<u64>,
    n_regs: usize,
    visited: HashMap<VarId, SVarId>,
    backend: Arc<dyn Backend>,
}

impl Debug for ScheduleIr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScheduleIr")
            .field("vars", &self.vars)
            .field("params", &self.params)
            .field("n_regs", &self.n_regs)
            .field("visited", &self.visited)
            // .field("backend", &self.backend)
            .finish()
    }
}

impl ScheduleIr {
    pub fn new(backend: &Arc<dyn Backend>, first_register: usize, size: usize) -> Self {
        Self {
            n_regs: first_register,
            params: vec![size as _],
            vars: Default::default(),
            visited: Default::default(),
            backend: backend.clone(),
        }
    }
    pub fn ids(&self) -> impl Iterator<Item = SVarId> {
        (0..self.vars.len()).map(|i| SVarId(i))
    }
    pub fn var(&self, id: SVarId) -> &ScheduleVar {
        &self.vars[id.0]
    }
    pub fn reg(&self, id: SVarId) -> Reg {
        self.var(id).reg()
    }
    pub fn n_params(&self) -> usize {
        self.params.len()
    }
    pub fn n_regs(&self) -> usize {
        self.n_regs
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
    fn push_var(&mut self, mut var: ScheduleVar) -> SVarId {
        let id = SVarId(self.vars.len());
        self.vars.push(var);
        id
    }
    fn push_param(&mut self, param: u64) -> usize {
        let idx = self.params.len();
        self.params.push(param);
        idx
    }
    pub fn collect_vars(&mut self, ir: &mut Ir, ids: &[VarId]) {
        for id in ids {
            self.collect(ir, *id);
        }
    }
    pub fn collect(&mut self, ir: &mut Ir, id: VarId) -> SVarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }
        let var = ir.var(id);

        // Collect dependencies
        let deps = var
            .deps
            .clone()
            .into_iter()
            .map(|id| self.collect(ir, id))
            .collect::<SmallVec<[_; 4]>>();

        let var = ir.var_mut(id);

        let mut param_offset = 0;
        let reg = self.n_regs;
        self.n_regs += 1;
        if var.param_ty == ParamType::Input {
            // TODO: This should be compatible with diffrent backends
            let offset = self.push_param(var.buffer.as_ref().unwrap().as_ptr());
            param_offset = offset;
        } else if var.param_ty == ParamType::Output {
            var.buffer = Some(self.backend.buffer_uninit(var.size * var.ty.size()));
            let offset = self.push_param(var.buffer.as_ref().unwrap().as_ptr());
            param_offset = offset;
        } else {
        }

        let id = self.push_var(ScheduleVar {
            op: var.op,
            ty: var.ty.clone(),
            deps,
            param_ty: var.param_ty,
            reg,
            param_offset,
        });

        id
    }
}

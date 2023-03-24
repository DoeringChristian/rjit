use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use smallvec::{smallvec, SmallVec};

use crate::backend::Backend;
use crate::trace::{Ir, Op, ParamType, VarId, VarType};

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
    pub fn reg(&self) -> Reg {
        Reg(self)
    }
    pub fn reg_idx(&self) -> usize {
        self.reg
    }
    // pub fn lit(&self) -> Literal {
    //     Literal(self)
    // }
}

///
/// Intermediate representation for scheduled variables
///
// #[derive(Default)]
pub struct ScheduleIr {
    vars: Vec<ScheduleVar>,
    params: Vec<u64>,
    n_regs: usize,
    // The index into the hashmap consists of
    // the variable id and an optional index for
    // reindexing.
    visited: HashMap<(VarId, Option<SVarId>), SVarId>,
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
    pub fn collect_vars(&mut self, ir: &mut Ir, ids: &[VarId]) {
        for id in ids {
            self.collect(ir, *id, None);
        }
    }
    fn is_trivial(ir: &Ir, id: VarId, is_trivial: bool) -> bool {
        let var = ir.var(id);
        if var.param_ty == ParamType::Input {
            return false;
        } else {
            var.deps
                .iter()
                .map(|id| Self::is_trivial(ir, *id, is_trivial))
                .fold(true, |a, b| a && b)
        }
    }
    ///
    /// Traverse computation graph and collect variables into Schedule.
    ///
    /// If a gather operation is encountered, that only depends on trivial operations we can
    /// reindex it using the parameter idx.
    ///
    pub fn collect(&mut self, ir: &Ir, id: VarId, idx: Option<SVarId>) -> SVarId {
        if self.visited.contains_key(&(id, idx)) {
            return self.visited[&(id, idx)];
        }

        let var = ir.var(id);

        // Apply reindexing if we encountered an Index and are reindexing.
        // Input variables cannot be reindexed.
        if idx.is_some() {
            if var.param_ty == ParamType::Input {
                panic!("We cannot reindex input variables");
            }
            if var.op == Op::Idx {
                return idx.unwrap();
            }
        }

        let mut sv = ScheduleVar {
            op: var.op,
            ty: var.ty.clone(),
            deps: smallvec![],
            reg: self.next_reg(),
            param_ty: var.param_ty,
            param_offset: 0,
            literal: var.literal,
            size: var.size,
        };

        // Collect dependencies
        if var.stop_traversal {
            sv.deps = smallvec![]
        } else {
            sv.deps = var
                .deps
                .iter()
                .map(|id| self.collect(ir, *id, idx))
                .collect::<SmallVec<[_; 4]>>();
        };
        match var.param_ty {
            ParamType::Input => {
                if var.is_literal() {
                    sv.param_offset = self.push_param(var.literal);

                    // Literal is pushed via params => we can set it to zero so that snapshot
                    // testing works.
                    sv.literal = 0;
                } else {
                    sv.param_offset = self.push_param(var.buffer.as_ref().unwrap().as_ptr());
                }
            }
            ParamType::Output => {
                sv.param_offset = self.push_param(var.buffer.as_ref().unwrap().as_ptr());
            }
            ParamType::None => {}
        }

        let id = self.push_var(sv);

        id
    }
}

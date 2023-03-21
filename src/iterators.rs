use std::collections::HashSet;

use crate::ir::{Ir, Op, VarId};

pub struct DepIterOp {
    op: Op,
    i: usize,
}

impl DepIterOp {
    pub fn new(op: Op) -> Self {
        Self { i: 0, op }
    }
    pub fn next_id<const N: usize>(&mut self, ids: [VarId; N]) -> Option<VarId> {
        let ret = ids.get(self.i).map(|id| *id);
        self.i += 1;
        ret
    }
}

impl Iterator for DepIterOp {
    type Item = VarId;

    fn next(&mut self) -> Option<Self::Item> {
        match self.op {
            Op::Add(lhs, rhs) => self.next_id([lhs, rhs]),
            _ => None,
        }
    }
}

pub struct DepIter<'a> {
    pub ir: &'a Ir,
    pub stack: Vec<VarId>,
    pub discovered: HashSet<VarId>,
}

impl<'a> Iterator for DepIter<'a> {
    type Item = VarId;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(id) = self.stack.pop() {
            if !self.discovered.contains(&id) {
                let var = self.ir.var(id);
                for id in var.op.deps() {
                    if !self.discovered.contains(&id) {
                        self.stack.push(id);
                    }
                }
                self.discovered.insert(id);
                return Some(id);
            }
        }
        None
    }
}

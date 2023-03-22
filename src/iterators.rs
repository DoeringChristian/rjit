use std::collections::HashSet;

use crate::ir::{Ir, Op, VarId};

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
                for id in var.deps().iter().rev() {
                    if !self.discovered.contains(id) {
                        self.stack.push(*id);
                    }
                }
                self.discovered.insert(id);
                return Some(id);
            }
        }
        None
    }
}

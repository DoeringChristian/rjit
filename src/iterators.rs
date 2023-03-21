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
            Op::Nop => None,
            Op::Neg(src) => self.next_id([src]),
            Op::Not(src) => self.next_id([src]),
            Op::Sqrt(src) => self.next_id([src]),
            Op::Abs(src) => self.next_id([src]),
            Op::Add(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Sub(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Mul(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Div(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Mod(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Mulhi(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Fma(x, y, z) => self.next_id([x, y, z]),
            Op::Min(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Max(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Cail(src) => self.next_id([src]),
            Op::Floor(src) => self.next_id([src]),
            Op::Round(src) => self.next_id([src]),
            Op::Trunc(src) => self.next_id([src]),
            Op::Eq(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Neq(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Lt(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Le(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Gt(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Ge(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Select(c, x, y) => self.next_id([c, x, y]),
            Op::Popc(src) => self.next_id([src]),
            Op::Clz(src) => self.next_id([src]),
            Op::Ctz(src) => self.next_id([src]),
            Op::And(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Or(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Xor(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Shl(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Shr(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Rcp(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Rsqrt(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Sin(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Cos(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Exp2(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Log2(lhs, rhs) => self.next_id([lhs, rhs]),
            Op::Cast(src) => self.next_id([src]),
            Op::Bitcast(src) => self.next_id([src]),
            Op::Gather { from, idx, mask } => self.next_id([from, idx, mask]),
            Op::Scatter {
                from,
                to,
                idx,
                mask,
            } => self.next_id([from, to, idx, mask]),
            Op::Idx => None,
            Op::ConstF32(src) => None,
            Op::ConstU32(src) => None,
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

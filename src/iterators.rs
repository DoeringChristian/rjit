use crate::ir::{Op, VarId};

struct DepIter {
    op: Op,
    i: usize,
}

impl DepIter {
    pub fn next_id(&mut self, ids: &[VarId]) -> Option<VarId> {
        let ret = ids.get(self.i).map(|id| *id);
        self.i += 1;
        ret
    }
}

impl Iterator for DepIter {
    type Item = VarId;

    fn next(&mut self) -> Option<Self::Item> {
        match self.op {
            Op::Add(lhs, rhs) => self.next_id(&[lhs, rhs]),
            _ => None,
        }
    }
}

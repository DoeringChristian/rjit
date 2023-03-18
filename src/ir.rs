#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add(VarId, VarId),
    ConstF32(f32),
}

#[derive(Debug, Clone, Copy)]
pub struct Var {
    pub op: Op,
}

#[derive(Clone, Copy, Debug)]
pub struct VarId(usize);

#[derive(Debug, Default)]
pub struct Ir {
    vars: Vec<Var>,
}

impl Ir {
    pub fn push_var(&mut self, var: Var) -> VarId {
        let id = self.vars.len();
        self.vars.push(var);
        VarId(id)
    }
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    pub fn var_mut(&mut self, id: VarId) -> &mut Var {
        &mut self.vars[id.0]
    }
}

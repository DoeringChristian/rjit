#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add(VarId, VarId),
    ConstF32(f32),
}

#[derive(Clone, Copy, Debug)]
pub enum VarType {
    F32,
}
impl VarType {
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::F32 => "%f",
        }
    }
    pub fn name_cuda(&self) -> &'static str {
        match self {
            VarType::F32 => "f32",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Var {
    pub op: Op,
    pub ty: VarType,
    // pub id: VarId,
}

#[derive(Clone, Copy, Debug)]
pub struct VarId(pub usize);

impl std::fmt::Display for VarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Default)]
pub struct Ir {
    vars: Vec<Var>,
}

pub struct PVar<'a>(pub VarId, pub &'a Var);
impl<'a> std::fmt::Display for PVar<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.1.ty.prefix(), self.0)
    }
}

impl Ir {
    pub fn push_var(&mut self, mut var: Var) -> VarId {
        let id = VarId(self.vars.len());
        self.vars.push(var);
        id
    }
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    pub fn var_mut(&mut self, id: VarId) -> &mut Var {
        &mut self.vars[id.0]
    }
    pub fn pvar(&self, id: VarId) -> PVar {
        PVar(id, self.var(id))
    }
    pub fn ids(&self) -> impl Iterator<Item = VarId> {
        (0..self.vars.len()).map(|i| VarId(i))
    }
}

use cust::prelude::DeviceBuffer;

pub const VAR_OFFSET: usize = 4; // register offset of variables

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add(VarId, VarId),
    ConstF32(f32),
    Load(usize),
    Store(VarId, usize),
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
            Self::F32 => "f32",
        }
    }
    pub fn size(&self) -> u64 {
        match self {
            Self::F32 => 4,
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
        write!(f, "{}", self.0 + VAR_OFFSET)
    }
}

#[derive(Debug)]
pub struct Ir {
    vars: Vec<Var>,
    pub params: Vec<u64>,
}

///
/// Intermediate Representation:
///
/// The structure sould be something like this...
///
/// Trace -> Ir -> CUDA Ptx
///
///
///
impl Default for Ir {
    fn default() -> Self {
        Self {
            vars: Default::default(),
            params: vec![0],
        }
    }
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
    pub fn push_param(&mut self, param: u64) -> usize {
        let id = self.params.len();
        self.params.push(param);
        id
    }
    pub fn vars(&self) -> &[Var] {
        &self.vars
    }
    pub fn set_size(&mut self, size: u64) {
        self.params[0] = size;
    }
}

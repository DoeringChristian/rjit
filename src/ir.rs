use cust::prelude::DeviceBuffer;

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
    Ptr,
}
impl VarType {
    // Returns the register prefix for this variable
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::F32 => "%f",
            Self::Ptr => "%rd",
        }
    }
    // Retuns the cuda/ptx Representation for this type
    pub fn name_cuda(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Ptr => "u64",
        }
    }
    // Returns the size/stride of this variable
    pub fn size(&self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Ptr => 8,
        }
    }
}

///
///
///
#[derive(Debug, Clone, Copy)]
pub struct Var {
    pub op: Op,      // Operation used to construct the variable
    pub ty: VarType, // Type of the variable
    pub reg: usize,  // Register Index of that variable
}

impl Var {
    pub fn reg(&self) -> Reg {
        Reg(self)
    }
}

///
/// Helper struct for printing register names.
/// <prefix><register_index>
///
pub struct Reg<'a>(pub &'a Var);
impl<'a> std::fmt::Display for Reg<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.0.ty.prefix(), self.0.reg)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VarId(pub usize);

impl std::fmt::Display for VarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct Ir {
    vars: Vec<Var>,
    pub params: Vec<u64>, // Params vec![size, &buffer0, &buffer1]
    pub n_regs: usize,    // Next register index to use
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
            n_regs: Self::FIRST_REGISTER,
        }
    }
}

impl Ir {
    const FIRST_REGISTER: usize = 4;
    pub fn push_var(&mut self, mut var: Var) -> VarId {
        let id = VarId(self.vars.len());
        var.reg = self.n_regs;
        self.n_regs += 1;
        self.vars.push(var);
        id
    }
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    pub fn var_mut(&mut self, id: VarId) -> &mut Var {
        &mut self.vars[id.0]
    }
    pub fn reg(&self, id: VarId) -> Reg {
        self.var(id).reg()
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

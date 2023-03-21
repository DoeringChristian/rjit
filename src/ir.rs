use std::collections::HashSet;
use std::num::NonZeroU64;
use std::sync::Arc;

use cust::prelude::DeviceBuffer;

use crate::iterators::{DepIter, DepIterOp};

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Neg(VarId),
    Not(VarId),
    Sqrt(VarId),
    Abs(VarId),
    Add(VarId, VarId), // Add two variables
    Sub(VarId, VarId),
    Mul(VarId, VarId),
    Div(VarId, VarId),
    Mod(VarId, VarId),
    Mulhi(VarId, VarId),
    Fma(VarId, VarId, VarId),
    Min(VarId, VarId),
    Max(VarId, VarId),
    Cail(VarId),
    Floor(VarId),
    Round(VarId),
    Trunc(VarId),
    Eq(VarId, VarId),
    Neq(VarId, VarId),
    Lt(VarId, VarId),
    Le(VarId, VarId),
    Gt(VarId, VarId),
    Ge(VarId, VarId),
    Select(VarId, VarId, VarId),
    Popc(VarId),
    Clz(VarId),
    Ctz(VarId),
    And(VarId, VarId),
    Or(VarId, VarId),
    Xor(VarId, VarId),
    Shl(VarId, VarId),
    Shr(VarId, VarId),
    Rcp(VarId, VarId),
    Rsqrt(VarId, VarId),
    Sin(VarId, VarId),
    Cos(VarId, VarId),
    Exp2(VarId, VarId),
    Log2(VarId, VarId),
    Cast(VarId),
    Bitcast(VarId),
    Gather {
        from: VarId,
        idx: VarId,
        mask: VarId,
    },
    Scatter {
        from: VarId,
        to: VarId,
        idx: VarId,
        mask: VarId,
    },
    Idx,
    ConstF32(f32),         // Set a constant value
    ConstU32(u32),         // Set a constant value
    Load(ParamId),         // Load from buffer with pointer in params at offset
    LoadLiteral(ParamId),  // Load from params at offset
    Store(VarId, ParamId), // Store at buffer with pointer in params at offset
}

impl Op {
    pub fn deps(self) -> DepIterOp {
        DepIterOp::new(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarType {
    // Void,
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Ptr,
    F16,
    F32,
    F64,
}
impl VarType {
    // Returns the register prefix for this variable
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::Bool => "%p",
            Self::I8 => "%b",
            Self::U8 => "%b",
            Self::I16 => "%w",
            Self::U16 => "%w",
            Self::I32 => "%r",
            Self::U32 => "%r",
            Self::I64 => "%rd",
            Self::U64 => "%rd",
            Self::Ptr => "%rd",
            Self::F16 => "%h",
            Self::F32 => "%f",
            Self::F64 => "%d",
        }
    }
    // Retuns the cuda/ptx Representation for this type
    pub fn name_cuda(&self) -> &'static str {
        match self {
            Self::Bool => "pred",
            Self::I8 => "s8",
            Self::U8 => "u8",
            Self::I16 => "s16",
            Self::U16 => "u16",
            Self::I32 => "s32",
            Self::U32 => "u32",
            Self::I64 => "s64",
            Self::U64 => "u64",
            Self::Ptr => "u64",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
    pub fn name_cuda_bin(&self) -> &'static str {
        match self {
            Self::Bool => "pred",
            Self::I8 => "b8",
            Self::U8 => "b8",
            Self::I16 => "b16",
            Self::U16 => "b16",
            Self::I32 => "b32",
            Self::U32 => "b32",
            Self::I64 => "b64",
            Self::U64 => "b64",
            Self::Ptr => "b64",
            Self::F16 => "b16",
            Self::F32 => "b32",
            Self::F64 => "b64",
        }
    }
    // Returns the size/stride of this variable
    pub fn size(self) -> u64 {
        match self {
            Self::Bool => 1,
            Self::I8 => 1,
            Self::U8 => 1,
            Self::I16 => 2,
            Self::U16 => 2,
            Self::I32 => 4,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::U64 => 8,
            Self::Ptr => 8,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
    pub fn is_uint(self) -> bool {
        match self {
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => true,
            _ => false,
        }
    }
    pub fn is_single(self) -> bool {
        match self {
            Self::F32 => true,
            _ => false,
        }
    }
    pub fn is_double(self) -> bool {
        match self {
            Self::F64 => true,
            _ => false,
        }
    }
}

///
///
///
#[derive(Debug, Clone)]
pub struct Var {
    pub op: Op,      // Operation used to construct the variable
    pub ty: VarType, // Type of the variable
    pub reg: usize,  // Register Index of that variable
    pub buffer: Option<Arc<cust::memory::DeviceBuffer<u8>>>,
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct VarId(pub usize);

impl std::fmt::Display for VarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParamId(usize);

impl ParamId {
    pub fn offset(self) -> usize {
        self.0 * 8
    }
}

impl std::ops::Deref for ParamId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub struct Ir {
    vars: Vec<Var>,
    // pub params: Vec<u64>, // Params vec![size, &buffer0, &buffer1]
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
            // params: vec![0],
            // n_regs: Self::FIRST_REGISTER,
        }
    }
}

impl Ir {
    // const FIRST_REGISTER: usize = 4;
    pub fn push_var(&mut self, mut var: Var) -> VarId {
        let id = VarId(self.vars.len());
        // var.reg = self.n_regs;
        // self.n_regs += 1;
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
    pub fn deps(&self, schedule: &[VarId]) -> DepIter {
        DepIter {
            ir: self,
            stack: Vec::from(schedule),
            discovered: HashSet::new(),
        }
    }
    // pub fn push_param(&mut self, param: u64) -> ParamId {
    //     let id = ParamId(self.params.len());
    //     // self.params.push(param);
    //     id
    // }
    pub fn vars(&self) -> &[Var] {
        &self.vars
    }
    // pub fn set_size(&mut self, size: u64) {
    //     self.params[0] = size;
    // }
    // pub fn size(&self) -> usize {
    //     self.params[0] as _
    // }
}

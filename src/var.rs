use std::fmt::Debug;

use slotmap::DefaultKey;
use smallvec::SmallVec;

use crate::backend::Buffer;
///
/// TODO: better param enum
///
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParamType {
    #[default]
    None,
    Input,
    Output,
    // Literal,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Op {
    // Data,
    #[default]
    Nop,
    Literal,
    Data,
    Neg,
    Not,
    Sqrt,
    Abs,
    Add, // Add two variables
    Sub,
    Mul,
    Div,
    Mod,
    Mulhi,
    Fma,
    Min,
    Max,
    Cail,
    Floor,
    Round,
    Trunc,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    Select,
    Popc,
    Clz,
    Ctz,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Rcp,
    Rsqrt,
    Sin,
    Cos,
    Exp2,
    Log2,
    Cast,
    Bitcast,
    Gather, // Gather operation (gathering directly from buffer).
    Scatter,
    Idx,
    // ConstF32(f32), // Set a constant value
    // ConstU32(u32), // Set a constant value
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub enum VarType {
    // Void,
    #[default]
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
    pub fn size(&self) -> usize {
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
    pub fn is_uint(&self) -> bool {
        match self {
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => true,
            _ => false,
        }
    }
    pub fn is_float(&self) -> bool {
        *self == VarType::F16 || *self == VarType::F32 || *self == VarType::F64
    }
    pub fn is_single(&self) -> bool {
        *self == Self::F32
    }
    pub fn is_double(&self) -> bool {
        *self == Self::F64
    }
    pub fn is_bool(&self) -> bool {
        *self == Self::Bool
    }
}

///
///
///
#[derive(Default)]
pub struct Var {
    pub op: Op, // Operation used to construct the variable
    pub deps: SmallVec<[VarId; 4]>,
    pub ty: VarType,                     // Type of the variable
    pub buffer: Option<Box<dyn Buffer>>, // Optional buffer
    pub size: usize,                     // number of elements
    pub param_ty: ParamType,             // Parameter type
    pub rc: usize,
    pub literal: u64,
    pub stop_traversal: bool, // Tells the scheduling routine to stop traversing at this variable even
                              // though it has dependencies.
}
impl Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Var")
            .field("op", &self.op)
            .field("deps", &self.deps)
            .field("ty", &self.ty)
            // .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("param_ty", &self.param_ty)
            .field("rc", &self.rc)
            .finish()
    }
}

impl Var {
    pub fn is_literal(&self) -> bool {
        self.op == Op::Literal
    }
    pub fn is_data(&self) -> bool {
        self.op == Op::Data
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct VarId(pub DefaultKey);

#[derive(Debug)]
pub struct VarInfo {
    pub ty: VarType,
    pub size: usize,
}

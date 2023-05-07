use std::fmt::Debug;
use std::sync::Arc;

use slotmap::DefaultKey;
use smallvec::SmallVec;

use crate::backend::{Accel, Buffer, Texture};
///
/// TODO: better param enum
///
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ParamType {
    #[default]
    None,
    Input,
    Output,
    // Literal,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    #[default]
    None,
    Add,
    Mul,
    Min,
    Max,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Op {
    // Data,
    #[default]
    Nop,
    Literal,
    Data, // TODO: maybe remove
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
    Ceil,
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
    Scatter {
        op: ReduceOp,
    },
    TexLookup {
        dim: u8,
    },
    Extract {
        offset: usize,
    },
    TexUpload,
    TraceRay {
        payload_count: usize,
    },
    Idx,
}

// TODO: Vector types
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub enum VarType {
    #[default]
    Void,
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    // Ptr,
    F16,
    F32,
    F64,
}
impl VarType {
    // Returns the register prefix for this variable
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::Void => "%u",
            Self::Bool => "%p",
            Self::I8 => "%b",
            Self::U8 => "%b",
            Self::I16 => "%w",
            Self::U16 => "%w",
            Self::I32 => "%r",
            Self::U32 => "%r",
            Self::I64 => "%rd",
            Self::U64 => "%rd",
            // Self::Ptr => "%rd",
            Self::F16 => "%h",
            Self::F32 => "%f",
            Self::F64 => "%d",
        }
    }
    // Retuns the cuda/ptx Representation for this type
    pub fn name_cuda(&self) -> &'static str {
        match self {
            Self::Void => "???",
            Self::Bool => "pred",
            Self::I8 => "s8",
            Self::U8 => "u8",
            Self::I16 => "s16",
            Self::U16 => "u16",
            Self::I32 => "s32",
            Self::U32 => "u32",
            Self::I64 => "s64",
            Self::U64 => "u64",
            // Self::Ptr => "u64",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
    pub fn name_cuda_bin(&self) -> &'static str {
        match self {
            Self::Void => "???",
            Self::Bool => "pred",
            Self::I8 => "b8",
            Self::U8 => "b8",
            Self::I16 => "b16",
            Self::U16 => "b16",
            Self::I32 => "b32",
            Self::U32 => "b32",
            Self::I64 => "b64",
            Self::U64 => "b64",
            // Self::Ptr => "b64",
            Self::F16 => "b16",
            Self::F32 => "b32",
            Self::F64 => "b64",
        }
    }
    // Returns the size/stride of this variable
    pub fn size(&self) -> usize {
        match self {
            Self::Void => 0,
            Self::Bool => 1,
            Self::I8 => 1,
            Self::U8 => 1,
            Self::I16 => 2,
            Self::U16 => 2,
            Self::I32 => 4,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::U64 => 8,
            // Self::Ptr => 8,
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
#[derive(Default, Debug)]
pub struct Var {
    pub op: Op, // Operation used to construct the variable
    pub deps: SmallVec<[VarId; 4]>,
    // Variable performing the latest write operation.
    // (used when building a scatter dependency chain).
    pub last_write: Option<VarId>,
    pub ty: VarType, // Type of the variable
    // pub buffer: Option<Arc<dyn Buffer>>, // Optional buffer
    // TODO: unify buffer and texture to
    // single enum
    // pub texture: Option<Arc<dyn Texture>>,
    pub size: usize, // number of elements
    pub rc: usize,
    pub data: Data,
    // pub literal: u64,
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

#[derive(Debug, Default)]
pub enum Data {
    #[default]
    None,
    Literal(u64),
    Buffer(Arc<dyn Buffer>),
    Texture(Arc<dyn Texture>),
    Accel(Arc<dyn Accel>),
}
impl Data {
    pub fn is_none(&self) -> bool {
        match self {
            Self::None => true,
            _ => false,
        }
    }
    pub fn is_literal(&self) -> bool {
        match self {
            Self::Literal(_) => true,
            _ => false,
        }
    }
    pub fn is_buffer(&self) -> bool {
        match self {
            Self::Buffer(_) => true,
            _ => false,
        }
    }
    pub fn is_texture(&self) -> bool {
        match self {
            Self::Texture(_) => true,
            _ => false,
        }
    }
    pub fn is_accel(&self) -> bool {
        match self {
            Self::Accel(_) => true,
            _ => false,
        }
    }
    pub fn is_storage(&self) -> bool {
        self.is_buffer() || self.is_texture()
    }
    pub fn literal(&self) -> Option<u64> {
        match self {
            Self::Literal(lit) => Some(*lit),
            _ => None,
        }
    }
    pub fn buffer(&self) -> Option<&Arc<dyn Buffer>> {
        match self {
            Self::Buffer(buf) => Some(buf),
            _ => None,
        }
    }
    pub fn texture(&self) -> Option<&Arc<dyn Texture>> {
        match self {
            Self::Texture(tex) => Some(tex),
            _ => None,
        }
    }
    pub fn accel(&self) -> Option<&Arc<dyn Accel>> {
        match self {
            Self::Accel(accel) => Some(accel),
            _ => None,
        }
    }
}

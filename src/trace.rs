use parking_lot::{MappedRwLockReadGuard, Mutex, RwLock, RwLockReadGuard};
use std::fmt::Debug;
use std::sync::Arc;

use bytemuck::cast_slice;
use bytemuck::checked::cast;
use slotmap::{DefaultKey, SlotMap};
use smallvec::{smallvec, SmallVec};

use crate::backend::{Backend, Buffer};
use crate::jit::Jit;

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

pub struct Ir {
    vars: SlotMap<DefaultKey, Var>,
    pub scheduled: Vec<VarId>,
    // backend: Arc<dyn Backend>,
}
impl Debug for Ir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ir")
            .field("vars", &self.vars)
            .field("scheduled", &self.scheduled)
            // .field("backend", &self.backend)
            .finish()
    }
}

#[derive(Debug)]
struct VarInfo {
    ty: VarType,
    size: usize,
}

impl Ir {
    pub fn new() -> Self {
        Self {
            vars: SlotMap::default(),
            scheduled: Vec::default(),
        }
    }
    pub fn push_var(&mut self, mut var: Var) -> VarId {
        for dep in var.deps.iter() {
            self.inc_rc(*dep);
        }
        var.rc = 1;
        VarId(self.vars.insert(var))
    }
    pub fn inc_rc(&mut self, id: VarId) {
        self.var_mut(id).rc += 1;
    }
    pub fn dec_rc(&mut self, id: VarId) {
        let var = self.var_mut(id);
        var.rc -= 1;
        if var.rc == 0 {
            for dep in var.deps.clone() {
                self.dec_rc(dep);
            }
            self.vars.remove(id.0).unwrap();
        }
    }
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    pub fn var_mut(&mut self, id: VarId) -> &mut Var {
        &mut self.vars[id.0]
    }
    fn var_info(&self, ids: &[VarId]) -> VarInfo {
        let ty = self.var(*ids.first().unwrap()).ty.clone(); // TODO: Fix (first non void)

        let size = ids
            .iter()
            .map(|id| &self.var(*id).size)
            .reduce(|s0, s1| s0.max(s1))
            .unwrap()
            .clone();
        VarInfo { ty, size }
    }
}

///
/// Wrapper arround an Intermediate Representation (Ir) for implementing automatic Refcounting.
///
#[derive(Debug)]
pub struct Trace {
    pub ir: Arc<RwLock<Ir>>,
    backend: Arc<dyn Backend>,
    jit: Mutex<Jit>,
}
///
/// Helper functions:
///
impl Trace {
    pub fn new(backend: &Arc<dyn Backend>) -> Self {
        Self {
            jit: Mutex::new(Jit::new(backend)),
            ir: Arc::new(RwLock::new(Ir::new())),
            backend: backend.clone(),
        }
    }
    fn var(&self, r: &Ref) -> MappedRwLockReadGuard<Var> {
        assert!(Arc::ptr_eq(&self.ir, &r.ir));
        RwLockReadGuard::map(self.ir.read(), |d| d.var(r.id))
    }
    fn push_var(&self, var: Var) -> Ref {
        let id = self.ir.write().push_var(var);
        Ref {
            id,
            ir: self.ir.clone(),
        }
    }
    fn push_var_intermediate(&self, op: Op, deps: &[Ref], ty: VarType, size: usize) -> Ref {
        for dep in deps {
            assert!(Arc::ptr_eq(&self.ir, &dep.ir));
        }
        self.push_var(Var {
            op,
            deps: deps.iter().map(|r| r.id).collect(),
            ty,
            param_ty: ParamType::None,
            buffer: None,
            size,
            rc: 0,
            literal: 0,
            stop_traversal: false,
        })
    }
}
///
/// Constant initializers:
///
impl Trace {
    pub fn const_f32(&self, val: f32) -> Ref {
        self.push_var(Var {
            op: Op::Literal,
            deps: smallvec![],
            ty: VarType::F32,
            literal: cast::<_, u32>(val) as _,
            ..Default::default()
        })
    }
    pub fn const_u32(&self, val: u32) -> Ref {
        self.push_var(Var {
            op: Op::Literal,
            deps: smallvec![],
            ty: VarType::U32,
            literal: cast::<_, u32>(val) as _,
            ..Default::default()
        })
    }
    pub fn const_bool(&self, val: bool) -> Ref {
        self.push_var(Var {
            op: Op::Literal,
            deps: smallvec![],
            ty: VarType::Bool,
            literal: cast::<_, u8>(val) as _,
            ..Default::default()
        })
    }
}
///
/// Buffer initializers:
///
impl Trace {
    pub fn buffer_f32(&self, slice: &[f32]) -> Ref {
        self.push_var(Var {
            param_ty: ParamType::Input,
            buffer: Some(self.backend.buffer_from_slice(cast_slice(slice))),
            size: slice.len(),
            ty: VarType::F32,
            op: Op::Data,
            ..Default::default()
        })
    }
    pub fn buffer_u32(&self, slice: &[u32]) -> Ref {
        self.push_var(Var {
            param_ty: ParamType::Input,
            buffer: Some(self.backend.buffer_from_slice(cast_slice(slice))),
            size: slice.len(),
            ty: VarType::U32,
            op: Op::Data,
            ..Default::default()
        })
    }
}
///
/// To host functions.
///
impl Trace {
    pub fn to_vec_f32(&self, r: &Ref) -> Vec<f32> {
        let var = self.var(&r);
        assert_eq!(var.ty, VarType::F32);
        let v = var.buffer.as_ref().unwrap().as_vec();
        Vec::from(cast_slice(&v))
    }
    pub fn to_vec_u32(&self, r: &Ref) -> Vec<u32> {
        let var = self.var(&r);
        assert_eq!(var.ty, VarType::U32);
        let v = var.buffer.as_ref().unwrap().as_vec();
        Vec::from(cast_slice(&v))
    }
}
///
/// Unary operations:
///
impl Trace {
    pub fn cast(&self, src: Ref, ty: VarType) -> Ref {
        let v = self.var(&src);
        self.push_var(Var {
            op: Op::Cast,
            deps: smallvec![src.id],
            ty,
            size: v.size,
            ..Default::default()
        })
    }
}
///
/// Evaluation related functions:
///
impl Trace {
    pub fn kernel_debug(&self) -> String {
        self.jit.lock().kernel_debug()
    }
    pub fn schedule(&self, refs: &[Ref]) {
        let mut ir = self.ir.write();
        for r in refs {
            ir.inc_rc(r.id);
            ir.scheduled.push(r.id);
        }
    }
    pub fn eval(&self) {
        self.jit.lock().eval(&mut self.ir.write());
    }
}
///
/// Binary Operations:
///
impl Trace {
    pub fn add(&self, lhs: Ref, rhs: Ref) -> Ref {
        let info = self.ir.read().var_info(&[lhs.id, rhs.id]);
        self.push_var_intermediate(Op::Add, &[lhs, rhs], info.ty, info.size)
    }
    pub fn mul(&self, lhs: Ref, rhs: Ref) -> Ref {
        let info = self.ir.read().var_info(&[lhs.id, rhs.id]);
        self.push_var_intermediate(Op::Mul, &[lhs, rhs], info.ty, info.size)
    }
    pub fn and(&self, lhs: Ref, rhs: Ref) -> Ref {
        let info = self.ir.read().var_info(&[lhs.id, rhs.id]);

        let ty_lhs = self.ir.read().var(lhs.id).ty.clone();
        let ty_rhs = self.ir.read().var(rhs.id).ty.clone();

        if info.size > 0 && ty_lhs != ty_rhs && !ty_rhs.is_bool() {
            panic!("Invalid operands!");
        }

        self.push_var_intermediate(Op::And, &[lhs, rhs], info.ty, info.size)
    }
}
///
/// Special operations such as Gather, Scatter etc.
///
impl Trace {
    pub fn index(&self, size: usize) -> Ref {
        self.push_var(Var {
            op: Op::Idx,
            deps: smallvec![],
            ty: VarType::U32,
            size,
            ..Default::default()
        })
    }
    pub fn pointer_to(&self, src: &Ref) -> Option<Ref> {
        // TODO: Eval var if needed.
        let ptr = self.var(&src).buffer.as_ref().map(|b| b.as_ptr());
        if let Some(ptr) = ptr {
            Some(self.push_var(Var {
                op: Op::Literal,
                param_ty: ParamType::Input,
                deps: smallvec![src.id],
                ty: VarType::Ptr,
                literal: ptr,
                size: 1,
                stop_traversal: true,
                ..Default::default()
            }))
        } else {
            None
        }
    }
    /// Reindex a variable with a new index and size.
    /// (Normally size is the size of the index)
    ///
    /// For now we construct a separate set of variables in the Ir.
    /// However, it should sometimes be possible to reuse the old ones.
    ///
    fn reindex(&self, r: &Ref, new_idx: &Ref, size: usize) -> Option<Ref> {
        let var = self.var(&r);

        if var.is_data() {
            return None;
        }

        let mut deps = smallvec![];
        if !var.is_literal() {
            for dep in var.deps.clone() {
                let dep = Ref {
                    id: dep,
                    ir: self.ir.clone(),
                };
                if let Some(dep) = self.reindex(&dep, new_idx, size) {
                    deps.push(dep.id);
                } else {
                    return None;
                }
            }
        }

        if var.op == Op::Idx {
            drop(var);
            // self.inc_rc(new_idx);
            return Some(new_idx.clone());
        } else {
            return Some(self.push_var(Var {
                op: var.op,
                deps,
                ty: var.ty.clone(),
                buffer: None,
                size,
                param_ty: var.param_ty,
                rc: 0,
                literal: var.literal,
                stop_traversal: var.stop_traversal,
            }));
        }
    }
    ///
    /// For gather operations there are three ways to resolve them:
    ///
    /// If the source is a Pointer (i.e. VarType::Ptr) we can simply gather from that
    /// pointer.
    ///
    /// If the source is trivial (i.e. there are no dependencies on Input variablse
    /// ParamType::Input) we can
    /// reindex the variable.
    ///
    /// Finally, if the variable depends on Inputs we need to launch multiple Kernels (this
    /// is not yet implemented).
    ///
    pub fn gather(&self, src: Ref, index: Ref, mask: Option<Ref>) -> Ref {
        let mask = mask.unwrap_or(self.const_bool(true));

        let res = self.pointer_to(&src);

        // let var = self.var(&src);
        let ty = self.var(&src).ty.clone();

        if let Some(src) = res {
            let size = self.var(&index).size;
            let ret = self.push_var(Var {
                op: Op::Gather,
                deps: smallvec![src.id, index.id, mask.id],
                ty,
                size,
                ..Default::default()
            });
            return ret;
        }

        let size = self.var(&index).size;

        let res = self.reindex(&src, &index, size);

        if let Some(res) = res {
            let res = self.and(res, mask); // TODO: masking
            return res;
        }

        unimplemented!();
    }
}

///
/// Reference to a variable.
///
#[derive(Debug)]
pub struct Ref {
    id: VarId,
    ir: Arc<RwLock<Ir>>,
}

impl Clone for Ref {
    fn clone(&self) -> Self {
        self.ir.write().inc_rc(self.id);
        Self {
            id: self.id.clone(),
            ir: self.ir.clone(),
        }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        self.ir.write().dec_rc(self.id);
    }
}

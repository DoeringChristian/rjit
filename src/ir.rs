use std::borrow::Borrow;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::Arc;

use bytemuck::cast_slice;
use slotmap::{DefaultKey, SlotMap};
use smallvec::{smallvec, SmallVec};

use crate::backend::{Backend, Buffer};
use crate::jit;

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

thread_local! {pub static IR: RefCell<Ir> = RefCell::new(Ir::default())}
thread_local! {pub static BACKEND: RefCell<Option<Arc<dyn Backend>>> = RefCell::new(None)}

pub fn set_backend(backend: &Arc<dyn Backend>) {
    BACKEND.with(|b| {
        if b.borrow_mut().is_some() {
            panic!("Cannot change Backend!");
        } else {
            *b.borrow_mut() = Some(backend.clone());
        }
    })
}

#[derive(Default)]
pub struct Ir {
    vars: SlotMap<DefaultKey, Var>,
    // pub scheduled: Vec<VarId>,
}
impl Debug for Ir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ir")
            .field("vars", &self.vars)
            // .field("scheduled", &self.scheduled)
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
    pub fn inc_rc(&mut self, id: VarId) {
        self.vars[id.0].rc += 1;
    }
    pub fn dec_rc(&mut self, id: VarId) {
        self.vars[id.0].rc -= 1;
        let rc = self.vars[id.0].rc;
        if rc == 0 {
            for dep in self.vars[id.0].deps.clone() {
                self.dec_rc(dep);
            }
            self.vars.remove(id.0);
        }
    }
}

///
/// I'm not quite sure about safety, however I think this might work.
///
pub fn var(r: &Ref) -> std::cell::Ref<Var> {
    let ir = unsafe { IR.with(|ir| (ir) as *const RefCell<Ir>).as_ref().unwrap() };
    std::cell::Ref::map(ir.borrow(), |ir| &ir.vars[r.id().0])
}
pub fn var_mut(r: &Ref) -> std::cell::RefMut<Var> {
    let ir = unsafe { IR.with(|ir| (ir) as *const RefCell<Ir>).as_ref().unwrap() };
    std::cell::RefMut::map(ir.borrow_mut(), |ir| &mut ir.vars[r.id().0])
}

pub fn push_var(mut var: Var) -> Ref {
    var.rc = 1;
    for dep in var.deps.clone() {
        IR.with(|ir| ir.borrow_mut().inc_rc(dep));
    }
    IR.with(|ir| Ref(VarId(ir.borrow_mut().vars.insert(var))))
}
fn push_var_intermediate(op: Op, deps: &[&Ref], ty: VarType, size: usize) -> Ref {
    let deps = deps.iter().map(|r| r.0).collect();
    let ret = push_var(Var {
        op,
        deps,
        ty,
        param_ty: ParamType::None,
        buffer: None,
        size,
        rc: 0,
        literal: 0,
        stop_traversal: false,
    });
    ret
}
fn var_info(ids: &[&Ref]) -> VarInfo {
    let ty = var(*ids.first().unwrap()).ty.clone(); // TODO: Fix (first non void)

    let size = ids
        .iter()
        .map(|id| &var(*id).size)
        .reduce(|s0, s1| s0.max(s1))
        .unwrap()
        .clone();
    VarInfo { ty, size }
}

pub fn const_f32(val: f32) -> Ref {
    push_var(Var {
        op: Op::Literal,
        deps: smallvec![],
        ty: VarType::F32,
        literal: bytemuck::cast::<_, u32>(val) as _,
        ..Default::default()
    })
}
pub fn const_u32(val: u32) -> Ref {
    push_var(Var {
        op: Op::Literal,
        deps: smallvec![],
        ty: VarType::U32,
        literal: bytemuck::cast::<_, u32>(val) as _,
        ..Default::default()
    })
}
pub fn const_bool(val: bool) -> Ref {
    push_var(Var {
        op: Op::Literal,
        deps: smallvec![],
        ty: VarType::Bool,
        literal: bytemuck::cast::<_, u8>(val) as _,
        ..Default::default()
    })
}

// Buffer initializers:
pub fn buffer_f32(slice: &[f32]) -> Ref {
    BACKEND.with(|backend| {
        push_var(Var {
            param_ty: ParamType::Input,
            buffer: Some(
                backend
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .buffer_from_slice(cast_slice(slice)),
            ),
            size: slice.len(),
            ty: VarType::F32,
            op: Op::Data,
            ..Default::default()
        })
    })
}
pub fn buffer_u32(slice: &[u32]) -> Ref {
    BACKEND.with(|backend| {
        push_var(Var {
            param_ty: ParamType::Input,
            buffer: Some(
                backend
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .buffer_from_slice(cast_slice(slice)),
            ),
            size: slice.len(),
            ty: VarType::U32,
            op: Op::Data,
            ..Default::default()
        })
    })
}
// To Host functions:
pub fn to_vec_f32(r: &Ref) -> Vec<f32> {
    let var = var(r);
    assert_eq!(var.ty, VarType::F32);
    let v = var.buffer.as_ref().unwrap().as_vec();
    Vec::from(cast_slice(&v))
}
pub fn to_vec_u32(r: &Ref) -> Vec<u32> {
    let var = var(r);
    assert_eq!(var.ty, VarType::U32);
    let v = var.buffer.as_ref().unwrap().as_vec();
    Vec::from(cast_slice(&v))
}
// Unarry operations:
pub fn cast(src: &Ref, ty: VarType) -> Ref {
    let v = var(src);
    push_var(Var {
        op: Op::Cast,
        deps: smallvec![src.id()],
        ty,
        size: v.size,
        ..Default::default()
    })
}
// Binarry operations:
pub fn add(lhs: &Ref, rhs: &Ref) -> Ref {
    let info = var_info(&[&lhs, &rhs]);
    push_var_intermediate(Op::Add, &[&lhs, &rhs], info.ty, info.size)
}
pub fn mul(lhs: &Ref, rhs: &Ref) -> Ref {
    let info = var_info(&[&lhs, &rhs]);
    push_var_intermediate(Op::Mul, &[&lhs, &rhs], info.ty, info.size)
}
pub fn and(lhs: &Ref, rhs: &Ref) -> Ref {
    let info = var_info(&[&lhs, &rhs]);
    IR.with(|ir| {
        let ir = ir.borrow();

        let ty_lhs = var(&lhs).ty.clone();
        let ty_rhs = var(&rhs).ty.clone();

        if info.size > 0 && ty_lhs != ty_rhs && !ty_rhs.is_bool() {
            panic!("Invalid operands!");
        }
    });
    let ret = push_var_intermediate(Op::And, &[&lhs, &rhs], info.ty, info.size);
    dbg!();
    ret
}
// Special operations:
pub fn index(size: usize) -> Ref {
    push_var(Var {
        op: Op::Idx,
        deps: smallvec![],
        ty: VarType::U32,
        size,
        ..Default::default()
    })
}
pub fn pointer_to(src: &Ref) -> Option<Ref> {
    let ptr = var(&src).buffer.as_ref().map(|b| b.as_ptr());
    // TODO: Eval var if needed.
    if let Some(ptr) = ptr {
        Some(push_var(Var {
            op: Op::Literal,
            param_ty: ParamType::Input,
            deps: smallvec![src.id()],
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
fn reindex(r: &Ref, new_idx: &Ref, size: usize) -> Option<Ref> {
    let var = var(&r);
    let is_literal = var.is_literal();
    let is_data = var.is_data();

    if is_data {
        return None;
    }

    dbg!();
    let mut deps = smallvec![];
    if !is_literal {
        for dep in var.deps.clone() {
            let dep = Ref::borrow(dep);
            if let Some(dep) = reindex(&dep, new_idx, size) {
                deps.push(dep.id());
            } else {
                return None;
            }
        }
    }

    let op = var.op;
    let ty = var.ty.clone();
    let param_ty = var.param_ty;
    let literal = var.literal;
    let stop_traversal = var.stop_traversal;

    drop(var);

    if op == Op::Idx {
        // self.inc_rc(new_idx);
        return Some(new_idx.clone());
    } else {
        return Some(push_var(Var {
            op,
            deps,
            ty,
            buffer: None,
            size,
            param_ty,
            rc: 0,
            literal,
            stop_traversal,
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
pub fn gather(src: &Ref, index: &Ref, mask: Option<&Ref>) -> Ref {
    let size = var(&index).size;
    let ty = var(&index).ty;

    let mask: Ref = mask.map(|m| m.clone()).unwrap_or(const_bool(true));

    let res = pointer_to(&src);

    if let Some(src) = res {
        let ret = push_var(Var {
            op: Op::Gather,
            deps: smallvec![src.id(), index.id(), mask.id()],
            ty,
            size,
            ..Default::default()
        });
        return ret;
    }

    let res = reindex(&src, &index, size);

    if let Some(res) = res {
        let res = and(&res, &mask); // TODO: masking
        return res;
    }

    jit::schedule(&[src]);
    jit::eval();

    let res = pointer_to(&src);

    if let Some(src) = res {
        let ret = push_var(Var {
            op: Op::Gather,
            deps: smallvec![src.id(), index.id(), mask.id()],
            ty,
            size,
            ..Default::default()
        });
        return ret;
    }

    unimplemented!();
}

///
/// Reference to a variable.
///
#[derive(Debug)]
pub struct Ref(VarId);

impl Ref {
    pub fn id(&self) -> VarId {
        self.0
    }
    pub fn borrow(id: VarId) -> Self {
        IR.with(|ir| ir.borrow_mut().inc_rc(id));
        Self(id)
    }
    pub fn is_data(&self) -> bool {
        let var = var(self);
        if var.buffer.is_some() {
            assert_eq!(var.op, Op::Data);
            true
        } else {
            assert_ne!(var.op, Op::Data);
            false
        }
    }
}

impl Clone for Ref {
    fn clone(&self) -> Self {
        IR.with(|ir| ir.borrow_mut().inc_rc(self.0));
        Self(self.0)
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        IR.with(|ir| ir.borrow_mut().dec_rc(self.0));
    }
}

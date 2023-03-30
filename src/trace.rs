use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use bytemuck::cast_slice;
use once_cell::sync::Lazy;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use slotmap::{DefaultKey, SlotMap};
use smallvec::smallvec;

use crate::backend::cuda::CUDABackend;
use crate::backend::Backend;
use crate::jit::{self, Jit};
use crate::var::{Op, Var, VarId, VarInfo, VarType};

// We have one global Intermediate Representation that tracks all operations.
// However, Other Intermediate representations can also be constructed.
pub static IR: Lazy<Trace> = Lazy::new(|| Trace::default());

///
/// A wrapper arrund an Intermediate Representation.
///
#[derive(Clone, Debug, Default)]
pub struct Trace(Arc<Mutex<Internal>>);
impl Deref for Trace {
    type Target = Arc<Mutex<Internal>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Trace {
    fn push_var(&self, mut v: Var) -> VarRef {
        for dep in v.deps.iter() {
            self.lock().inc_rc(*dep);
        }
        v.rc = 1;
        let id = VarId(self.lock().vars.insert(v));
        VarRef::steal(&self, id)
    }
    fn var_info(&self, refs: &[&VarRef]) -> VarInfo {
        let ty = refs.first().unwrap().var().ty.clone(); // TODO: Fix (first non void)

        let size = refs
            .iter()
            .map(|id| id.var().size)
            .reduce(|s0, s1| s0.max(s1))
            .unwrap()
            .clone();
        VarInfo { ty, size }
    }
    fn push_var_op(&self, op: Op, deps: &[&VarRef], ty: VarType, size: usize) -> VarRef {
        let deps = deps
            .iter()
            .map(|r| {
                assert!(Arc::ptr_eq(&self.0, &r.ir));
                r.id()
            })
            .collect();
        let ret = self.push_var(Var {
            op,
            deps,
            ty,
            // param_ty: ParamType::None,
            buffer: None,
            size,
            rc: 0,
            literal: 0,
        });
        ret
    }
    pub fn set_backend(&self, backend: impl AsRef<str>) {
        if self.lock().backend.is_some() {
            return;
        }
        let backend = backend.as_ref();
        if backend == "cuda" {
            self.lock().backend = Some(Box::new(CUDABackend::new()));
        }
    }
    pub fn schedule(&self, refs: &[&VarRef]) {
        for r in refs {
            assert!(Arc::ptr_eq(&r.ir, &self.0));
            self.lock().schedule(&[r.id()])
        }
    }
}
impl Trace {
    // Constatns:
    pub fn const_f32(&self, val: f32) -> VarRef {
        self.push_var(Var {
            op: Op::Literal,
            deps: smallvec![],
            ty: VarType::F32,
            literal: bytemuck::cast::<_, u32>(val) as _,
            ..Default::default()
        })
    }
    pub fn const_u32(&self, val: u32) -> VarRef {
        self.push_var(Var {
            op: Op::Literal,
            deps: smallvec![],
            ty: VarType::U32,
            literal: bytemuck::cast::<_, u32>(val) as _,
            ..Default::default()
        })
    }
    pub fn const_bool(&self, val: bool) -> VarRef {
        self.push_var(Var {
            op: Op::Literal,
            deps: smallvec![],
            ty: VarType::Bool,
            literal: bytemuck::cast::<_, u8>(val) as _,
            ..Default::default()
        })
    }
    // Buffer initializers:
    pub fn buffer_f32(&self, slice: &[f32]) -> VarRef {
        let buffer = Some(
            self.lock()
                .backend
                .as_ref()
                .unwrap()
                .buffer_from_slice(cast_slice(slice)),
        );
        self.push_var(Var {
            // param_ty: ParamType::Input,
            buffer,
            size: slice.len(),
            ty: VarType::F32,
            op: Op::Data,
            ..Default::default()
        })
    }
    pub fn buffer_u32(&self, slice: &[u32]) -> VarRef {
        let buffer = Some(
            self.lock()
                .backend
                .as_ref()
                .unwrap()
                .buffer_from_slice(cast_slice(slice)),
        );
        self.push_var(Var {
            // param_ty: ParamType::Input,
            buffer,
            size: slice.len(),
            ty: VarType::U32,
            op: Op::Data,
            ..Default::default()
        })
    }
    // Special operations:
    pub fn index(&self, size: usize) -> VarRef {
        let v = self.push_var(Var {
            op: Op::Idx,
            deps: smallvec![],
            ty: VarType::U32,
            size,
            ..Default::default()
        });
        v
    }
}

#[derive(Default, Debug)]
pub struct Internal {
    vars: SlotMap<DefaultKey, Var>,
    pub backend: Option<Box<dyn Backend>>,
    pub functions: Vec<VarId>,
    pub scheduled: Vec<VarId>,
}

impl Internal {
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    pub fn var_mut(&mut self, id: VarId) -> &mut Var {
        &mut self.vars[id.0]
    }
    pub fn get_var(&mut self, id: VarId) -> Option<&Var> {
        self.vars.get(id.0)
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
            self.vars.remove(id.0);
        }
    }
    pub fn schedule(&mut self, ids: &[VarId]) {
        for id in ids {
            self.inc_rc(*id);
            self.scheduled.push(*id);
        }
    }
}

///
/// Reference to a variable.
///
#[derive(Debug)]
pub struct VarRef {
    id: VarId,
    ir: Trace,
}

impl VarRef {
    pub fn id(&self) -> VarId {
        self.id
    }
    pub fn borrow(trace: &Trace, id: VarId) -> Self {
        trace.lock().inc_rc(id);

        Self {
            id,
            ir: trace.clone(),
        }
    }
    pub fn steal(trace: &Trace, id: VarId) -> Self {
        Self {
            id,
            ir: trace.clone(),
        }
    }
    pub fn var(&self) -> MappedMutexGuard<Var> {
        MutexGuard::map(self.ir.lock(), |ir| &mut ir.vars[self.id().0])
    }
    pub fn is_data(&self) -> bool {
        let var = self.var();
        if var.buffer.is_some() {
            assert_eq!(var.op, Op::Data);
            true
        } else {
            assert_ne!(var.op, Op::Data);
            false
        }
    }
    pub fn schedule(&self) {
        self.ir.lock().schedule(&[self.id()])
    }
    // To Host functions:
    pub fn to_vec_f32(&self) -> Vec<f32> {
        let var = self.var();
        assert_eq!(var.ty, VarType::F32);
        let v = var.buffer.as_ref().unwrap().as_vec();
        Vec::from(cast_slice(&v))
    }
    pub fn to_vec_u32(&self) -> Vec<u32> {
        let var = self.var();
        assert_eq!(var.ty, VarType::U32);
        let v = var.buffer.as_ref().unwrap().as_vec();
        Vec::from(cast_slice(&v))
    }
    // Unarry operations:
    pub fn cast(&self, ty: VarType) -> VarRef {
        let v = self.var();
        self.ir.push_var(Var {
            op: Op::Cast,
            deps: smallvec![self.id()],
            ty,
            size: v.size,
            ..Default::default()
        })
    }
    // Binarry operations:
    pub fn add(&self, rhs: &VarRef) -> VarRef {
        let info = self.ir.var_info(&[self, &rhs]);
        self.ir
            .push_var_op(Op::Add, &[self, &rhs], info.ty, info.size)
    }
    pub fn mul(&self, rhs: &VarRef) -> VarRef {
        let info = self.ir.var_info(&[self, &rhs]);
        self.ir
            .push_var_op(Op::Mul, &[self, &rhs], info.ty, info.size)
    }
    pub fn and(&self, rhs: &VarRef) -> VarRef {
        assert!(Arc::ptr_eq(&self.ir, &rhs.ir));
        let info = self.ir.var_info(&[self, &rhs]);

        let ty_lhs = self.var().ty.clone();
        let ty_rhs = rhs.var().ty.clone();

        if info.size > 0 && ty_lhs != ty_rhs && !ty_rhs.is_bool() {
            panic!("Invalid operands!");
        }

        let ret = self
            .ir
            .push_var_op(Op::And, &[self, &rhs], info.ty, info.size);
        ret
    }
    // pub fn as_ptr(&self) -> Option<VarRef> {
    //     let ptr = self.var().buffer.as_ref().map(|b| b.as_ptr());
    //     // TODO: Eval var if needed.
    //     if let Some(ptr) = ptr {
    //         Some(self.ir.push_var(Var {
    //             op: Op::Literal,
    //             // param_ty: ParamType::Input,
    //             deps: smallvec![self.id()],
    //             ty: VarType::Ptr,
    //             literal: ptr,
    //             size: 1,
    //             stop_traversal: true,
    //             ..Default::default()
    //         }))
    //     } else {
    //         None
    //     }
    // }
    /// Reindex a variable with a new index and size.
    /// (Normally size is the size of the index)
    ///
    /// For now we construct a separate set of variables in the Ir.
    /// However, it should sometimes be possible to reuse the old ones.
    ///
    fn reindex(&self, new_idx: &VarRef, size: usize) -> Option<VarRef> {
        assert!(Arc::ptr_eq(&self.ir, &new_idx.ir));
        // let mut ir = self.ir.lock();

        let v = self.var();
        let is_literal = v.is_literal();
        let is_data = v.is_data();
        let v_deps = v.deps.clone();

        if is_data {
            return None;
        }

        drop(v);

        let mut deps = smallvec![];
        if !is_literal {
            for dep in v_deps {
                let dep = VarRef::borrow(&self.ir, dep);
                if let Some(dep) = dep.reindex(new_idx, size) {
                    deps.push(dep.id());
                } else {
                    return None;
                }
            }
        }

        let v = self.var();

        let op = v.op;
        let ty = v.ty.clone();
        // let param_ty = v.param_ty;
        let literal = v.literal;
        drop(v);

        if op == Op::Idx {
            // self.inc_rc(new_idx);
            return Some(new_idx.clone());
        } else {
            return Some(self.ir.push_var(Var {
                op,
                deps,
                ty,
                buffer: None,
                size,
                // param_ty,
                rc: 0,
                literal,
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
    pub fn gather(&self, index: &VarRef, mask: Option<&VarRef>) -> VarRef {
        assert!(Arc::ptr_eq(&self.ir, &index.ir));

        let size = index.var().size;
        let ty = self.var().ty.clone();

        let mask: VarRef = mask
            .map(|m| {
                assert!(Arc::ptr_eq(&self.ir, &m.ir));
                m.clone()
            })
            .unwrap_or(self.ir.const_bool(true));

        // let res = self.as_ptr();

        if self.var().buffer.is_some() {
            let ret = self.ir.push_var(Var {
                op: Op::Gather,
                deps: smallvec![self.id(), index.id(), mask.id()],
                ty,
                size,
                ..Default::default()
            });
            return ret;
        }

        let res = self.reindex(&index, size);

        if let Some(res) = res {
            let res = res.and(&mask);
            return res;
        }

        self.schedule(); // TODO: instead of evaluation, use dependency scheduling
        let mut jit = Jit::default();
        jit.eval(&mut self.ir.lock());

        if self.var().buffer.is_some() {
            let ret = self.ir.push_var(Var {
                op: Op::Gather,
                deps: smallvec![self.id(), index.id(), mask.id()],
                ty,
                size,
                ..Default::default()
            });
            return ret;
        }

        unimplemented!();
    }
}

impl Clone for VarRef {
    fn clone(&self) -> Self {
        self.ir.lock().inc_rc(self.id);
        Self {
            ir: self.ir.clone(),
            id: self.id,
        }
    }
}

impl Drop for VarRef {
    fn drop(&mut self) {
        self.ir.lock().dec_rc(self.id);
    }
}

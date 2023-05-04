use half::f16;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use std::cell::{RefCell, RefMut};
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

use bytemuck::cast_slice;
use slotmap::{DefaultKey, SlotMap};
use smallvec::smallvec;

use crate::backend::Backend;
pub use crate::var::{Data, Op, ReduceOp, Var, VarId, VarInfo, VarType};

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
        // Pus side effects of sources as extra dependencies
        for dep in v.deps.clone() {
            if let Some(se) = self.lock().var(dep).last_write {
                v.deps.push(se);
            }
        }
        // Inc rc for deps
        for dep in v.deps.iter() {
            self.lock().inc_rc(*dep);
        }
        v.rc = 1;
        let id = VarId(self.lock().vars.insert(v));
        VarRef::steal_from(&self, id)
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
        let mut deps: smallvec::SmallVec<[VarId; 4]> = deps
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
            size,
            ..Default::default()
        });
        ret
    }
    pub fn set_backend(&self, backend: impl AsRef<str>) {
        if self.lock().backend.is_some() {
            return;
        }
        let backend = backend.as_ref();
        if backend == "cuda" {
            self.lock().backend = Some(Box::new(crate::backend::cuda::Backend::new().unwrap()));
        } else if backend == "optix" {
            self.lock().backend = Some(Box::new(crate::backend::optix::Backend::new().unwrap()));
        }
    }
    pub fn backend(&self) -> MappedMutexGuard<dyn Backend> {
        MutexGuard::map(self.0.lock(), |ir| ir.backend.as_mut().unwrap().as_mut())
    }
    pub fn backend_as<T: Backend + 'static>(&self) -> MappedMutexGuard<T> {
        MutexGuard::map(self.0.lock(), |ir| {
            ir.backend
                .as_mut()
                .unwrap()
                .as_mut()
                .as_any_mut()
                .downcast_mut::<T>()
                .unwrap()
        })
    }
    pub fn schedule(&self, refs: &[&VarRef]) {
        for r in refs {
            assert!(Arc::ptr_eq(&r.ir, &self.0));
            self.lock().schedule(&[r.id()])
        }
    }
}
macro_rules! buffer {
    ($TY:ident, $ty:ident) => {
        paste::paste! {
            pub fn [<buffer_$ty>](&self, slice: &[$ty]) -> VarRef {
                let buffer = self.lock()
                        .backend
                        .as_ref()
                        .unwrap()
                        .buffer_from_slice(cast_slice(slice));
                self.push_var(Var {
                    data: Data::Buffer(buffer),
                    size: slice.len(),
                    ty: VarType::$TY,
                    op: Op::Data,
                    ..Default::default()
                })
            }
        }
    };
}

macro_rules! literal {
    ($TY:ident, $i:ident, $ty:ident) => {
        paste::paste! {
            pub fn [<literal_$ty>](&self, val: $ty) -> VarRef {
                self.push_var(Var {
                    op: Op::Literal,
                    deps: smallvec![],
                    ty: VarType::$TY,
                    data: Data::Literal(bytemuck::cast::<_, $i>(val) as _),
                    size: 1,
                    ..Default::default()
                })
            }
        }
    };
}

impl Trace {
    // Buffer initializers:
    buffer!(Bool, bool);
    buffer!(I8, i8);
    buffer!(U8, u8);
    buffer!(I16, i16);
    buffer!(U16, u16);
    buffer!(I32, i32);
    buffer!(U32, u32);
    buffer!(I64, i64);
    buffer!(U64, u64);
    buffer!(F16, f16);
    buffer!(F32, f32);
    buffer!(F64, f64);

    // Literal initializers:
    literal!(Bool, u8, bool);
    literal!(I8, u8, i8);
    literal!(U8, u8, u8);
    literal!(I16, u16, i16);
    literal!(U16, u16, u16);
    literal!(I32, u32, i32);
    literal!(U32, u32, u32);
    literal!(I64, u64, i64);
    literal!(U64, u64, u64);
    literal!(F16, u16, f16);
    literal!(F32, u32, f32);
    literal!(F64, u64, f64);

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
    pub fn texture(&self, shape: &[usize], n_channels: usize) -> VarRef {
        let size = shape.iter().cloned().reduce(|a, b| a * b).unwrap() * n_channels;
        let texture = self
            .0
            .lock()
            .backend
            .as_ref()
            .unwrap()
            .create_texture(shape, n_channels);
        self.push_var(Var {
            op: Op::Nop,
            ty: VarType::Void,
            size,
            data: Data::Texture(texture),
            ..Default::default()
        })
    }
    pub fn accel(&self, vertices: &VarRef, indices: &VarRef) -> VarRef {
        let vertices = vertices.var().data.buffer().unwrap().clone();
        let indices = indices.var().data.buffer().unwrap().clone();
        let accel = self
            .lock()
            .backend
            .as_ref()
            .unwrap()
            .create_accel(&vertices, &indices);
        self.push_var(Var {
            op: Op::Nop,
            ty: VarType::Void,
            size: 0,
            data: Data::Accel(accel),
            ..Default::default()
        })
    }
}

#[derive(Default, Debug)]
pub struct Internal {
    vars: SlotMap<DefaultKey, Var>,
    pub backend: Option<Box<dyn Backend>>,
    // pub functions: Vec<VarId>,
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
            let var = self.var_mut(id);
            if let Some(se) = var.last_write.clone() {
                self.dec_rc(se);
            }
            self.vars.remove(id.0);
        }
    }
    pub fn schedule(&mut self, ids: &[VarId]) {
        for id in ids {
            // Don't schedule data
            if self.var(*id).op == Op::Data {
                // TODO: maybe we only need to test if a buffer
                // exists
                continue;
            }
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
    fn borrow_from(trace: &Trace, id: VarId) -> Self {
        trace.lock().inc_rc(id);

        Self {
            id,
            ir: trace.clone(),
        }
    }
    fn steal_from(trace: &Trace, id: VarId) -> Self {
        Self {
            id,
            ir: trace.clone(),
        }
    }
    pub(crate) fn var(&self) -> MappedMutexGuard<Var> {
        MutexGuard::map(self.ir.lock(), |ir| &mut ir.vars[self.id().0])
        // MutexGuard::map(self.ir.lock(), |ir| &mut ir.vars[self.id().0])
    }
}
macro_rules! bop_arythmetic {
    ($Op:ident) => {
        paste::paste! {
            pub fn [<$Op:lower>](&self, rhs: &VarRef) -> VarRef {
                let info = self.ir.var_info(&[self, &rhs]);
                self.ir
                    .push_var_op(Op::$Op, &[self, &rhs], info.ty, info.size)
            }
        }
    };
}
macro_rules! to_host {
    ($Ty:ident) => {
        paste::paste! {
            pub fn [<to_host_$Ty:lower>](&self) -> Vec<[<$Ty:lower>]> {
                let var = self.var();
                assert_eq!(var.ty, VarType::$Ty);

                let mut dst = Vec::with_capacity(var.size);
                unsafe{dst.set_len(var.size)};

                var.data.buffer().unwrap().copy_to_host(bytemuck::cast_slice_mut(&mut dst));
                dst
            }
        }
    };
}

macro_rules! uop {
    ($Op:ident) => {
        paste::paste! {
            pub fn [<$Op:lower>](&self) -> Self {
                let info = self.ir.var_info(&[self]);
                self.ir.push_var_op(Op::$Op, &[self], info.ty, info.size)
            }
        }
    };
}

impl VarRef {
    pub fn ty(&self) -> VarType {
        self.var().ty.clone()
    }
    pub fn size(&self) -> usize {
        self.var().size
    }
    pub fn schedule(&self) {
        self.ir.lock().schedule(&[self.id()])
    }
    // To Host functions:
    // to_host!(Bool, bool);
    //

    // =========================================
    // Recursive expansion of the to_host! macro
    // =========================================

    pub fn to_host_bool(&self) -> Vec<bool> {
        let var = self.var();
        assert_eq!(var.ty, VarType::Bool);

        let mut dst = Vec::with_capacity(var.size);
        unsafe { dst.set_len(var.size) };
        unsafe {
            var.data
                .buffer()
                .unwrap()
                .copy_to_host(std::slice::from_raw_parts_mut(
                    dst.as_mut_ptr() as *mut _,
                    dst.len(),
                ));
        }
        dst
    }

    to_host!(I8);
    to_host!(U8);
    to_host!(I16);
    to_host!(U16);
    to_host!(I32);
    to_host!(U32);
    to_host!(I64);
    to_host!(U64);
    to_host!(F16);
    to_host!(F32);
    to_host!(F64);
    // Unarry operations:
    pub fn cast(&self, ty: &VarType) -> VarRef {
        self.ir
            .push_var_op(Op::Cast, &[self], ty.clone(), self.size())
    }
    pub fn bitcast(&self, ty: &VarType) -> VarRef {
        assert_eq!(self.var().ty.size(), ty.size());
        self.ir
            .push_var_op(Op::Bitcast, &[self], ty.clone(), self.size())
    }
    // Binarry operations:
    bop_arythmetic!(Add);
    bop_arythmetic!(Sub);
    bop_arythmetic!(Mul);
    bop_arythmetic!(Div);

    uop!(Rcp);
    uop!(Rsqrt);
    uop!(Sin);
    uop!(Cos);
    uop!(Exp2);
    uop!(Log2);

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

    pub fn to_texture(&self, shape: &[usize], n_channels: usize) -> Self {
        self.schedule();
        // TODO: deferr texture copy

        let size = shape.iter().cloned().reduce(|a, b| a * b).unwrap() * n_channels;
        assert_eq!(self.size(), size,);

        let texture = self
            .ir
            .lock()
            .backend
            .as_ref()
            .unwrap()
            .create_texture(shape, n_channels);
        let dst = self.ir.push_var(Var {
            op: Op::TexUpload,
            deps: smallvec![self.id()],
            ty: VarType::Void,
            size,
            data: Data::Texture(texture),
            ..Default::default()
        });
        dst.schedule();
        dst
    }
    pub fn tex_to_buffer(&self) -> Self {
        let dst = self.ir.buffer_f32(&vec![0.; self.size()]);
        let buf = dst.var().data.buffer().unwrap().clone();
        self.var()
            .data
            .texture()
            .unwrap()
            .copy_to_buffer(buf.as_ref());
        dst
    }

    pub fn tex_lookup(&self, pos: &[&VarRef]) -> smallvec::SmallVec<[Self; 4]> {
        assert_eq!(pos[0].size(), pos[1].size());
        let size = pos[0].size();
        assert!(pos.iter().all(|p| p.size() == size));

        let mut deps = smallvec![self.id()];
        deps.extend(pos.iter().map(|r| r.id()));

        let lookup = self.ir.push_var(Var {
            op: Op::TexLookup {
                dim: pos.len() as _,
            },
            deps,
            ty: VarType::F32,
            size,
            ..Default::default()
        });

        let n_dims = self.var().data.texture().unwrap().dimensions();
        (0..n_dims)
            .into_iter()
            .map(|i| {
                self.ir
                    .push_var_op(Op::Extract { offset: i }, &[&lookup], VarType::F32, size)
            })
            .collect::<smallvec::SmallVec<_>>()
    }
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
                let dep = VarRef::borrow_from(&self.ir, dep);
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

        let data = if let Some(lit) = v.data.literal() {
            Data::Literal(lit)
        } else {
            Data::None
        };

        drop(v);

        if op == Op::Idx {
            // self.inc_rc(new_idx);
            return Some(new_idx.clone());
        } else {
            return Some(self.ir.push_var(Var {
                op,
                deps,
                ty,
                size,
                rc: 0,
                data,
                ..Default::default()
            }));
        }
    }
    pub fn scatter_reduce(&self, dst: &Self, idx: &Self, mask: Option<&Self>, reduce_op: ReduceOp) {
        dst.schedule();

        let mask: VarRef = mask
            .map(|m| {
                assert!(Arc::ptr_eq(&self.ir, &m.ir));
                m.clone()
            })
            .unwrap_or(self.ir.literal_bool(true));

        let size = idx.var().size;

        let res = self.ir.push_var(Var {
            op: Op::Scatter { op: reduce_op },
            deps: smallvec![self.id(), dst.id(), idx.id(), mask.id()],
            size,
            ..Default::default()
        });
        res.schedule();
        dst.var().last_write = Some(res.id()); // Set side effect
        dst.ir.lock().inc_rc(dst.id);
    }
    pub fn scatter(&self, dst: &Self, idx: &Self, mask: Option<&Self>) {
        self.scatter_reduce(dst, idx, mask, ReduceOp::None);
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
            .unwrap_or(self.ir.literal_bool(true));

        // let res = self.as_ptr();

        if self.var().data.is_storage() {
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
                         // let mut jit = Jit::default();
                         // jit.eval(&mut self.ir.lock());

        // if self.var().buffer.is_some() {
        let ret = self.ir.push_var(Var {
            op: Op::Gather,
            deps: smallvec![self.id(), index.id(), mask.id()],
            ty,
            size,
            ..Default::default()
        });
        return ret;
    }

    pub fn trace_ray(
        &self,
        payload_count: usize,
        o: [&Self; 3],
        d: [&Self; 3],
        tmin: &Self,
        tmax: &Self,
        t: &Self,
        vis_mask: Option<&Self>,
        flags: Option<&Self>,
        sbt_offset: Option<&Self>,
        sbt_stride: Option<&Self>,
        miss_sbt: Option<&Self>,
        mask: Option<&Self>,
    ) -> Vec<Self> {
        let null = self.ir.literal_u32(0);
        let mask: Self = mask
            .map(|m| {
                assert!(Arc::ptr_eq(&self.ir, &m.ir));
                m.clone()
            })
            .unwrap_or(self.ir.literal_bool(true));
        let vis_mask = vis_mask
            .map(|m| {
                assert!(Arc::ptr_eq(&self.ir, &m.ir));
                m.clone()
            })
            .unwrap_or(self.ir.literal_u32(255));
        let flags = flags
            .map(|m| {
                assert!(Arc::ptr_eq(&self.ir, &m.ir));
                m.clone()
            })
            .unwrap_or(null.clone());
        let sbt_offset = sbt_offset
            .map(|m| {
                assert!(Arc::ptr_eq(&self.ir, &m.ir));
                m.clone()
            })
            .unwrap_or(null.clone());
        let sbt_stride = sbt_stride
            .map(|m| {
                assert!(Arc::ptr_eq(&self.ir, &m.ir));
                m.clone()
            })
            .unwrap_or(null.clone());
        let miss_sbt = miss_sbt
            .map(|m| {
                assert!(Arc::ptr_eq(&self.ir, &m.ir));
                m.clone()
            })
            .unwrap_or(null.clone());

        // assert_eq!(o[0].size(), o[1].size());
        // assert_eq!(o[0].size(), o[2].size());
        // assert_eq!(d[0].size(), d[1].size());
        // assert_eq!(d[0].size(), d[2].size());

        let size = o[0]
            .size()
            .max(o[1].size())
            .max(o[2].size())
            .max(d[0].size())
            .max(d[1].size())
            .max(d[2].size())
            .max(tmin.size())
            .max(tmax.size())
            .max(t.size());

        let rt = self.ir.push_var(Var {
            op: Op::TraceRay { payload_count },
            ty: VarType::Void,
            deps: smallvec![
                self.id(),
                mask.id(),
                o[0].id(),
                o[1].id(),
                o[2].id(),
                d[0].id(),
                d[1].id(),
                d[2].id(),
                tmin.id(),
                tmax.id(),
                t.id(),
                vis_mask.id(),
                flags.id(),
                sbt_offset.id(),
                sbt_stride.id(),
                miss_sbt.id(),
            ],
            size,
            ..Default::default()
        });

        let ret = (0..payload_count)
            .into_iter()
            .map(|i| {
                self.ir
                    .push_var_op(Op::Extract { offset: i }, &[&rt], VarType::U32, size)
            })
            .collect::<Vec<_>>();
        ret
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

#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use crate::jit::Jit;

    use super::Trace;

    macro_rules! test_uop {
        ($jop:ident($init:expr; $ty:ident) $(,$mod:literal)?) => {
            test_uop!($ty::$jop => $jop($init; $ty) $(,$mod)?);
        };
        ($rop:expr => $jop:ident($init:expr; $ty:ident) $(,$mod:literal)?) => {
            paste::paste! {
                #[test]
                fn [<$jop _$ty $(__$mod)?>]() {

                    let initial: &[$ty] = &$init;

                    let ir = Trace::default();
                    ir.set_backend("cuda");

                    let x = ir.[<buffer_$ty>](initial);

                    let y = x.$jop();

                    ir.schedule(&[&y]);

                    let mut jit = Jit::default();
                    jit.eval(&mut ir.lock());


                    for (i, calculated) in y.[<to_host_$ty>]().into_iter().enumerate(){
                        let expected = ($rop)(initial[i]);
                        approx::assert_abs_diff_eq!(calculated, expected, epsilon = 0.0001);
                    }
                }
            }
        };
    }

    test_uop!(|x:f32| {1./x} => rcp(         [0.1, 0.5, 1., std::f32::consts::PI]; f32));
    test_uop!(|x:f32| {1./x.sqrt()} => rsqrt([0.1, 0.5, 1., std::f32::consts::PI]; f32));
    test_uop!(sin(                           [0.1, 0.5, 1., std::f32::consts::PI]; f32));
    test_uop!(cos(                           [0.1, 0.5, 1., std::f32::consts::PI]; f32));
    test_uop!(exp2(                          [0.1, 0.5, 1., std::f32::consts::PI]; f32));
    test_uop!(log2(                          [0.1, 0.5, 1., std::f32::consts::PI]; f32));
}

use anyhow::{anyhow, bail, ensure, Result};
use half::f16;
use itertools::Itertools;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use resource_pool::hashpool::HashPool;
use std::cell::{RefCell, RefMut};
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::IntoIterator;
use std::ops::Deref;
use std::sync::Arc;

use bytemuck::cast_slice;
use slotmap::{DefaultKey, SlotMap};
use smallvec::{smallvec, SmallVec};

use crate::backend::Backend;
use crate::registry::BACKEND_REGISTRY;
use crate::schedule::Env;
pub use crate::var::{Data, Op, ReduceOp, Var, VarId, VarInfo, VarType};
use crate::{AsVarType, Jit};

pub use crate::backend::{HitGroupDesc, MissGroupDesc, ModuleDesc, SBTDesc};

pub enum GeometryDesc<'a> {
    Triangles {
        vertices: &'a VarRef,
        indices: &'a VarRef,
    },
}
pub struct InstanceDesc {
    pub hit_group: u32,
    pub geometry: usize,
    pub transform: [f32; 12],
}
pub struct AccelDesc<'a> {
    pub sbt: SBTDesc<'a>,
    pub geometries: &'a [GeometryDesc<'a>],
    pub instances: &'a [InstanceDesc],
}

///
/// A wrapper arrund an Intermediate Representation.
///
#[derive(Clone, Debug, Default)]
pub struct Trace {
    internal: Arc<Mutex<Internal>>,
    pub jit: Arc<Mutex<Jit>>,
}
impl Deref for Trace {
    type Target = Arc<Mutex<Internal>>;

    fn deref(&self) -> &Self::Target {
        &self.internal
    }
}

impl Trace {
    fn push_var(&self, mut v: Var) -> VarRef {
        // Push side effects of sources as extra dependencies
        // for dep in v.deps.clone() {
        //     if let Some(se) = self.lock().var(dep).last_write {
        //         v.deps.push(se);
        //     }
        // }
        // Inc rc for deps
        for dep in v.deps.iter() {
            self.lock().inc_rc(*dep);
        }
        v.rc = 1;
        let id = VarId(self.lock().vars.insert(v));
        VarRef::steal_from(&self, id)
    }
    fn var_info(&self, refs: &[&VarRef]) -> Result<VarInfo> {
        for (a, b) in refs.iter().tuple_windows() {
            ensure!(Arc::ptr_eq(&a.ir.internal, &b.ir.internal))
        }
        let ty = refs.first().unwrap().var().ty.clone(); // TODO: Fix (first non void)

        let size = refs
            .iter()
            .map(|id| id.var().size)
            .reduce(|s0, s1| s0.max(s1))
            .unwrap()
            .clone();
        Ok(VarInfo { ty, size })
    }
    fn push_var_op(&self, op: Op, deps: &[&VarRef], ty: VarType, size: usize) -> Result<VarRef> {
        for dep in deps {
            ensure!(Arc::ptr_eq(&self.internal, &dep.ir));
        }
        let deps: smallvec::SmallVec<[VarId; 4]> = deps.iter().map(|r| r.id()).collect();
        let ret = self.push_var(Var {
            op,
            deps,
            ty,
            size,
            ..Default::default()
        });
        Ok(ret)
    }
    pub fn set_backend<'a>(
        &self,
        backends: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<()> {
        ensure!(
            self.lock().backend.is_none(),
            "Overwriting the backend is currently not supported!"
        );

        let backend_registry = BACKEND_REGISTRY.lock();

        for name in backends.into_iter() {
            let name = name.as_ref();
            let factory = backend_registry
                .get(name)
                .ok_or(anyhow!("Could not find {name} backend!"))?;

            match factory() {
                Ok(backend) => {
                    log::trace!("Selected backend {name}");
                    self.lock().backend = Some(backend);
                    break;
                }
                Err(err) => log::trace!("Could not initialize {name} backend with error: {err}"),
            }
        }

        Ok(())
    }
    pub fn backend(&self) -> MappedMutexGuard<dyn Backend> {
        MutexGuard::map(self.internal.lock(), |ir| {
            ir.backend.as_mut().unwrap().as_mut()
        })
    }
    pub fn backend_as<T: Backend + 'static>(&self) -> MappedMutexGuard<T> {
        MutexGuard::map(self.internal.lock(), |ir| {
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
            assert!(Arc::ptr_eq(&r.ir, &self.internal));
            self.lock().schedule(&[r.id()])
        }
    }
    pub fn eval(&self) -> Result<()> {
        self.jit.lock().eval(&mut self.internal.lock())
    }
    pub fn kernel_history(&self) -> String {
        self.jit.lock().kernel_history()
    }
    pub fn kernel_cache_size(&self) -> usize {
        self.jit.lock().kernels.len()
    }
    pub fn n_variables(&self) -> usize {
        self.internal.lock().vars.len()
    }
}
impl Trace {
    pub fn array<T: AsVarType>(&self, slice: &[T]) -> Result<VarRef> {
        let ty = T::as_var_type();
        let buffer = self
            .lock()
            .backend
            .as_ref()
            .ok_or(anyhow!("Backend not initialized!"))?
            .buffer_from_slice(unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const _, slice.len() * ty.size())
            })?;
        Ok(self.push_var(Var {
            data: Data::Buffer(buffer),
            size: slice.len(),
            ty,
            op: Op::Data,
            ..Default::default()
        }))
    }
    pub fn array_uninit<T: AsVarType>(&self, size: usize) -> Result<VarRef> {
        let ty = T::as_var_type();
        let buffer = self
            .lock()
            .backend
            .as_ref()
            .unwrap()
            .buffer_uninit(size * ty.size())?;
        Ok(self.push_var(Var {
            data: Data::Buffer(buffer),
            size,
            ty,
            op: Op::Data,
            ..Default::default()
        }))
    }

    pub fn literal<T: AsVarType>(&self, val: T) -> Result<VarRef> {
        self.sized_literal(val, 1)
    }
    pub fn sized_literal<T: AsVarType>(&self, val: T, size: usize) -> Result<VarRef> {
        let ty = T::as_var_type();
        ensure!(ty.size() == std::mem::size_of::<T>());
        let mut data: u64 = 0;
        unsafe { *(&mut data as *mut _ as *mut T) = val };
        Ok(self.push_var(Var {
            op: Op::Literal,
            deps: smallvec![],
            ty,
            data: Data::Literal(data),
            size,
            ..Default::default()
        }))
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
    pub fn texture(&self, shape: &[usize], n_channels: usize) -> Result<VarRef> {
        let size = shape.iter().cloned().reduce(|a, b| a * b).unwrap() * n_channels;
        let texture = self
            .internal
            .lock()
            .backend
            .as_ref()
            .unwrap()
            .create_texture(shape, n_channels)?;
        Ok(self.push_var(Var {
            op: Op::Nop,
            ty: VarType::Void,
            size,
            data: Data::Texture(texture),
            ..Default::default()
        }))
    }
    pub fn accel(&self, desc: AccelDesc) -> Result<VarRef> {
        let gds = desc
            .geometries
            .iter()
            .map(|g| match g {
                GeometryDesc::Triangles { vertices, indices } => {
                    let vertices = vertices.var().data.buffer().unwrap().clone();
                    let indices = indices.var().data.buffer().unwrap().clone();
                    crate::backend::GeometryDesc::Triangles { vertices, indices }
                }
            })
            .collect::<Vec<_>>();
        let instances = desc
            .instances
            .iter()
            .map(|inst| crate::backend::InstanceDesc {
                hit_goup: inst.hit_group,
                geometry: inst.geometry,
                transform: inst.transform,
            })
            .collect::<Vec<_>>();
        let desc = crate::backend::AccelDesc {
            sbt: desc.sbt,
            geometries: gds.as_slice(),
            instances: instances.as_slice(),
        };
        // let vertices = vertices.var().data.buffer().unwrap().clone();
        // let indices = indices.var().data.buffer().unwrap().clone();
        let accel = self.lock().backend.as_ref().unwrap().create_accel(desc)?;
        Ok(self.push_var(Var {
            op: Op::Nop,
            ty: VarType::Void,
            size: 0,
            data: Data::Accel(accel),
            ..Default::default()
        }))
    }
    pub fn loop_record(&self, before: &[&VarRef], after: &[&VarRef]) {
        let after = after.iter().map(|r| (*r).clone()).collect::<Vec<_>>();
        let before = before.iter().map(|r| (*r).clone()).collect::<Vec<_>>();
        self.push_var(Var {
            ..Default::default()
        });
    }
}

#[derive(Default, Debug)]
pub struct Internal {
    vars: SlotMap<DefaultKey, Var>,
    pub backend: Option<Box<dyn Backend>>,
    pub scheduled: Vec<VarId>,
}
impl Drop for Internal {
    fn drop(&mut self) {
        self.clear_schedule();
        assert_eq!(self.vars.len(), 0);
    }
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
            // if let Some(se) = var.last_write.clone() {
            //     self.dec_rc(se);
            // }
            self.vars.remove(id.0);
        }
    }
    pub fn schedule(&mut self, ids: &[VarId]) {
        for id in ids {
            if self.scheduled.contains(id) {
                continue;
            }
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
    pub fn clear_schedule(&mut self) {
        for id in self.scheduled.clone() {
            // let var = self.var_mut(id);
            // let deps = var.deps.clone();
            // for dep in deps {
            //     self.dec_rc(dep);
            // }
            self.dec_rc(id);
        }
        self.scheduled.clear();
    }
}

///
/// Reference to a variable.
///
#[derive(Debug)]
pub struct VarRef {
    id: VarId,
    pub ir: Trace,
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
macro_rules! bop {
    ($Op:ident $(-> $ty:ident)?) => {
        paste::paste! {
            pub fn [<$Op:lower>](&self, rhs: &VarRef) -> Result<VarRef> {
                #[allow(unused_mut)]
                let mut info = self.ir.var_info(&[self, &rhs])?;
                $(info.ty = VarType::$ty;)?
                self.ir
                    .push_var_op(Op::$Op, &[self, &rhs], info.ty, info.size)
            }
        }
    };
}
macro_rules! uop {
    ($Op:ident) => {
        paste::paste! {
            pub fn [<$Op:lower>](&self) -> Result<Self> {
                let info = self.ir.var_info(&[self])?;
                self.ir.push_var_op(Op::$Op, &[self], info.ty, info.size)
            }
        }
    };
}
macro_rules! top {
    ($Op:ident) => {
        paste::paste! {
            pub fn [<$Op:lower>](&self, d1: &VarRef, d2: &VarRef) -> Result<VarRef> {
                let info = self.ir.var_info(&[self, &d1, &d2])?;
                self.ir
                    .push_var_op(Op::$Op, &[self, &d1, &d2], info.ty, info.size)
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

    pub fn to_host<T: AsVarType>(&self) -> Result<Vec<T>> {
        let var = self.var();
        let ty = T::as_var_type();
        ensure!(
            var.ty == ty,
            "Type of variable {:?} does not match requested type {:?}!",
            var.ty,
            ty
        );

        let mut dst = Vec::with_capacity(var.size);
        unsafe {
            var.data
                .buffer()
                .ok_or(anyhow!("Variable is not a buffer!"))?
                .copy_to_host(std::slice::from_raw_parts_mut(
                    dst.as_mut_ptr() as *mut _,
                    var.size * ty.size(),
                ));
        }
        unsafe { dst.set_len(var.size) };
        Ok(dst)
    }

    // Unarry operations:
    pub fn cast(&self, ty: &VarType) -> Result<VarRef> {
        self.ir
            .push_var_op(Op::Cast, &[self], ty.clone(), self.size())
    }
    pub fn bitcast(&self, ty: &VarType) -> Result<VarRef> {
        assert_eq!(self.var().ty.size(), ty.size());
        self.ir
            .push_var_op(Op::Bitcast, &[self], ty.clone(), self.size())
    }

    // Unarry Operations
    uop!(Rcp);
    uop!(Rsqrt);
    uop!(Sin);
    uop!(Cos);
    uop!(Exp2);
    uop!(Log2);

    uop!(Neg);
    uop!(Not);
    uop!(Abs);

    uop!(Ceil);
    uop!(Floor);
    uop!(Round);
    uop!(Trunc);

    uop!(Popc);
    uop!(Clz);
    uop!(Ctz);

    // Binarry operations:
    bop!(Add);
    bop!(Sub);
    bop!(Mul);
    bop!(Div);

    bop!(Min);
    bop!(Max);

    bop!(Eq -> Bool);
    bop!(Neq -> Bool);
    bop!(Lt -> Bool);
    bop!(Le -> Bool);
    bop!(Gt -> Bool);
    bop!(Ge -> Bool);

    bop!(Or);
    bop!(Xor);
    bop!(Shl);
    bop!(Shr);

    top!(Fma);
    // top!(Select);

    pub fn select(&self, var_true: &VarRef, var_false: &VarRef) -> Result<VarRef> {
        ensure!(
            self.ty() == VarType::Bool,
            "Mask has to be of type Bool but is of type {:?}!",
            self.ty()
        );
        let info = self.ir.var_info(&[&var_true, &var_false, self])?;
        self.ir.push_var_op(
            Op::Select,
            &[self, &var_true, &var_false],
            info.ty,
            info.size,
        )
    }

    pub fn modulo(&self, rhs: &VarRef) -> Result<VarRef> {
        let info = self.ir.var_info(&[self, &rhs])?;
        self.ir
            .push_var_op(Op::Mod, &[self, &rhs], info.ty, info.size)
    }
    bop!(Mulhi);

    pub fn and(&self, rhs: &VarRef) -> Result<VarRef> {
        assert!(Arc::ptr_eq(&self.ir, &rhs.ir));
        let info = self.ir.var_info(&[self, &rhs])?;

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

    pub fn to_texture(&self, shape: &[usize], n_channels: usize) -> Result<Self> {
        self.schedule();
        self.ir.eval()?;

        let size = shape.iter().cloned().reduce(|a, b| a * b).unwrap() * n_channels;
        ensure!(self.size() == size);

        let texture = self
            .ir
            .lock()
            .backend
            .as_ref()
            .unwrap()
            .create_texture(shape, n_channels)?;
        texture.copy_from_buffer(self.var().data.buffer().unwrap().as_ref())?;
        let dst = self.ir.push_var(Var {
            op: Op::Nop,
            deps: smallvec![self.id()],
            ty: VarType::Void,
            size,
            data: Data::Texture(texture),
            ..Default::default()
        });
        Ok(dst)
    }
    pub fn tex_to_buffer(&self) -> Result<Self> {
        let dst = self.ir.array(&vec![0.; self.size()])?;
        let buf = dst.var().data.buffer().unwrap().clone();
        self.var()
            .data
            .texture()
            .ok_or(anyhow!("Variable is not a texture!"))?
            .copy_to_buffer(buf.as_ref())?;
        Ok(dst)
    }

    pub fn tex_lookup(&self, pos: &[&VarRef]) -> Result<SmallVec<[Self; 4]>> {
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
            .collect()
    }
    /// Reindex a variable with a new index and size.
    /// (Normally size is the size of the index)
    ///
    /// For now we construct a separate set of variables in the Ir.
    /// However, it should sometimes be possible to reuse the old ones.
    ///
    fn reindex(&self, new_idx: &VarRef, size: usize) -> Result<VarRef> {
        assert!(Arc::ptr_eq(&self.ir, &new_idx.ir));
        // let mut ir = self.ir.lock();

        let v = self.var();
        let is_literal = v.is_literal();
        let is_data = v.is_data();
        let v_deps = v.deps.clone();

        if is_data {
            bail!("The variable contains data and cannot be reindexed!");
        }

        drop(v);

        let mut deps = smallvec![];
        if !is_literal {
            for dep in v_deps {
                let dep = VarRef::borrow_from(&self.ir, dep);
                deps.push(dep.reindex(new_idx, size)?.id());
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
            return Ok(new_idx.clone());
        } else {
            return Ok(self.ir.push_var(Var {
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
    pub fn scatter_reduce(
        &self,
        target: &Self,
        idx: &Self,
        mask: Option<&Self>,
        reduce_op: ReduceOp,
    ) -> Result<()> {
        target.schedule();
        self.ir.eval()?;

        let mask: VarRef = mask
            .map(|m| {
                assert!(Arc::ptr_eq(&self.ir, &m.ir));
                m.clone()
            })
            .unwrap_or(self.ir.literal(true)?);

        let size = idx.var().size;

        let res = self.ir.push_var(Var {
            op: Op::Scatter { op: reduce_op },
            deps: smallvec![self.id(), target.id(), idx.id(), mask.id()],
            size,
            ..Default::default()
        });
        target.var().dirty = true;
        res.schedule();
        Ok(())
    }
    pub fn scatter(&self, dst: &Self, idx: &Self, mask: Option<&Self>) -> Result<()> {
        self.scatter_reduce(dst, idx, mask, ReduceOp::None)
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
    pub fn gather(&self, index: &VarRef, mask: Option<&VarRef>) -> Result<VarRef> {
        assert!(Arc::ptr_eq(&self.ir, &index.ir));

        let size = index.var().size;
        let ty = self.var().ty.clone();

        if self.var().dirty || index.var().dirty {
            self.ir.eval()?;
        }

        let mask: VarRef = mask
            .map(|m| {
                assert!(Arc::ptr_eq(&self.ir, &m.ir));
                m.clone()
            })
            .unwrap_or(self.ir.literal(true)?);

        // let res = self.as_ptr();

        if self.var().data.is_storage() {
            let ret = self.ir.push_var(Var {
                op: Op::Gather,
                deps: smallvec![self.id(), index.id(), mask.id()],
                ty,
                size,
                ..Default::default()
            });
            return Ok(ret);
        }

        let res = self.reindex(&index, size);

        match res {
            Ok(res) => {
                let res = res.and(&mask);
                return res;
            }
            Err(err) => {
                log::trace!("Reindexing failed. Adding gather operation");
            }
        }

        self.schedule();
        self.ir.eval()?;

        let ret = self.ir.push_var(Var {
            op: Op::Gather,
            deps: smallvec![self.id(), index.id(), mask.id()],
            ty,
            size,
            ..Default::default()
        });
        return Ok(ret);
    }

    pub fn trace_ray(
        &self,
        payload: &[&Self],
        o: [&Self; 3],
        d: [&Self; 3],
        tmin: &Self,
        tmax: &Self,
        t: &Self,
        vis_mask: Option<&Self>,
        ray_flags: Option<&Self>,
        sbt_offset: Option<&Self>,
        sbt_stride: Option<&Self>,
        miss_sbt: Option<&Self>,
        mask: Option<&Self>,
    ) -> Result<Vec<Self>> {
        let null = self.ir.literal(0u32)?;
        let mask: Self = mask.cloned().unwrap_or(self.ir.literal(true)?);
        let vis_mask = vis_mask.cloned().unwrap_or(self.ir.literal(255u32)?);
        let ray_flags = ray_flags.cloned().unwrap_or(null.clone());
        let sbt_offset = sbt_offset.cloned().unwrap_or(null.clone());
        let sbt_stride = sbt_stride.cloned().unwrap_or(null.clone());
        let miss_sbt = miss_sbt.cloned().unwrap_or(null.clone());

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

        let mut v = Var {
            op: Op::TraceRay {
                payload_count: payload.len(),
            },
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
                ray_flags.id(),
                sbt_offset.id(),
                sbt_stride.id(),
                miss_sbt.id(),
            ],
            size,
            ..Default::default()
        };
        v.deps.extend(payload.iter().map(|i| i.id()));
        let rt = self.ir.push_var(v);

        (0..payload.len())
            .into_iter()
            .map(|i| {
                self.ir
                    .push_var_op(Op::Extract { offset: i }, &[&rt], VarType::U32, size)
            })
            .collect()
    }
    pub fn make_opaque(&self) {
        if self.var().is_literal() {
            self.var().opaque = true;
        } else {
            self.schedule();
        }
    }
    pub fn opaque(&self) -> Self {
        let var = self.var().clone();
        let res = self.ir.push_var(var);
        res.make_opaque();
        res
    }
    pub fn backend_ident(&self) -> &'static str {
        self.ir.backend().ident()
    }
    pub fn ptr(&self) -> Option<u64> {
        match self.var().data {
            Data::Buffer(ref buf) => buf.ptr(),
            _ => None,
        }
    }
    pub fn compress(&self) -> Result<Self> {
        let ty = self.ty();
        ensure!(
            ty == VarType::Bool,
            "Cannot compress array of type {:?}!",
            ty
        );
        self.schedule();
        self.ir.eval()?;

        let mask = self
            .var()
            .data
            .buffer()
            .ok_or(anyhow!("Mask is not data after evaluation!"))?
            .clone();
        let indices = mask.compress()?;
        let size = indices.size() / std::mem::size_of::<u32>();

        Ok(self.ir.push_var(Var {
            ty: VarType::U32,
            op: Op::Data,
            size,
            data: Data::Buffer(indices),
            ..Default::default()
        }))
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

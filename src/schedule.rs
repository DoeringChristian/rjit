use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use smallvec::{smallvec, SmallVec};

use crate::backend::{Accel, Buffer, Texture};
use crate::trace::Internal;
use crate::var::{self, Op, VarId, VarType};
use crate::ReduceOp;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SVarId(pub usize);
impl std::fmt::Display for SVarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub enum Data {
    #[default]
    None,
    BufferIdx(u64),
    TextureIdx(u64),
    AccelIdx(u64),
    Literal(u64),
    OpaqueIdx(u64),
}
impl Data {
    pub fn is_none(&self) -> bool {
        return match self {
            Data::None => true,
            _ => false,
        };
    }
    pub fn buffer(&self) -> Option<u64> {
        match self {
            Data::BufferIdx(id) => Some(*id),
            _ => None,
        }
    }
    pub fn accel(&self) -> Option<u64> {
        match self {
            Data::AccelIdx(id) => Some(*id),
            _ => None,
        }
    }
    pub fn texture(&self) -> Option<u64> {
        match self {
            Data::TextureIdx(id) => Some(*id),
            _ => None,
        }
    }
    pub fn literal(&self) -> Option<u64> {
        match self {
            Data::Literal(lit) => Some(*lit),
            _ => None,
        }
    }
    pub fn opaque(&self) -> Option<u64> {
        match self {
            Data::OpaqueIdx(id) => Some(*id),
            _ => None,
        }
    }
}

///
/// A Representation of a variable used in the ScheduleIr.
/// This only holds data that is necessary to compile the Kernel.
///
/// Variables are densly stored in the ScheduleIr, simplifying the compilation.
///
// TODO: byte hash
//#[derive(bytemuck::NoUninit, bytemuck::ByteHash)]
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ScheduleVar {
    pub op: Op,
    pub ty: VarType,

    pub deps: (usize, usize),

    pub data: Data,

    // We have to build a new kernel when we get new hit/miss shaders.
    pub sbt_hash: u64,
}

impl ScheduleVar {
    pub fn is_literal(&self) -> bool {
        self.op == Op::Literal
    }
}

#[derive(Debug)]
pub struct BufferAccess {
    pub buffer: Arc<dyn Buffer>,
    pub write: bool,
}

#[derive(Debug, Default)]
pub struct Env {
    opaques: Vec<u64>,
    buffers: Vec<BufferAccess>,
    textures: Vec<Arc<dyn Texture>>,
    accels: Vec<Arc<dyn Accel>>,
}

impl Env {
    fn push_opaque(&mut self, literal: u64) -> Data {
        let idx = self.opaques.len();
        self.opaques.push(literal);
        Data::OpaqueIdx(idx as _)
    }
    pub fn push_buffer(&mut self, buf: &Arc<dyn Buffer>, write: bool) -> Data {
        let idx = self.buffers.len();
        self.buffers.push(BufferAccess {
            buffer: buf.clone(),
            write,
        });
        Data::BufferIdx(idx as _)
    }
    fn push_texture(&mut self, tex: &Arc<dyn Texture>) -> Data {
        let idx = self.textures.len();
        self.textures.push(tex.clone());
        Data::TextureIdx(idx as _)
    }
    fn push_accel(&mut self, accel: &Arc<dyn Accel>) -> Data {
        let idx = self.accels.len();
        self.accels.push(accel.clone());
        Data::AccelIdx(idx as _)
    }
    pub fn buffers(&self) -> &[BufferAccess] {
        &self.buffers
    }
    pub fn textures(&self) -> &[Arc<dyn Texture>] {
        &self.textures
    }
    pub fn accels(&self) -> &[Arc<dyn Accel>] {
        &self.accels
    }
    pub fn opaques(&self) -> &[u64] {
        &self.opaques
    }
}

/// Intermediary representation for scheduled variables
///
/// * `vars`: Variables to be used in code generation
/// * `n_regs`: Total number of registers required
/// * `n_payloads`: Number of ray tracing payloads
/// * `visited`: HashMap used for construction
/// * `independent`: HashMap of variables without dependencies used to dedup literals
#[derive(Debug, Default)]
pub struct ScheduleIr {
    vars: Vec<ScheduleVar>,
    deps: Vec<SVarId>,

    n_payloads: usize,

    visited: HashMap<VarId, SVarId>,
    independent: HashMap<ScheduleVar, SVarId>,
}

impl ScheduleIr {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
    pub fn ids(&self) -> impl Iterator<Item = SVarId> {
        (0..self.vars.len()).map(|i| SVarId(i))
    }
    pub fn var(&self, id: SVarId) -> &ScheduleVar {
        &self.vars[id.0]
    }
    pub fn var_mut(&mut self, id: SVarId) -> &mut ScheduleVar {
        &mut self.vars[id.0]
    }
    pub fn n_payloads(&self) -> usize {
        self.n_payloads
    }
    pub fn n_vars(&self) -> usize {
        self.vars.len()
    }
    pub fn deps(&self, id: SVarId) -> &[SVarId] {
        let (deps_start, deps_end) = self.var(id).deps;
        &self.deps[deps_start..deps_end]
    }
    pub fn dep(&self, id: SVarId, idx: usize) -> SVarId {
        let (deps_start, deps_end) = self.var(id).deps;
        self.deps[deps_start..deps_end][idx]
    }
    fn push_var(&mut self, mut var: ScheduleVar, deps: impl IntoIterator<Item = SVarId>) -> SVarId {
        let deps = deps.into_iter();
        let dep_start = self.deps.len();
        self.deps.extend(deps);
        let dep_end = self.deps.len();
        var.deps = (dep_start, dep_end);

        if dep_end - dep_start == 0 {
            //for hashing set indices to 0
            var.deps = (0, 0);
            // We can reuse variables if they have no dependencies
            if self.independent.contains_key(&var) {
                *self.independent.get(&var).unwrap()
            } else {
                let id = SVarId(self.vars.len());
                self.independent.insert(var.clone(), id);

                self.vars.push(var);
                id
            }
        } else {
            let id = SVarId(self.vars.len());
            self.vars.push(var);
            id
        }
    }
    pub fn collect_vars(&mut self, env: &mut Env, ir: &Internal, schedule: &[VarId]) {
        for id in schedule {
            let sv_id = self.collect(env, ir, *id);

            let var = ir.var(*id);
            if var.ty.size() == 0 {
                continue;
            }

            // Fake scatter
            let dst = self.collect_data(env, ir, *id, true);
            let idx = self.push_var(
                ScheduleVar {
                    op: Op::Idx,
                    ty: VarType::U32,
                    ..Default::default()
                },
                [],
            );
            let mask = self.push_var(
                ScheduleVar {
                    op: Op::Literal,
                    ty: VarType::Bool,
                    data: Data::Literal(1),
                    ..Default::default()
                },
                [],
            );

            self.push_var(
                ScheduleVar {
                    op: Op::Scatter { op: ReduceOp::None },
                    ty: var.ty.clone(),
                    ..Default::default()
                },
                [sv_id, dst, idx, mask],
            );
        }
    }
    ///
    /// Traverse computation graph and collect variables into Schedule.
    ///
    /// If a gather operation is encountered, that only depends on trivial operations we can
    /// reindex it using the parameter idx.
    ///
    pub fn collect(&mut self, env: &mut Env, ir: &Internal, id: VarId) -> SVarId {
        if self.visited.contains_key(&id) {
            return self.visited[&id];
        }

        let var = ir.var(id);

        // let mut sv = ScheduleVar {
        //     op: var.op,
        //     ty: var.ty.clone(),
        //     ..Default::default()
        // };

        // Collect dependencies

        let svid = match var.op {
            Op::Data => {
                let bv = self.collect_data(env, ir, id, false);
                // Fake gather
                let idx = if var.size > 1 {
                    self.push_var(
                        ScheduleVar {
                            op: Op::Idx,
                            ty: VarType::U32,
                            ..Default::default()
                        },
                        [],
                    )
                } else {
                    self.push_var(
                        ScheduleVar {
                            op: Op::Literal,
                            ty: VarType::U32,
                            data: Data::Literal(0),
                            ..Default::default()
                        },
                        [],
                    )
                };
                let mask = self.push_var(
                    ScheduleVar {
                        op: Op::Literal,
                        ty: VarType::Bool,
                        data: Data::Literal(1),
                        ..Default::default()
                    },
                    [],
                );
                self.push_var(
                    ScheduleVar {
                        op: Op::Gather,
                        ty: var.ty.clone(),
                        ..Default::default()
                    },
                    [bv, idx, mask],
                )
            }
            Op::Literal => {
                // TODO: cannot evaluate a literal (maybe neccesarry for tensors)
                // sv.param_offset = self.push_param(var.literal);
                if var.opaque {
                    self.push_var(
                        ScheduleVar {
                            op: Op::Literal,
                            ty: var.ty.clone(),
                            data: env.push_opaque(var.data.literal().unwrap()),
                            ..Default::default()
                        },
                        [],
                    )
                } else {
                    self.push_var(
                        ScheduleVar {
                            op: Op::Literal,
                            ty: var.ty.clone(),
                            data: Data::Literal(var.data.literal().unwrap()),
                            ..Default::default()
                        },
                        [],
                    )
                }
            }
            Op::Gather => {
                let src = self.collect_data(env, ir, var.deps[0], false);
                let index = self.collect(env, ir, var.deps[1]);
                let mask = self.collect(env, ir, var.deps[2]);
                self.push_var(
                    ScheduleVar {
                        op: Op::Gather,
                        ty: var.ty.clone(),
                        ..Default::default()
                    },
                    [src, index, mask],
                )
            }
            Op::Scatter { .. } => {
                let src = self.collect(env, ir, var.deps[0]);
                let dst = self.collect_data(env, ir, var.deps[1], true);
                let index = self.collect(env, ir, var.deps[2]);
                let mask = self.collect(env, ir, var.deps[3]);
                self.push_var(
                    ScheduleVar {
                        op: var.op,
                        ty: var.ty.clone(),
                        ..Default::default()
                    },
                    [src, dst, index, mask],
                )
            }
            Op::TexLookup { dim, channels } => {
                let deps = [self.collect_data(env, ir, var.deps[0], false)]
                    .into_iter()
                    .chain(
                        var.deps[1..(dim as usize + 1)]
                            .iter()
                            .map(|dep| self.collect(env, ir, *dep)),
                    )
                    .collect::<Vec<_>>();
                self.push_var(
                    ScheduleVar {
                        op: var.op,
                        ty: var.ty.clone(),
                        ..Default::default()
                    },
                    deps,
                )
            }
            Op::TraceRay { payload_count } => {
                self.n_payloads = self.n_payloads.max(payload_count);
                let deps = [self.collect_data(env, ir, var.deps[0], false)]
                    .into_iter()
                    .chain(
                        var.deps[1..(16 + payload_count)]
                            .iter()
                            .map(|dep| self.collect(env, ir, *dep)),
                    )
                    .collect::<Vec<_>>();

                self.push_var(
                    ScheduleVar {
                        op: var.op,
                        ty: var.ty.clone(),
                        ..Default::default()
                    },
                    deps,
                )
            }
            Op::Loop {} => {
                todo!()
            }
            _ => {
                let deps = var
                    .deps
                    .iter()
                    .map(|id| self.collect(env, ir, *id))
                    .collect::<SmallVec<[_; 4]>>();
                self.push_var(
                    ScheduleVar {
                        op: var.op,
                        ty: var.ty.clone(),
                        ..Default::default()
                    },
                    deps,
                )
            }
        };

        self.visited.insert(id, svid);

        svid
    }
    ///
    /// Collect variable only as data input/output (for example when it is src/dst for a
    /// gather/scatter operation).
    ///
    /// This only inserts this variable but not its dependencies.
    ///
    pub fn collect_data(&mut self, env: &mut Env, ir: &Internal, id: VarId, write: bool) -> SVarId {
        let var = ir.var(id);
        if let Some(id) = self.visited.get(&id).cloned() {
            // In case this variable has already been traversed, just ensure that the buffer is
            // added as a parameter.
            // let sv = self.var(id);
            if self.var(id).data.is_none() {
                self.var_mut(id).data = match &var.data {
                    var::Data::None => Data::None,
                    var::Data::Literal(_) => Data::None,
                    var::Data::Buffer(buf) => env.push_buffer(&buf, write),
                    var::Data::Texture(tex) => env.push_texture(&tex),
                    var::Data::Accel(accel) => env.push_accel(&accel),
                };
            }
            id
        } else {
            let data = match &var.data {
                var::Data::None => Data::None,
                var::Data::Literal(_) => Data::None,
                var::Data::Buffer(buf) => env.push_buffer(&buf, write),
                var::Data::Texture(tex) => env.push_texture(&tex),
                var::Data::Accel(accel) => env.push_accel(&accel),
            };

            let sbt_hash = var.data.accel().map(|accel| accel.sbt_hash()).unwrap_or(0);
            let svid = self.push_var(
                ScheduleVar {
                    op: Op::Data,
                    ty: var.ty.clone(),
                    data,
                    sbt_hash,
                    ..Default::default()
                },
                [],
            );
            self.visited.insert(id, svid);
            svid
        }
    }
    pub fn internal_hash(&self) -> u128 {
        let mut hasher = fasthash::murmur3::Hasher128_x64::default();
        self.vars.hash(&mut hasher);
        self.deps.hash(&mut hasher);

        hasher.finish() as _
    }
}

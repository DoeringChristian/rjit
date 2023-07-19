use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use smallvec::{smallvec, SmallVec};

use crate::backend::{Accel, Buffer, Texture};
use crate::trace::Internal;
use crate::var::{self, Op, VarId, VarType};
use crate::ReduceOp;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SVarId(pub usize);
impl std::fmt::Display for SVarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Default, Hash, Clone, Copy)]
pub enum DataIdx {
    #[default]
    None,
    Buffer(u64),
    Texture(u64),
    Accel(u64),
    Literal(u64),
    Opaque(u64),
}
impl DataIdx {
    pub fn is_none(&self) -> bool {
        return match self {
            DataIdx::None => true,
            _ => false,
        };
    }
    pub fn buffer(&self) -> Option<u64> {
        match self {
            DataIdx::Buffer(id) => Some(*id),
            _ => None,
        }
    }
    pub fn accel(&self) -> Option<u64> {
        match self {
            DataIdx::Accel(id) => Some(*id),
            _ => None,
        }
    }
    pub fn texture(&self) -> Option<u64> {
        match self {
            DataIdx::Texture(id) => Some(*id),
            _ => None,
        }
    }
    pub fn literal(&self) -> Option<u64> {
        match self {
            DataIdx::Literal(lit) => Some(*lit),
            _ => None,
        }
    }
    pub fn opaque(&self) -> Option<u64> {
        match self {
            DataIdx::Opaque(id) => Some(*id),
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
#[derive(Debug, Default, Hash, Clone)]
pub struct ScheduleVar {
    pub op: Op,
    pub deps: SmallVec<[SVarId; 4]>,
    pub ty: VarType,
    pub reg: usize,

    pub data: DataIdx,

    // We have to build a new kernel when we get new hit/miss shaders.
    pub sbt_hash: u64,
}

impl ScheduleVar {
    pub fn is_literal(&self) -> bool {
        self.op == Op::Literal
    }
}

#[derive(Debug, Default)]
pub struct Env {
    opaques: Vec<u64>,
    buffers: Vec<Arc<dyn Buffer>>,
    textures: Vec<Arc<dyn Texture>>,
    accels: Vec<Arc<dyn Accel>>,
}

impl Env {
    fn push_opaque(&mut self, literal: u64) -> DataIdx {
        let idx = self.opaques.len();
        self.opaques.push(literal);
        DataIdx::Opaque(idx as _)
    }
    pub fn push_buffer(&mut self, buf: &Arc<dyn Buffer>) -> DataIdx {
        let idx = self.buffers.len();
        self.buffers.push(buf.clone());
        DataIdx::Buffer(idx as _)
    }
    fn push_texture(&mut self, tex: &Arc<dyn Texture>) -> DataIdx {
        let idx = self.textures.len();
        self.textures.push(tex.clone());
        DataIdx::Texture(idx as _)
    }
    fn push_accel(&mut self, accel: &Arc<dyn Accel>) -> DataIdx {
        let idx = self.accels.len();
        self.accels.push(accel.clone());
        DataIdx::Accel(idx as _)
    }
    pub fn buffers(&self) -> &[Arc<dyn Buffer>] {
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

///
/// Intermediate representation for scheduled variables
/// TODO: split into ir and env
///
#[derive(Debug, Default)]
pub struct ScheduleIr {
    vars: Vec<ScheduleVar>,

    n_regs: usize,
    n_payloads: usize,

    visited: HashMap<VarId, SVarId>,
}

impl ScheduleIr {
    pub fn new(first_register: usize) -> Self {
        Self {
            n_regs: first_register,
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
    // pub fn reg(&self, id: SVarId) -> Reg {
    //     self.var(id).reg()
    // }
    pub fn n_regs(&self) -> usize {
        self.n_regs
    }
    pub fn n_payloads(&self) -> usize {
        self.n_payloads
    }
    fn next_reg(&mut self) -> usize {
        let reg = self.n_regs;
        self.n_regs += 1;
        reg
    }
    fn push_var(&mut self, mut var: ScheduleVar) -> SVarId {
        let id = SVarId(self.vars.len());
        var.reg = self.next_reg();
        self.vars.push(var);
        id
    }
    pub fn collect_vars(&mut self, env: &mut Env, ir: &Internal, schedule: &[VarId]) {
        for id in schedule {
            let sv_id = self.collect(env, ir, *id);

            let var = ir.var(*id);
            if var.ty.size() == 0 {
                continue;
            }

            // Fake scatter
            let dst = self.collect_data(env, ir, *id);
            let idx = self.push_var(ScheduleVar {
                op: Op::Idx,
                ty: VarType::U32,
                ..Default::default()
            });
            let mask = self.push_var(ScheduleVar {
                op: Op::Literal,
                ty: VarType::Bool,
                data: DataIdx::Literal(1),
                ..Default::default()
            });

            self.push_var(ScheduleVar {
                op: Op::Scatter { op: ReduceOp::None },
                deps: smallvec![
                    sv_id, // src
                    dst, idx, mask,
                ],
                ty: var.ty.clone(),
                ..Default::default()
            });
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

        let mut sv = ScheduleVar {
            op: var.op,
            ty: var.ty.clone(),
            deps: smallvec![],
            ..Default::default()
        };

        // Collect dependencies

        match var.op {
            Op::Data => {
                let bv = self.collect_data(env, ir, id);
                // Fake gather
                sv.op = Op::Gather;
                let idx = if var.size > 1 {
                    self.push_var(ScheduleVar {
                        op: Op::Idx,
                        ty: VarType::U32,
                        ..Default::default()
                    })
                } else {
                    self.push_var(ScheduleVar {
                        op: Op::Literal,
                        ty: VarType::U32,
                        data: DataIdx::Literal(0),
                        ..Default::default()
                    })
                };
                let mask = self.push_var(ScheduleVar {
                    op: Op::Literal,
                    ty: VarType::Bool,
                    data: DataIdx::Literal(1),
                    ..Default::default()
                });
                sv.deps = smallvec![
                    bv, idx,  // index
                    mask, // mask
                ];
                sv.data = env.push_buffer(var.data.buffer().unwrap());
            }
            Op::Literal => {
                // TODO: cannot evaluate a literal (maybe neccesarry for tensors)
                // sv.param_offset = self.push_param(var.literal);
                if var.opaque {
                    sv.data = env.push_opaque(var.data.literal().unwrap());
                } else {
                    sv.data = DataIdx::Literal(var.data.literal().unwrap());
                }
            }
            Op::Gather => {
                sv.deps = smallvec![
                    self.collect_data(env, ir, var.deps[0]),
                    self.collect(env, ir, var.deps[1]), // index
                    self.collect(env, ir, var.deps[2])  // mask
                ];
            }
            Op::Scatter { .. } => {
                sv.deps = smallvec![
                    self.collect(env, ir, var.deps[0]), // src
                    self.collect_data(env, ir, var.deps[1]),
                    self.collect(env, ir, var.deps[2]), // index
                    self.collect(env, ir, var.deps[3])  // mask
                ];
            }
            Op::TexLookup { dim } => {
                sv.deps = smallvec![self.collect_data(env, ir, var.deps[0]),];
                sv.deps.extend(
                    var.deps[1..(dim as usize + 1)]
                        .iter()
                        .map(|dep| self.collect(env, ir, *dep)),
                );
            }
            Op::TraceRay { payload_count } => {
                self.n_payloads = self.n_payloads.max(payload_count);
                sv.deps = smallvec![self.collect_data(env, ir, var.deps[0])];
                sv.deps.extend(
                    var.deps[1..(16 + payload_count)]
                        .iter()
                        .map(|dep| self.collect(env, ir, *dep)),
                );
            }
            Op::Loop {} => {
                todo!()
            }
            _ => {
                sv.deps = var
                    .deps
                    .iter()
                    .map(|id| self.collect(env, ir, *id))
                    .collect::<SmallVec<[_; 4]>>();
            }
        }

        let svid = self.push_var(sv);

        self.visited.insert(id, svid);

        svid
    }
    ///
    /// Collect variable only as data input/output (for example when it is src/dst for a
    /// gather/scatter operation).
    ///
    /// This only inserts this variable but not its dependencies.
    ///
    pub fn collect_data(&mut self, env: &mut Env, ir: &Internal, id: VarId) -> SVarId {
        let var = ir.var(id);
        if let Some(id) = self.visited.get(&id).cloned() {
            // In case this variable has already been traversed, just ensure that the buffer is
            // added as a parameter.
            // let sv = self.var(id);
            if self.var(id).data.is_none() {
                self.var_mut(id).data = match &var.data {
                    var::Data::None => DataIdx::None,
                    var::Data::Literal(_) => DataIdx::None,
                    var::Data::Buffer(buf) => env.push_buffer(&buf),
                    var::Data::Texture(tex) => env.push_texture(&tex),
                    var::Data::Accel(accel) => env.push_accel(&accel),
                };
            }
            id
        } else {
            let data = match &var.data {
                var::Data::None => DataIdx::None,
                var::Data::Literal(_) => DataIdx::None,
                var::Data::Buffer(buf) => env.push_buffer(&buf),
                var::Data::Texture(tex) => env.push_texture(&tex),
                var::Data::Accel(accel) => env.push_accel(&accel),
            };

            let sbt_hash = var.data.accel().map(|accel| accel.sbt_hash()).unwrap_or(0);
            let svid = self.push_var(ScheduleVar {
                op: Op::Data,
                ty: var.ty.clone(),
                data,
                sbt_hash,
                ..Default::default()
            });
            self.visited.insert(id, svid);
            svid
        }
    }
    pub fn internal_hash(&self) -> u128 {
        let mut hasher = fasthash::murmur3::Hasher128_x64::default();
        self.vars.hash(&mut hasher);

        hasher.finish() as _
    }
}

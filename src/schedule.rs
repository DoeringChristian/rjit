use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use smallvec::{smallvec, SmallVec};

use crate::backend::{Accel, Buffer, Texture};
use crate::trace::Internal;
use crate::var::{Data, Op, ParamType, VarId, VarType};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SVarId(pub usize);
impl std::fmt::Display for SVarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

///
/// A Representation of a variable used in the ScheduleIr.
/// This only holds data that is needed to compile the Kernel.
///
/// Variables are densly stored in the ScheduleIr, simplifying the compilation.
///
#[derive(Debug, Default, Hash)]
pub struct ScheduleVar {
    pub op: Op,
    pub deps: SmallVec<[SVarId; 4]>,
    pub ty: VarType,
    pub param_ty: ParamType,
    pub reg: usize,
    pub buf: Option<usize>, // Index into literal/buffer/texture vec
    pub tex: Option<usize>,
    pub accel: Option<usize>,
    pub opaque: Option<usize>,
    pub literal: u64,
    pub size: usize,

    // We have to build a new kernel when we get new hit/miss shaders.
    pub sbt_hash: u64,
}

impl ScheduleVar {
    pub fn is_literal(&self) -> bool {
        self.op == Op::Literal
    }
}

impl ScheduleVar {
    // ///
    // /// Returns a helper struct (Reg) that can be displayed with the correct prefix for the
    // /// variable type.
    // ///
    // pub fn reg(&self) -> Reg {
    //     Reg(self)
    // }
    // ///
    // /// Returns the raw index of the register for this variable.
    // ///
    // pub fn reg_idx(&self) -> usize {
    //     self.reg
    // }
}

#[derive(Debug, Default)]
pub struct Env {
    opaques: Vec<u64>,
    buffers: Vec<Arc<dyn Buffer>>,
    textures: Vec<Arc<dyn Texture>>,
    accels: Vec<Arc<dyn Accel>>,
}

impl Env {
    fn push_opaque(&mut self, literal: u64) -> usize {
        let idx = self.opaques.len();
        self.opaques.push(literal);
        idx
    }
    pub fn push_buffer(&mut self, buf: &Arc<dyn Buffer>) -> usize {
        let idx = self.buffers.len();
        self.buffers.push(buf.clone());
        idx
    }
    fn push_texture(&mut self, tex: &Arc<dyn Texture>) -> usize {
        let idx = self.buffers.len();
        self.textures.push(tex.clone());
        idx
    }
    fn push_accel(&mut self, accel: &Arc<dyn Accel>) -> usize {
        let idx = self.buffers.len();
        self.accels.push(accel.clone());
        idx
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
    fn push_var(&mut self, var: ScheduleVar) -> SVarId {
        let id = SVarId(self.vars.len());
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
            let param = env.push_buffer(var.data.buffer().unwrap());

            let mut sv = self.var_mut(sv_id);

            sv.param_ty = ParamType::Output;
            sv.buf = Some(param);
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
            reg: self.next_reg(),
            param_ty: ParamType::None,
            buf: None,
            tex: None,
            accel: None,
            literal: 0,
            opaque: None,
            size: var.size,
            sbt_hash: 0,
        };

        // Collect dependencies

        match var.op {
            Op::Data => {
                sv.buf = Some(env.push_buffer(var.data.buffer().unwrap()));
                sv.param_ty = ParamType::Input;
            }
            Op::Literal => {
                // TODO: cannot evaluate a literal (maybe neccesarry for tensors)
                // sv.param_offset = self.push_param(var.literal);
                if var.opaque {
                    sv.opaque = Some(env.push_opaque(var.data.literal().unwrap()));
                } else {
                    sv.literal = var.data.literal().unwrap();
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
            match &var.data {
                Data::Buffer(buf) => {
                    let buf = Some(env.push_buffer(buf));
                    self.var_mut(id).buf = buf;
                }
                Data::Texture(tex) => {
                    let tex = Some(env.push_texture(tex));
                    self.var_mut(id).tex = tex;
                }
                _ => {}
            }
            id
        } else {
            let reg = self.next_reg();
            let buf = var.data.buffer().map(|buf| env.push_buffer(&buf));
            let tex = var.data.texture().map(|tex| env.push_texture(&tex));
            let accel = var.data.accel().map(|accel| env.push_accel(&accel));

            let sbt_hash = var.data.accel().map(|accel| accel.sbt_hash()).unwrap_or(0);
            let svid = self.push_var(ScheduleVar {
                op: Op::Data,
                ty: var.ty.clone(),
                reg,
                buf,
                tex,
                accel,
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

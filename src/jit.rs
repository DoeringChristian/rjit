use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use cust::util::SliceExt;

use crate::compiler::CUDACompiler;
use crate::ir::{Ir, Op, ParamType, VarId};

#[derive(Clone, Debug)]
pub struct ScheduledGroup {
    pub range: Range<usize>,
    pub size: usize, // size of the varibles in group
}
#[derive(Debug, Clone)]
pub struct ScheduledVar {
    pub size: usize,
    pub id: VarId,
}

// TODO: pooling for paralel exectution
#[derive(Default)]
pub struct Jit {
    schedule: Vec<ScheduledVar>,
    pub params: Vec<u64>, // Kernel parameters used to transfer information to the kernel
    pub n_regs: usize,
    pub compiler: CUDACompiler,
    pub visited: HashSet<(usize, VarId)>,
}

impl Jit {
    pub fn eval(&mut self, ir: &mut Ir) {
        self.params.clear();
        // self.n_regs = self.compiler.first_register();
        self.schedule.clear();

        // Sort scheduled variables by size
        for id in ir.scheduled.iter() {
            let var = ir.var(*id);
            self.var_traverse(ir, *id, var.size);
        }
        self.schedule.sort_by(|sv0, sv1| sv0.size.cmp(&sv1.size));

        dbg!(&self.schedule);

        // Group variables of same size
        let mut groups = vec![];
        let cur = 0;
        for i in 1..self.schedule.len() {
            // let var0 = ir.var(self.schedule[i - 1].id);
            // let var1 = ir.var(self.schedule[i].id);
            let sv0 = &self.schedule[i - 1];
            let sv1 = &self.schedule[i];
            if sv0.size != sv1.size {
                groups.push(ScheduledGroup {
                    range: cur..i,
                    size: sv0.size,
                });
            }
        }
        groups.push(ScheduledGroup {
            range: cur..self.schedule.len(),
            size: self.schedule.last().unwrap().size,
        });

        for id in ir.scheduled.clone() {
            ir.var_mut(id).param_ty = ParamType::Output;
        }
        dbg!(&groups);

        // TODO: this can be paralelized (rayon)
        for group in groups {
            // Reset parameters for group.
            self.params.clear();
            self.params.push(group.size as _);
            self.n_regs = self.compiler.first_register();

            dbg!(&self.params);
            self.preprocess(ir, &group);
            self.compiler
                .assemble(ir, &group, &self.schedule, self.params.len(), self.n_regs);
            self.compiler.execute(ir, group.size, &mut self.params);
        }
    }
    fn var_traverse(&mut self, ir: &Ir, id: VarId, size: usize) {
        if !self.visited.insert((size, id)) {
            return;
        }
        let var = ir.var(id);
        for id in var.deps.iter() {
            self.var_traverse(ir, *id, size);
        }
        self.schedule.push(ScheduledVar { size, id });
    }
    ///
    /// Assembles a group of varaibles.
    /// This writes into the variables of the internal representation.
    ///
    fn preprocess(&mut self, ir: &mut Ir, group: &ScheduledGroup) {
        // TODO: Implement diffrent backends
        for schdule_idx in group.range.clone() {
            let sv = &self.schedule[schdule_idx];
            let var = ir.var_mut(sv.id);

            // assert_eq!(var.size, group.size);

            var.reg = self.n_regs;
            self.n_regs += 1;

            if var.param_ty == ParamType::Input {
                // TODO: This should be compatible with diffrent backends
                let offset = self.push_param(var.buffer.as_ref().unwrap().as_device_ptr().as_raw());
                var.param_offset = offset;
            } else if var.param_ty == ParamType::Output {
                var.buffer = Some(Box::new(
                    vec![0u8; var.size * var.ty.size()]
                        .as_slice()
                        .as_dbuf()
                        .unwrap(),
                ));
                let offset = self.push_param(var.buffer.as_ref().unwrap().as_device_ptr().as_raw());
                var.param_offset = offset;
            } else {
            }
        }
    }
    fn push_param(&mut self, param: u64) -> usize {
        let idx = self.params.len();
        self.params.push(param);
        idx
    }
}

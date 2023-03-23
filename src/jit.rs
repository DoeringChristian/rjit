use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use cust::util::SliceExt;

use crate::compiler::CUDACompiler;
use crate::schedule::ScheduleIr;
use crate::trace::{Ir, Op, ParamType, VarId};

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
    pub compiler: CUDACompiler,
}

impl Jit {
    pub fn eval(&mut self, ir: &mut Ir) {
        let mut scheduled = ir.scheduled.clone();
        scheduled.sort_by(|id0, id1| ir.var(*id0).size.cmp(&ir.var(*id1).size));

        for id in scheduled.iter() {
            ir.var_mut(*id).param_ty = ParamType::Output;
        }

        let cur = 0;
        let mut size = 0;
        let mut schedules = vec![];
        for i in 1..scheduled.len() {
            let var0 = ir.var(scheduled[i - 1]);
            let var1 = ir.var(scheduled[i]);
            size = var0.size;
            if var0.size != var1.size {
                let mut tmp = ScheduleIr::new(&self.compiler, size);
                tmp.collect_vars(ir, &scheduled[cur..i]);
                schedules.push(tmp);
            }
        }
        size = ir.var(*scheduled.last().unwrap()).size;
        let mut tmp = ScheduleIr::new(&self.compiler, size);

        tmp.collect_vars(ir, &scheduled[cur..scheduled.len()]);
        schedules.push(tmp);
        dbg!(&schedules);

        // TODO: this can be paralelized (rayon)
        for schedule in schedules.iter_mut() {
            let mut compiler = CUDACompiler::default();
            compiler.assemble(schedule);
            compiler.execute(schedule);
        }
    }
}

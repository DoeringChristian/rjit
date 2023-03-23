use std::sync::Arc;

use crate::backend::{Backend, Kernel};
use crate::schedule::ScheduleIr;
use crate::trace::{Ir, ParamType};

// TODO: pooling for paralel exectution
pub struct Jit {
    pub backend: Arc<dyn Backend>,
}

impl Jit {
    pub fn new(backend: &Arc<dyn Backend>) -> Self {
        Self {
            backend: backend.clone(),
        }
    }
    pub fn compile(&mut self, ir: &mut Ir) -> Vec<(ScheduleIr, Box<dyn Kernel>)> {
        let mut scheduled = ir.scheduled.clone();
        scheduled.sort_by(|id0, id1| ir.var(*id0).size.cmp(&ir.var(*id1).size));

        for id in scheduled.iter() {
            ir.var_mut(*id).param_ty = ParamType::Output;
        }

        let cur = 0;
        let mut size;
        let mut schedules = vec![];
        for i in 1..scheduled.len() {
            let var0 = ir.var(scheduled[i - 1]);
            let var1 = ir.var(scheduled[i]);
            size = var0.size;
            if var0.size != var1.size {
                let mut tmp = ScheduleIr::new(&self.backend, self.backend.first_register(), size);
                tmp.collect_vars(ir, &scheduled[cur..i]);
                schedules.push(tmp);
            }
        }
        size = ir.var(*scheduled.last().unwrap()).size;
        let mut tmp = ScheduleIr::new(&self.backend, self.backend.first_register(), size);

        tmp.collect_vars(ir, &scheduled[cur..scheduled.len()]);
        schedules.push(tmp);
        // dbg!(&schedules);

        // TODO: this can be paralelized (rayon)
        let kernels = schedules
            .into_iter()
            .map(|mut s| {
                let mut kernel = self.backend.new_kernel();
                kernel.assemble(&mut s);
                kernel.compile();
                (s, kernel)
            })
            .collect::<Vec<_>>();
        kernels
    }
    pub fn eval(&mut self, ir: &mut Ir) {
        let kernels = self.compile(ir);
        for (mut s, mut kernel) in kernels {
            kernel.execute(&mut s);
        }
    }
}

#[cfg(test)]
mod test {}

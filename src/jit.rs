use std::sync::Arc;

use crate::backend::{Backend, Kernel};
use crate::schedule::ScheduleIr;
use crate::trace::{Ir, ParamType};

// TODO: pooling for paralel exectution
pub struct Jit {
    pub backend: Arc<dyn Backend>,
    pub schedules: Vec<ScheduleIr>,
    pub kernels: Vec<Box<dyn Kernel>>,
}

impl Jit {
    pub fn new(backend: &Arc<dyn Backend>) -> Self {
        Self {
            backend: backend.clone(),
            schedules: vec![],
            kernels: vec![],
        }
    }
    pub fn compile(&mut self, ir: &mut Ir) {
        self.schedules.clear();
        self.kernels.clear();
        let mut scheduled = ir.scheduled.clone();
        scheduled.sort_by(|id0, id1| ir.var(*id0).size.cmp(&ir.var(*id1).size));

        for id in scheduled.iter() {
            ir.var_mut(*id).param_ty = ParamType::Output;
        }

        let cur = 0;
        let mut size;
        for i in 1..scheduled.len() {
            let var0 = ir.var(scheduled[i - 1]);
            let var1 = ir.var(scheduled[i]);
            size = var0.size;
            if var0.size != var1.size {
                let mut tmp = ScheduleIr::new(&self.backend, self.backend.first_register(), size);
                tmp.collect_vars(ir, &scheduled[cur..i]);
                self.schedules.push(tmp);
            }
        }
        size = ir.var(*scheduled.last().unwrap()).size;
        let mut tmp = ScheduleIr::new(&self.backend, self.backend.first_register(), size);

        tmp.collect_vars(ir, &scheduled[cur..scheduled.len()]);
        self.schedules.push(tmp);
        // dbg!(&schedules);

        // TODO: this can be paralelized (rayon)
        self.kernels = self
            .schedules
            .iter_mut()
            .map(|mut s| {
                let mut kernel = self.backend.new_kernel();
                kernel.assemble(&mut s);
                kernel.compile();
                kernel
            })
            .collect::<Vec<_>>();
    }
    pub fn eval(&mut self, ir: &mut Ir) {
        self.compile(ir);
        for i in 0..self.kernels.len() {
            self.kernels[i].execute(&mut self.schedules[i]);
        }
    }
}

#[cfg(test)]
mod test {}

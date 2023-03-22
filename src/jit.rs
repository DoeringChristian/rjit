use std::ops::Range;
use std::sync::Arc;

use cust::util::SliceExt;

use crate::compiler::CUDACompiler;
use crate::ir::{Ir, Op, ParamType, VarId};

#[derive(Clone)]
pub struct Group {
    pub range: Range<usize>,
    pub num: usize, // size of the varibles in group
}

pub struct Jit {
    schedule: Vec<VarId>,
    pub params: Vec<u64>,
    pub n_regs: usize,
    pub compiler: CUDACompiler,
}

impl Jit {
    pub fn eval(&mut self, ir: &mut Ir) {
        self.params.clear();
        self.n_regs = self.compiler.first_register();
        self.schedule.clear();

        // Sort scheduled variables by size
        self.schedule = ir.deps(&ir.scheduled).collect::<Vec<_>>();
        self.schedule
            .sort_by(|id0, id1| ir.var(*id0).num.cmp(&ir.var(*id1).num));

        // Group variables of same size
        let groups = vec![];
        let cur = 0;
        for i in 1..self.schedule.len() {
            let var0 = ir.var(self.schedule[i - 1]);
            let var1 = ir.var(self.schedule[i]);
            if var0.num != var1.num {
                groups.push(Group {
                    range: cur..i,
                    num: var0.num,
                });
            }
        }

        // TODO: this can be paralelized
        for group in groups {
            // Assemble a group
            self.preprocess(ir, group);
        }
    }
    ///
    /// Assembles a group of varaibles.
    /// This writes into the variables of the internal representation.
    ///
    fn preprocess(&mut self, ir: &mut Ir, group: Group) {
        // TODO: Implement diffrent backends
        for schdule_idx in group.range {
            let id = self.schedule[schdule_idx];
            let var = ir.var_mut(id);

            assert_eq!(var.num, group.num);

            var.reg = self.n_regs;
            self.n_regs += 1;

            if var.param_ty == ParamType::Input {
                // TODO: This should be compatible with diffrent backends
                let offset = self.push_param(var.buffer.unwrap().as_device_ptr().as_raw());
                var.param_offset = offset;
            } else if var.param_ty == ParamType::Output {
                var.buffer = Some(Box::new(
                    vec![0u8; var.num * var.ty.size()]
                        .as_slice()
                        .as_dbuf()
                        .unwrap(),
                ));
                let offset = self.push_param(var.buffer.unwrap().as_device_ptr().as_raw());
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

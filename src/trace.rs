use std::sync::Arc;

use crate::ir;

#[derive(Clone)]
pub enum Var {
    Buffer,
    Intermediate { op: Op },
}

#[derive(Clone)]
pub enum Op {
    Neg(Arc<Var>),
    Not(Arc<Var>),
    Sqrt(Arc<Var>),
    Abs(Arc<Var>),
    Add(Arc<Var>, Arc<Var>), // Add two variables
    Sub(Arc<Var>, Arc<Var>),
    Mul(Arc<Var>, Arc<Var>),
    Div(Arc<Var>, Arc<Var>),
    Mod(Arc<Var>, Arc<Var>),
    Mulhi(Arc<Var>, Arc<Var>),
    Fma(Arc<Var>, Arc<Var>, Arc<Var>),
    Min(Arc<Var>, Arc<Var>),
    Max(Arc<Var>, Arc<Var>),
    Cail(Arc<Var>),
    Floor(Arc<Var>),
    Round(Arc<Var>),
    Trunc(Arc<Var>),
    Eq(Arc<Var>, Arc<Var>),
    Neq(Arc<Var>, Arc<Var>),
    Lt(Arc<Var>, Arc<Var>),
    Le(Arc<Var>, Arc<Var>),
    Gt(Arc<Var>, Arc<Var>),
    Ge(Arc<Var>, Arc<Var>),
    Select(Arc<Var>, Arc<Var>, Arc<Var>),
    Popc(Arc<Var>),
    Clz(Arc<Var>),
    Ctz(Arc<Var>),
    And(Arc<Var>, Arc<Var>),
    Or(Arc<Var>, Arc<Var>),
    Xor(Arc<Var>, Arc<Var>),
    Shl(Arc<Var>, Arc<Var>),
    Shr(Arc<Var>, Arc<Var>),
    Rcp(Arc<Var>, Arc<Var>),
    Rsqrt(Arc<Var>, Arc<Var>),
    Sin(Arc<Var>, Arc<Var>),
    Cos(Arc<Var>, Arc<Var>),
    Exp2(Arc<Var>, Arc<Var>),
    Log2(Arc<Var>, Arc<Var>),
    Cast(Arc<Var>),
    Bitcast(Arc<Var>),
    Gather {
        from: Arc<Var>,
        idx: Arc<Var>,
        mask: Arc<Var>,
    },
    Scatter {
        from: Arc<Var>,
        to: Arc<Var>,
        idx: Arc<Var>,
        mask: Arc<Var>,
    },
    Idx,
    ConstF32(f32),         // Set a constant value
    Load(Arc<Var>),        // Load from buffer with pointer in params at offset
    LoadLiteral(Arc<Var>), // Load from params at offset
}

pub fn generate_ir(vars: &[&Var], ir: &mut ir::Ir) {
    for var in vars {
        match *var {
            Var::Buffer => {
                todo!()
            }
            Var::Intermediate { op } => match op {
                _ => todo!(),
            },
        }
    }
}

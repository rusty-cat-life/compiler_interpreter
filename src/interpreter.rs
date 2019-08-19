use super::common::*;
use std::collections::HashMap;

pub struct Interpreter(HashMap<String, i64>);

impl Interpreter {
    pub fn new() -> Self {
        Interpreter(HashMap::new())
    }

    pub fn eval(&mut self, expr: &Ast) -> Result<i64, InterpreterError> {
        use super::common::AstKind::*;

        match expr.value {
            Num(n) => Ok(n as i64),
            Boolean(_) => unimplemented!(),
            UniOp { ref op, ref e } => {
                let e = self.eval(e)?;
                Ok(self.eval_uniop(op, e))
            }
            BinOp {
                ref op,
                ref l,
                ref r,
            } => {
                let l = self.eval(l)?;
                let r = self.eval(r)?;
                self.eval_binop(op, l, r)
                    .map_err(|e| InterpreterError::new(e, expr.loc.clone()))
            }
            EqOp {
                ref op,
                ref l,
                ref r,
            } => unimplemented!(),
            RelOp {
                ref op,
                ref l,
                ref r,
            } => unimplemented!(),
            Int { ref var, ref body } => {
                let e = self.eval(body)?;
                self.0.insert(var.clone(), e);
                Ok(0)
            }
            Var(ref s) => self.0.get(s).cloned().ok_or(InterpreterError::new(
                InterpreterErrorKind::UnboundVariable(s.clone()),
                expr.loc.clone(),
            )),
            Char { ref var, ref body } => unimplemented!(),
            CharLiteral(ref c) => unimplemented!(),
            
        }
    }

    fn eval_uniop(&mut self, op: &UniOp, n: i64) -> i64 {
        use super::common::UniOpKind::*;

        match op.value {
            Plus => n,
            Minus => -n,
        }
    }

    fn eval_binop(&mut self, op: &BinOp, l: i64, r: i64) -> Result<i64, InterpreterErrorKind> {
        use super::common::BinOpKind::*;
        match op.value {
            Add => Ok(l + r),
            Sub => Ok(l - r),
            Mult => Ok(l * r),
            Div => {
                if r == 0 {
                    Err(InterpreterErrorKind::DivisionByZero)
                } else {
                    Ok(l / r)
                }
            }
        }
    }
}

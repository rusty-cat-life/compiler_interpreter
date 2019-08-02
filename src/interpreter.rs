pub struct Interpreter;

impl Interpreter {
    pub fn new() -> Self {
        Interpreter
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum InterpreterErrorKind {
    DivisionByZero
}
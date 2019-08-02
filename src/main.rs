mod common;
mod interpreter;
mod parser;

use common::Ast;
use interpreter::Interpreter;
use std::io;

/// プロンプトを表示しユーザの入力を促す
fn prompt(s: &str) -> io::Result<()> {
    use std::io::{stdout, Write};
    let stdout = stdout();
    let mut stdout = stdout.lock();
    stdout.write(s.as_bytes())?;
    stdout.flush()
}

fn main() {
    use std::io::{stdin, BufRead, BufReader};

    let mut interp = Interpreter::new();

    let stdin = stdin();
    let stdin = stdin.lock();
    let stdin = BufReader::new(stdin);
    let mut lines = stdin.lines();
    loop {
        prompt("> ").unwrap();

        if let Some(Ok(line)) = lines.next() {
            let ast = match line.parse::<Ast>() {
                Ok(ast) => ast,
                Err(e) => {
                    eprintln!("{}", e);
                    e.show_diagnostic(&line);
                    continue;
                }
            };
            let n = match interp.eval(&ast) {
                Ok(n) => n,
                Err(e) => {
                    // eprintln!("{}", e);
                    // e.show_diagnostic(&line);
                    continue;
                }
            };

            println!("{}", n);
        } else {
            break;
        }
    }
}

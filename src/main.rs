use parser_2::{Ast};
use std::io;

/// プロンプトを表示しユーザの入力を促す
fn prompt(s: &str) -> io::Result<()> {
    use std::io::{stdout, Write};
    let stdout = stdout();
    let mut stdout = stdout.lock();
    stdout.write(s.as_bytes())?;
    stdout.flush()
}

fn show_trace<E: std::error::Error>(e: E) {
    eprintln!("{}", e);
    let mut source = e.source();

    while let Some(e) = source {
        eprintln!("caused by {}", e);
        source = e.source()
    }
}

fn main() {
    use std::io::{stdin, BufRead, BufReader};

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
                    // e.show_diagnostic(&line);
                    show_trace(e);
                    continue;
                },
            };
            println!("{:?}", ast);
        } else {
            break;
        }
    }
}

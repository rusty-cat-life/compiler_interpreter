use parser_2::{lex, parse};
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

    let stdin = stdin();
    let stdin = stdin.lock();
    let stdin = BufReader::new(stdin);
    let mut lines = stdin.lines();
    loop {
        prompt("> ").unwrap();

        if let Some(Ok(line)) = lines.next() {
            let tokens = lex(&line).unwrap();
            let ast = parse(tokens).unwrap();
            println!("{:?}", ast);
        } else {
            break;
        }
    }
}

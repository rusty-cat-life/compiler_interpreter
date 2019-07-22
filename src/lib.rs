use std::iter::Peekable;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Loc(usize, usize);

impl Loc {
    fn merge(&self, other: &Loc) -> Loc {
        use std::cmp::{max, min};
        Loc(min(self.0, other.0), max(self.1, other.1))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Annot<T> {
    value: T,
    loc: Loc,
}

impl<T> Annot<T> {
    fn new(value: T, loc: Loc) -> Self {
        Self { value, loc }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    /// [0-9][0-9]*
    Number(u64),
    /// +
    Plus,
    /// -
    Minus,
    /// *
    Asterisk,
    /// /
    Slash,
    /// ()
    LParen,
    /// )
    RParen,
}

pub type Token = Annot<TokenKind>;

impl Token {
    fn number(n: u64, loc: Loc) -> Self {
        Self::new(TokenKind::Number(n), loc)
    }

    fn plus(loc: Loc) -> Self {
        Self::new(TokenKind::Plus, loc)
    }

    fn minus(loc: Loc) -> Self {
        Self::new(TokenKind::Minus, loc)
    }

    fn asterisk(loc: Loc) -> Self {
        Self::new(TokenKind::Asterisk, loc)
    }

    fn slash(loc: Loc) -> Self {
        Self::new(TokenKind::Slash, loc)
    }

    fn lparen(loc: Loc) -> Self {
        Self::new(TokenKind::LParen, loc)
    }

    fn rparen(loc: Loc) -> Self {
        Self::new(TokenKind::RParen, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LexErrorKind {
    InvalidChar(char),
    Eof,
}

pub type LexError = Annot<LexErrorKind>;

impl LexError {
    fn invalid_char(c: char, loc: Loc) -> Self {
        LexError::new(LexErrorKind::InvalidChar(c), loc)
    }

    fn eof(loc: Loc) -> Self {
        LexError::new(LexErrorKind::Eof, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AstKind {
    /// 数値
    Num(u64),
    /// 単項演算
    UniOp { op: UniOp, e: Box<Ast> },
    /// 二項演算
    BinOp { op: BinOp, l: Box<Ast>, r: Box<Ast> },
}

type Ast = Annot<AstKind>;

impl Ast {
    fn num(n: u64, loc: Loc) -> Self {
        // impl<T> Annot<T>で実装したnewを呼ぶ
        Self::new(AstKind::Num(n), loc)
    }

    fn uniop(op: UniOp, e: Ast, loc: Loc) -> Self {
        Self::new(AstKind::UniOp { op, e: Box::new(e) }, loc)
    }

    fn binop(op: BinOp, l: Ast, r: Ast, loc: Loc) -> Self {
        Self::new(
            AstKind::BinOp {
                op,
                l: Box::new(l),
                r: Box::new(r),
            },
            loc,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum UniOpKind {
    /// 正号
    Plus,
    /// 負号
    Minus,
}

type UniOp = Annot<UniOpKind>;

impl UniOp {
    fn plus(loc: Loc) -> Self {
        Self::new(UniOpKind::Plus, loc)
    }

    fn minus(loc: Loc) -> Self {
        Self::new(UniOpKind::Minus, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum BinOpKind {
    /// 加算
    Add,
    /// 減算
    Sub,
    /// 乗算
    Mult,
    /// 除算
    Div,
}

type BinOp = Annot<BinOpKind>;

impl BinOp {
    fn add(loc: Loc) -> Self {
        Self::new(BinOpKind::Add, loc)
    }

    fn sub(loc: Loc) -> Self {
        Self::new(BinOpKind::Sub, loc)
    }

    fn mult(loc: Loc) -> Self {
        Self::new(BinOpKind::Mult, loc)
    }

    fn div(loc: Loc) -> Self {
        Self::new(BinOpKind::Div, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ParseError {
    /// 予期しないトークンが来た
    UnexpectedToken(Token),
    /// 式を期待していたのに式でないものが来た
    NotExpression(Token),
    /// 演算子を期待していたのに演算子でないものが来た
    NotOperator(Token),
    /// 括弧が閉じられていない
    UnclosedOpenParen(Token),
    /// 式の解析が終わったのにまだトークンが残っている
    RedundantExpression(Token),
    /// パース途中で入力が終わった
    Eof,
}

pub fn lex(input: &str) -> Result<Vec<Token>, LexError> {
    let mut result = vec![];

    let input = input.as_bytes();

    // 位置を管理する値
    let mut position = 0;

    macro_rules! lex_a_token {
        ($lexer:expr) => {{
            let (token, pos) = $lexer?;
            result.push(token);
            position = pos;
        }};
    }

    while position < input.len() {
        match input[position] {
            b'0'...b'9' => lex_a_token!(lex_number(input, position)),
            b'+' => lex_a_token!(lex_plus(input, position)),
            b'-' => lex_a_token!(lex_minus(input, position)),
            b'*' => lex_a_token!(lex_asterisk(input, position)),
            b'/' => lex_a_token!(lex_slash(input, position)),
            b'(' => lex_a_token!(lex_lparen(input, position)),
            b')' => lex_a_token!(lex_rparen(input, position)),
            b' ' | b'\n' | b'\t' => {
                let (_, pos) = skip_spaces(input, position)?;
                position = pos;
            }
            b => {
                return Err(LexError::invalid_char(
                    b as char,
                    Loc(position, position + 1),
                ))
            }
        }
    }

    Ok(result)
}

fn consume_byte(input: &[u8], pos: usize, b: u8) -> Result<(u8, usize), LexError> {
    // posが入力サイズ以上なら入力が終わっている
    // 1バイト期待しているのに終わっているのでエラー
    if input.len() <= pos {
        return Err(LexError::eof(Loc(pos, pos)));
    }

    // 入力が期待するものでなければエラー
    if input[pos] != b {
        return Err(LexError::invalid_char(
            input[pos] as char,
            Loc(pos, pos + 1),
        ));
    }

    Ok((b, pos + 1))
}

fn lex_plus(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'+').map(|(_, end)| (Token::plus(Loc(start, end)), end))
}

fn lex_minus(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'-').map(|(_, end)| (Token::minus(Loc(start, end)), end))
}

fn lex_asterisk(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'*').map(|(_, end)| (Token::asterisk(Loc(start, end)), end))
}

fn lex_slash(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'/').map(|(_, end)| (Token::slash(Loc(start, end)), end))
}

fn lex_lparen(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'(').map(|(_, end)| (Token::lparen(Loc(start, end)), end))
}

fn lex_rparen(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b')').map(|(_, end)| (Token::rparen(Loc(start, end)), end))
}

fn recognize_many(input: &[u8], mut pos: usize, mut f: impl FnMut(u8) -> bool) -> usize {
    while pos < input.len() && f(input[pos]) {
        pos += 1;
    }

    pos
}

fn lex_number(input: &[u8], pos: usize) -> Result<(Token, usize), LexError> {
    use std::str::from_utf8;

    let start = pos;
    let end = recognize_many(input, start, |b| b"1234567890".contains(&b));

    let num = from_utf8(&input[start..end]).unwrap().parse().unwrap();

    Ok((Token::number(num, Loc(start, end)), end))
}

fn skip_spaces(input: &[u8], pos: usize) -> Result<((), usize), LexError> {
    let pos = recognize_many(input, pos, |b| b" \n\t".contains(&b));

    Ok(((), pos))
}

#[test]
fn test_lexer() {
    assert_eq!(
        lex("1 + 2 * 3 - -10"),
        Ok(vec![
            Token::number(1, Loc(0, 1)),
            Token::plus(Loc(2, 3)),
            Token::number(2, Loc(4, 5)),
            Token::asterisk(Loc(6, 7)),
            Token::number(3, Loc(8, 9)),
            Token::minus(Loc(10, 11)),
            Token::minus(Loc(12, 13)),
            Token::number(10, Loc(13, 15)),
        ])
    )
}

fn parse(tokens: Vec<Token>) -> Result<Ast, ParseError> {
    // 入力をイテレータにし、Peekableにする
    let mut tokens = tokens.into_iter().peekable();
    // その後parse_exprを呼んでエラー処理をする
    let result = parse_expr(&mut tokens)?;
    match tokens.next() {
        Some(token) => Err(ParseError::RedundantExpression(token)),
        None => Ok(result),
    }
}

fn parse_expr<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError> 
    where Tokens: Iterator<Item = Token>,
    {
        parse_expr3(tokens)
    }

fn parse_expr3<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError> 
    where Tokens: Iterator<Item = Token>,
    {
        // EXPR2をパースする
        let mut e = parse_expr2(tokens)?;
        // EXPE3_Loop
        loop {
            match tokens.peek().map(|token| token.value) {
                // ("+" | "-")
                Some(TokenKind::Plus) | Some(TokenKind::Minus) => {
                    let operator = match tokens.next().unwrap() {
                        Token {
                            value: TokenKind::Plus,
                            loc,
                        } => BinOp::add(loc),
                        Token {
                            value: TokenKind::Minus,
                            loc,
                        } => BinOp::sub(loc),
                        // Peekで入力が"+"か"-"であることは確認したのでそれ以外はありえない
                        _ => unreachable!(),
                    };

                    // EXPR2
                    let r = parse_expr2(tokens)?;
                    // 位置情報やAST構築の処理
                    let loc = e.loc.merge(&r.loc);

                    e = Ast::binop(operator, e, r, loc);
                },
                _ => return Ok(e)
            }
        }
    }

fn parse_expr2<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError> 
    where Tokens: Iterator<Item = Token>,
    {
        let mut e = parse_expr1(tokens)?;

        loop {
            match tokens.peek().map(|token| token.value) {
                Some(TokenKind::Asterisk) | Some(TokenKind::Slash) => {
                    let operator = match tokens.next().unwrap() {
                        Token {
                            value: TokenKind::Asterisk,
                            loc,
                        } => BinOp::mult(loc),
                        Token {
                            value: TokenKind::Slash,
                            loc,
                        } => BinOp::div(loc),
                        _ => unreachable!(),
                    };

                    let r = parse_expr1(tokens)?;
                    let loc = e.loc.merge(&r.loc);

                    e = Ast::binop(operator, e , r, loc);
                },
                _ => return Ok(e),
            }
        }
    }

fn parse_expr1<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError> 
    where Tokens: Iterator<Item = Token>,
    {

    }

fn parse_left_binop<Tokens>(
    tokens: & mut Peekable<Tokens>, 
    subexpr_parser: fn(&mut Peekable<Tokens>) -> Result<Ast, ParseError>, 
    op_parser: fn(&mut Peekable<Tokens>) -> Result<BinOp, ParseError>,
    ) -> Result<Ast, ParseError>
    where Tokens: Iterator<Item = Token>, 
    {
        let mut e = subexpr_parser(tokens)?;

        loop {
            match tokens.peek() {
                Some(_) => {
                    let operator = match op_parser(tokens) {
                        Ok(op) => op,
                        // ここでパースに失敗したのはこれ以上中置演算子がないという意味
                        Err(_) => break,
                    };

                    let r = subexpr_parser(tokens)?;
                    let loc = e.loc.merge(&r.loc);
                    e = Ast::binop(operator, e, r, loc)
                },
                _ => break,
            }
        }

        Ok(e)
    }
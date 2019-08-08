use super::parser::{lex, parse};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Loc(pub usize, pub usize);

impl Loc {
    pub fn merge(&self, other: &Loc) -> Loc {
        use std::cmp::{max, min};
        Loc(min(self.0, other.0), max(self.1, other.1))
    }
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.0, self.1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Annot<T> {
    pub value: T,
    pub loc: Loc,
}

impl<T> Annot<T> {
    pub fn new(value: T, loc: Loc) -> Self {
        Self { value, loc }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    /// int type
    Int,
    /// =
    Assign,
    /// ;
    Semicolon,
    /// variable name
    Var(String),
    /// ==
    Equal,
    // !=
    Unequal,
    /// >
    Greater,
    /// >=
    GreaterEqual,
    /// <
    Less,
    /// <=
    LessEqual
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::TokenKind::*;

        match self {
            Number(n) => n.fmt(f),
            Plus => write!(f, "+"),
            Minus => write!(f, "-"),
            Asterisk => write!(f, "*"),
            Slash => write!(f, "/"),
            LParen => write!(f, "("),
            RParen => write!(f, ")"),
            Equal => write!(f, "=="),
            Unequal => write!(f, "!="),
            Greater => write!(f, ">"),
            GreaterEqual => write!(f, ">="),
            Less => write!(f, "<"),
            LessEqual => write!(f, "<="),
            Int => write!(f, "Int"),
            Assign => write!(f, "="),
            Semicolon => write!(f, ";"),
            Var(s) => write!(f, "variable name: {}", s)
        }
    }
}

pub type Token = Annot<TokenKind>;

impl Token {
    pub fn number(n: u64, loc: Loc) -> Self {
        Self::new(TokenKind::Number(n), loc)
    }

    pub fn plus(loc: Loc) -> Self {
        Self::new(TokenKind::Plus, loc)
    }

    pub fn minus(loc: Loc) -> Self {
        Self::new(TokenKind::Minus, loc)
    }

    pub fn asterisk(loc: Loc) -> Self {
        Self::new(TokenKind::Asterisk, loc)
    }

    pub fn slash(loc: Loc) -> Self {
        Self::new(TokenKind::Slash, loc)
    }

    pub fn lparen(loc: Loc) -> Self {
        Self::new(TokenKind::LParen, loc)
    }

    pub fn rparen(loc: Loc) -> Self {
        Self::new(TokenKind::RParen, loc)
    }

    pub fn int(loc: Loc) -> Self {
        Self::new(TokenKind::Int, loc)
    }

    pub fn assign(loc: Loc) -> Self {
        Self::new(TokenKind::Assign, loc)
    }

    pub fn semicolon(loc: Loc) -> Self {
        Self::new(TokenKind::Semicolon, loc)
    }

    pub fn var(s: impl Into<String>, loc: Loc) -> Self {
        Self::new(TokenKind::Var(s.into()), loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LexErrorKind {
    InvalidChar(char),
    Eof,
}

pub type LexError = Annot<LexErrorKind>;

impl LexError {
    pub fn invalid_char(c: char, loc: Loc) -> Self {
        LexError::new(LexErrorKind::InvalidChar(c), loc)
    }

    pub fn eof(loc: Loc) -> Self {
        LexError::new(LexErrorKind::Eof, loc)
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::LexErrorKind::*;

        let loc = &self.loc;

        match self.value {
            InvalidChar(c) => write!(f, "{}: invalid char '{}'", loc, c),
            Eof => write!(f, "End of file"),
        }
    }
}

impl std::error::Error for LexError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AstKind {
    /// 数値
    Num(u64),
    /// 単項演算
    UniOp { op: UniOp, e: Box<Ast> },
    /// 二項演算
    BinOp { op: BinOp, l: Box<Ast>, r: Box<Ast> },
    /// Int
    Int { var: String, body: Box<Ast> },
    /// 変数
    Var(String)
}

pub type Ast = Annot<AstKind>;

impl Ast {
    pub fn num(n: u64, loc: Loc) -> Self {
        // impl<T> Annot<T>で実装したnewを呼ぶ
        Self::new(AstKind::Num(n), loc)
    }

    pub fn uniop(op: UniOp, e: Ast, loc: Loc) -> Self {
        Self::new(AstKind::UniOp { op, e: Box::new(e) }, loc)
    }

    pub fn binop(op: BinOp, l: Ast, r: Ast, loc: Loc) -> Self {
        Self::new(
            AstKind::BinOp {
                op,
                l: Box::new(l),
                r: Box::new(r),
            },
            loc,
        )
    }

    pub fn int(var: String, body: Ast, loc: Loc) -> Self {
        Self::new(AstKind::Int {
            var,
            body: Box::new(body)
        }, loc)
    }

    pub fn var(s: String, loc: Loc) -> Self {
        Self::new(AstKind::Var(s), loc)
    }
}

use std::str::FromStr;

impl FromStr for Ast {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let tokens = lex(s)?;
        let ast = parse(tokens)?;
        Ok(ast)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UniOpKind {
    /// 正号
    Plus,
    /// 負号
    Minus,
}

pub type UniOp = Annot<UniOpKind>;

impl UniOp {
    pub fn plus(loc: Loc) -> Self {
        Self::new(UniOpKind::Plus, loc)
    }

    pub fn minus(loc: Loc) -> Self {
        Self::new(UniOpKind::Minus, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    /// 加算
    Add,
    /// 減算
    Sub,
    /// 乗算
    Mult,
    /// 除算
    Div,
}

pub type BinOp = Annot<BinOpKind>;

impl BinOp {
    pub fn add(loc: Loc) -> Self {
        Self::new(BinOpKind::Add, loc)
    }

    pub fn sub(loc: Loc) -> Self {
        Self::new(BinOpKind::Sub, loc)
    }

    pub fn mult(loc: Loc) -> Self {
        Self::new(BinOpKind::Mult, loc)
    }

    pub fn div(loc: Loc) -> Self {
        Self::new(BinOpKind::Div, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Error {
    Lexer(LexError),
    Parser(ParseError),
}

impl Error {
    /// 診断メッセージを表示する
    pub fn show_diagnostic(&self, input: &str) {
        use self::Error::*;
        use self::ParseError as P;
        // エラー情報とその位置情報を取り出す。エラーの種類によって位置情報を調整する。
        let (e, loc): (&std::error::Error, Loc) = match self {
            Lexer(e) => (e, e.loc.clone()),
            Parser(e) => {
                let loc = match e {
                    P::UnexpectedToken(Token { loc, .. })
                    | P::NotExpression(Token { loc, .. })
                    | P::NotOperator(Token { loc, .. })
                    | P::UnclosedOpenParen(Token { loc, .. }) => loc.clone(),
                    // redundant expressionはトークン以降行末までが余りなのでlocの終了位置を調整する
                    P::RedundantExpression(Token { loc, .. }) => Loc(loc.0, input.len()),
                    // EoFはloc情報を持っていないのでその場で作る
                    P::Eof => Loc(input.len(), input.len() + 1),
                };
                (e, loc)
            }
        };
        // エラー情報を簡単に表示し
        eprintln!("{}", e);
        // エラー位置を指示する
        print_annot(input, loc);
        // sourceを再帰的に呼び出し表示
        let mut source = e.source();

        while let Some(e) = source {
            eprintln!("caused by {}", e);
            source = e.source()
        }
    }
}

impl From<LexError> for Error {
    fn from(e: LexError) -> Self {
        Error::Lexer(e)
    }
}

impl From<ParseError> for Error {
    fn from(e: ParseError) -> Self {
        Error::Parser(e)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "parser error")
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use self::Error::*;

        match self {
            Lexer(lex) => Some(lex),
            Parser(parse) => Some(parse),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParseError {
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

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ParseError::*;

        match self {
            UnexpectedToken(t) => write!(f, "{}: {} is not expected.", t.loc, t.value),
            NotExpression(t) => write!(f, "{}: '{}' is not a start of expression", t.loc, t.value),
            NotOperator(t) => write!(f, "{}: '{}' is not an operator", t.loc, t.value),
            UnclosedOpenParen(t) => write!(f, "{}: '{}' is not closed", t.loc, t.value),
            RedundantExpression(t) => {
                write!(f, "{}: expression after '{}' is redundant", t.loc, t.value)
            }
            Eof => write!(f, "End of file"),
        }
    }
}

impl std::error::Error for ParseError {}

fn print_annot(input: &str, loc: Loc) {
    // 入力に対して
    eprintln!("{}", input);
    // 位置情報を分かりやすく示す
    eprintln!("{}{}", " ".repeat(loc.0), "^".repeat(loc.1 - loc.0));
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InterpreterErrorKind {
    DivisionByZero,
    UnboundVariable(String),
}

pub type InterpreterError = Annot<InterpreterErrorKind>;
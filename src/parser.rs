use super::common::*;
use std::iter::Peekable;

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
            b'>' => lex_a_token!(lex_greater_greater_equal(input, position)),
            b'<' => lex_a_token!(lex_less_less_equal(input, position)),
            b'=' => lex_a_token!(lex_assign_equal(input, position)),
            b'!' => lex_a_token!(lex_negative_unequal(input, position)),
            b';' => lex_a_token!(lex_semicolon(input, position)),
            b'a'...b'z' => lex_a_token!(lex_str(input, position)),
            b' ' | b'\n' | b'\t' => {
                let (_, pos) = skip_spaces(input, position)?;
                position = pos;
            }
            b'\'' => lex_a_token!(lex_char(input, position)),
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

fn lex_greater_greater_equal(input: &[u8], pos: usize) -> Result<(Token, usize), LexError> {
    let start = pos;
    let end = recognize_many(input, start, |b| b">".contains(&b));
    let end = recognize_many(input, end, |b| b"=".contains(&b));

    if end - start == 1 {
        return Ok((Token::greater(Loc(start, end)), end));
    }
    if end - start == 2 {
        return Ok((Token::greater_equal(Loc(start, end)), end));
    }
    Err(LexError::invalid_char(input[end] as char, Loc(start, end)))
}

fn lex_less_less_equal(input: &[u8], pos: usize) -> Result<(Token, usize), LexError> {
    let start = pos;
    let end = recognize_many(input, start, |b| b"<".contains(&b));
    let end = recognize_many(input, end, |b| b"=".contains(&b));

    if end - start == 1 {
        return Ok((Token::less(Loc(start, end)), end));
    }
    if end - start == 2 {
        return Ok((Token::less_equal(Loc(start, end)), end));
    }
    Err(LexError::invalid_char(input[end] as char, Loc(start, end)))
}

fn lex_assign_equal(input: &[u8], pos: usize) -> Result<(Token, usize), LexError> {
    let start = pos;
    let end = recognize_many(input, start, |b| b"=".contains(&b));

    if end - start == 1 {
        return Ok((Token::assign(Loc(start, end)), end));
    }
    if end - start == 2 {
        return Ok((Token::equal(Loc(start, end)), end));
    }

    Err(LexError::invalid_char(input[end] as char, Loc(start, end)))
}

fn lex_negative_unequal(input: &[u8], pos: usize) -> Result<(Token, usize), LexError> {
    let start = pos;
    let end = recognize_many(input, start, |b| b"!".contains(&b));
    let end = recognize_many(input, end, |b| b"=".contains(&b));

    if end - start == 1 {
        return Ok((Token::negative(Loc(start, end)), end));
    }
    if end - start == 2 {
        return Ok((Token::unequal(Loc(start, end)), end));
    }
    Err(LexError::invalid_char(input[end] as char, Loc(start, end)))
}

fn lex_semicolon(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b';').map(|(_, end)| (Token::semicolon(Loc(start, end)), end))
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

fn lex_str(input: &[u8], pos: usize) -> Result<(Token, usize), LexError> {
    use std::str::from_utf8;

    let start = pos;
    let end = recognize_many(input, start, |b| b"abcdefghijklmnopqrstuvwxyz".contains(&b));
    let s = from_utf8(&input[start..end])
        // start..posの構成から `from_utf8` は常に成功するため`unwrap`しても安全
        .unwrap();

    match s {
        "void" => Ok((Token::void(Loc(start, end)), end)),
        "char" => Ok((Token::char(Loc(start, end)), end)),
        "bool" => Ok((Token::bool(Loc(start, end)), end)),
        "int" => Ok((Token::int(Loc(start, end)), end)),
        "float" => Ok((Token::float(Loc(start, end)), end)),
        "double" => Ok((Token::double(Loc(start, end)), end)),
        "struct" => Ok((Token::r#struct(Loc(start, end)), end)),
        "union" => Ok((Token::union(Loc(start, end)), end)),
        "enum" => Ok((Token::r#enum(Loc(start, end)), end)),
        "unsigned" => Ok((Token::unsigned(Loc(start, end)), end)),
        "signed" => Ok((Token::signed(Loc(start, end)), end)),
        "long" => Ok((Token::long(Loc(start, end)), end)),
        "return" => Ok((Token::r#return(Loc(start, end)), end)),
        _ => Ok((Token::string(s, Loc(start, end)), end)),
    }
}

fn lex_char(input: &[u8], pos: usize) -> Result<(Token, usize), LexError> {
    let start = pos;

    let c = input[start + 1] as char;

    Ok((Token::char_literal(c, Loc(start, start + 1)), start + 3))
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

#[test]
fn test_lexer2() {
    assert_eq!(
        lex("int x = 1;"),
        Ok(vec![
            Token::int(Loc(0, 3)),
            Token::string("x", Loc(4, 5)),
            Token::assign(Loc(6, 7)),
            Token::number(1, Loc(8, 9)),
            Token::semicolon(Loc(9, 10)),
        ])
    )
}

#[test]
fn test_lexer3() {
    assert_eq!(
        lex("char x = 't';"),
        Ok(vec![
            Token::char(Loc(0, 4)),
            Token::string("x", Loc(5, 6)),
            Token::assign(Loc(7, 8)),
            Token::char_literal('t', Loc(9, 10)),
            Token::semicolon(Loc(12, 13)),
        ])
    )
}

#[test]
fn test_lexer4() {
    assert_eq!(
        lex("10 == 20"),
        Ok(vec![
            Token::number(10, Loc(0, 2)),
            Token::equal(Loc(3, 5)),
            Token::number(20, Loc(6, 8)),
        ])
    )
}

#[test]
fn test_lexer5() {
    assert_eq!(
        lex("999 != 1000"),
        Ok(vec![
            Token::number(999, Loc(0, 3)),
            Token::unequal(Loc(4, 6)),
            Token::number(1000, Loc(7, 11))
        ])
    )
}

#[test]
fn test_lexer6() {
    assert_eq!(
        lex("return 100;"),
        Ok(vec![
            Token::r#return(Loc(0, 6)),
            Token::number(100, Loc(7, 10)),
            Token::semicolon(Loc(10, 11))
        ])
    )
}

pub fn parse(tokens: Vec<Token>) -> Result<Ast, ParseError> {
    // 入力をイテレータにし、Peekableにする
    let mut tokens = tokens.into_iter().peekable();
    // その後parse_exprを呼んでエラー処理をする
    let result = parse_stmt(&mut tokens)?;
    match tokens.next() {
        Some(token) => Err(ParseError::RedundantExpression(token)),
        None => Ok(result),
    }
}

fn parse_expr3<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    fn parse_expr3_op<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<BinOp, ParseError>
    where
        Tokens: Iterator<Item = Token>,
    {
        let operator = tokens
            .peek()
            // イテレータの終わりは入力の終端なのでエラーを出す(Option -> Result)
            .ok_or(ParseError::Eof)
            // エラーを返すかもしれない値をつなげる
            .and_then(|token| match token.value {
                TokenKind::Plus => Ok(BinOp::add(token.loc.clone())),
                TokenKind::Minus => Ok(BinOp::sub(token.loc.clone())),
                _ => Err(ParseError::NotOperator(token.clone())),
            })?;

        tokens.next();

        Ok(operator)
    }

    parse_left_binop(tokens, parse_expr2, parse_expr3_op)
}

fn parse_expr2<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    // `parse_left_binop` に渡す関数を定義する
    fn parse_expr2_op<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<BinOp, ParseError>
    where
        Tokens: Iterator<Item = Token>,
    {
        let op = tokens
            .peek()
            .ok_or(ParseError::Eof)
            .and_then(|tok| match tok.value {
                TokenKind::Asterisk => Ok(BinOp::mult(tok.loc.clone())),
                TokenKind::Slash => Ok(BinOp::div(tok.loc.clone())),
                _ => Err(ParseError::NotOperator(tok.clone())),
            })?;
        tokens.next();
        Ok(op)
    }

    parse_left_binop(tokens, parse_expr1, parse_expr2_op)
}

fn parse_expr1<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    match tokens.peek().map(|token| token.value.clone()) {
        Some(TokenKind::Plus) | Some(TokenKind::Minus) => {
            // ("+" | "-")
            let operator = match tokens.next() {
                Some(Token {
                    value: TokenKind::Plus,
                    loc,
                }) => UniOp::plus(loc),
                Some(Token {
                    value: TokenKind::Minus,
                    loc,
                }) => UniOp::minus(loc),
                _ => unreachable!(),
            };

            // , ATOM
            let e = parse_atom(tokens)?;
            let loc = operator.loc.merge(&e.loc);

            Ok(Ast::uniop(operator, e, loc))
        }
        // | ATOM
        _ => parse_atom(tokens),
    }
}

fn parse_atom<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    tokens
        .next()
        .ok_or(ParseError::Eof)
        .and_then(|token| match token.value.clone() {
            // UNUMBER
            TokenKind::Number(n) => Ok(Ast::new(AstKind::Num(n), token.loc)),
            // | "(", EXPR3, ")" ;
            TokenKind::LParen => {
                let e = parse_expr3(tokens)?;

                match tokens.next() {
                    Some(Token {
                        value: TokenKind::RParen,
                        ..
                    }) => Ok(e),
                    Some(t) => Err(ParseError::RedundantExpression(t)),
                    _ => Err(ParseError::UnclosedOpenParen(token)),
                }
            }
            TokenKind::r#String(s) => {
                // char
                if s.starts_with("'") && s.ends_with("'") && s.len() == 3 {
                    let mut chars = s.chars();
                    chars.next();

                    let c = match chars.next() {
                        Some(c) => c,
                        None => {
                            return Err(ParseError::InvalidChar(token));
                        }
                    };

                    // char literal
                    return Ok(Ast::new(AstKind::CharLiteral(c), token.loc));
                }

                // Var
                Ok(Ast::new(AstKind::Var(s), token.loc))
            }
            _ => Err(ParseError::NotExpression(token)),
        })
}

fn parse_stmt<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    match tokens.peek().map(|token| token.value.clone()) {
        Some(TokenKind::Int) => parse_stmt1(tokens, TokenKind::Int),
        Some(TokenKind::Char) => parse_stmt1(tokens, TokenKind::Char),
        Some(TokenKind::Bool) => parse_stmt1(tokens, TokenKind::Bool),
        Some(TokenKind::Return) => parse_stmt_return(tokens),
        _ => parse_expr3(tokens),
    }
}

fn parse_stmt1<Tokens>(tokens: &mut Peekable<Tokens>, kind: TokenKind) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    let loc_start = match tokens.next() {
        Some(Token { loc, .. }) => loc,
        _ => unreachable!(),
    };
    let var = match tokens.next() {
        Some(Token {
            value: TokenKind::r#String(s),
            ..
        }) => s,
        Some(t) => return Err(ParseError::UnexpectedToken(t)),
        _ => unreachable!(),
    };
    match tokens.next() {
        Some(Token {
            value: TokenKind::Assign,
            ..
        }) => (),
        Some(t) => return Err(ParseError::UnexpectedToken(t)),
        _ => unreachable!(),
    };
    let body = expr_equality(tokens)?;
    let loc_end = match tokens.next() {
        Some(Token {
            value: TokenKind::Semicolon,
            loc,
        }) => loc,
        Some(t) => return Err(ParseError::UnexpectedToken(t)),
        _ => unreachable!(),
    };
    let loc = loc_start.merge(&loc_end);

    match kind {
        TokenKind::Int => Ok(Ast::int(var, body, loc)),
        TokenKind::Char => Ok(Ast::char(var, body, loc)),
        TokenKind::Bool => Ok(Ast::bool(var, body, loc)),
        _ => unimplemented!(),
    }
}

fn parse_stmt_return<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    let loc_start = match tokens.next() {
        Some(Token { loc, .. }) => loc,
        _ => unreachable!(),
    };

    match tokens.next() {
        Some(Token {
            value: TokenKind::Return,
            ..
        }) => (),
        Some(t) => return Err(ParseError::UnexpectedToken(t)),
        _ => unreachable!(),
    };

    let body = expr_equality(tokens)?;

    println!("{:?}", body);

    let loc_end = match tokens.next() {
        Some(Token {
            value: TokenKind::Semicolon,
            loc,
        }) => loc,
        Some(t) => return Err(ParseError::UnexpectedToken(t)),
        _ => unreachable!(),
    };

    let loc = loc_start.merge(&loc_end);

    Ok(Ast::r#return(body, loc))
}

fn expr_equality<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    fn parse_expr_equality_op<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<EqOp, ParseError>
    where
        Tokens: Iterator<Item = Token>,
    {
        let operator = tokens
            .peek()
            // イテレータの終わりは入力の終端なのでエラーを出す(Option -> Result)
            .ok_or(ParseError::Eof)
            // エラーを返すかもしれない値をつなげる
            .and_then(|token| match token.value {
                TokenKind::Equal => Ok(EqOp::equal(token.loc.clone())),
                TokenKind::Unequal => Ok(EqOp::unequal(token.loc.clone())),
                _ => Err(ParseError::NotOperator(token.clone())),
            })?;

        tokens.next();

        Ok(operator)
    }

    parse_left_eqop(tokens, parse_relational, parse_expr_equality_op)
}

fn parse_relational<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    fn parse_expr_relational_op<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<RelOp, ParseError>
    where
        Tokens: Iterator<Item = Token>,
    {
        let operator = tokens
            .peek()
            // イテレータの終わりは入力の終端なのでエラーを出す(Option -> Result)
            .ok_or(ParseError::Eof)
            // エラーを返すかもしれない値をつなげる
            .and_then(|token| match token.value {
                TokenKind::Greater => Ok(RelOp::greater(token.loc.clone())),
                TokenKind::GreaterEqual => Ok(RelOp::greater_equal(token.loc.clone())),
                TokenKind::Less => Ok(RelOp::less(token.loc.clone())),
                TokenKind::LessEqual => Ok(RelOp::less_equal(token.loc.clone())),
                _ => Err(ParseError::NotOperator(token.clone())),
            })?;

        tokens.next();

        Ok(operator)
    }

    parse_left_relop(tokens, parse_expr1, parse_expr_relational_op)
}

fn parse_left_binop<Tokens>(
    tokens: &mut Peekable<Tokens>,
    subexpr_parser: fn(&mut Peekable<Tokens>) -> Result<Ast, ParseError>,
    op_parser: fn(&mut Peekable<Tokens>) -> Result<BinOp, ParseError>,
) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
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
            }
            _ => break,
        }
    }
    Ok(e)
}

fn parse_left_relop<Tokens>(
    tokens: &mut Peekable<Tokens>,
    subexpr_parser: fn(&mut Peekable<Tokens>) -> Result<Ast, ParseError>,
    op_parser: fn(&mut Peekable<Tokens>) -> Result<RelOp, ParseError>,
) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
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
                e = Ast::relop(operator, e, r, loc)
            }
            _ => break,
        }
    }

    Ok(e)
}

fn parse_left_eqop<Tokens>(
    tokens: &mut Peekable<Tokens>,
    subexpr_parser: fn(&mut Peekable<Tokens>) -> Result<Ast, ParseError>,
    op_parser: fn(&mut Peekable<Tokens>) -> Result<EqOp, ParseError>,
) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
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
                e = Ast::eqop(operator, e, r, loc)
            }
            _ => break,
        }
    }
    Ok(e)
}

#[test]
fn test_parser() {
    // 1 + 2 * 3 - -10
    let ast = parse(vec![
        Token::number(1, Loc(0, 1)),
        Token::plus(Loc(2, 3)),
        Token::number(2, Loc(4, 5)),
        Token::asterisk(Loc(6, 7)),
        Token::number(3, Loc(8, 9)),
        Token::minus(Loc(10, 11)),
        Token::minus(Loc(12, 13)),
        Token::number(10, Loc(13, 15)),
    ]);
    assert_eq!(
        ast,
        Ok(Ast::binop(
            BinOp::sub(Loc(10, 11)),
            Ast::binop(
                BinOp::add(Loc(2, 3)),
                Ast::num(1, Loc(0, 1)),
                Ast::binop(
                    BinOp::new(BinOpKind::Mult, Loc(6, 7)),
                    Ast::num(2, Loc(4, 5)),
                    Ast::num(3, Loc(8, 9)),
                    Loc(4, 9)
                ),
                Loc(0, 9),
            ),
            Ast::uniop(
                UniOp::minus(Loc(12, 13)),
                Ast::num(10, Loc(13, 15)),
                Loc(12, 15)
            ),
            Loc(0, 15)
        ))
    )
}

#[test]
fn test_parser2() {
    // int x = 1;
    let ast = parse(vec![
        Token::int(Loc(0, 3)),
        Token::string("x", Loc(4, 5)),
        Token::assign(Loc(6, 7)),
        Token::number(1, Loc(8, 9)),
        Token::semicolon(Loc(9, 10)),
    ]);
    assert_eq!(
        ast,
        Ok(Ast::int(
            "x".to_string(),
            Ast::num(1, Loc(8, 9)),
            Loc(0, 10)
        ))
    )
}

#[test]
fn test_parser3() {
    // bool hoge = 10 == 20;
    let ast = parse(vec![
        Token::bool(Loc(0, 4)),
        Token::string("hoge", Loc(5, 9)),
        Token::assign(Loc(10, 11)),
        Token::number(10, Loc(12, 14)),
        Token::equal(Loc(15, 17)),
        Token::number(20, Loc(18, 20)),
        Token::semicolon(Loc(20, 21)),
    ]);
    assert_eq!(
        ast,
        Ok(Ast::bool(
            "hoge".to_string(),
            Ast::eqop(
                EqOp::equal(Loc(15, 17)),
                Ast::num(10, Loc(12, 14)),
                Ast::num(20, Loc(18, 20)),
                Loc(12, 20)
            ),
            Loc(0, 21)
        ))
    )
}

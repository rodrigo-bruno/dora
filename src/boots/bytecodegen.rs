use std::collections::HashMap;

use ty::BuiltinType;
use ctxt::{VM, Fct, FctId, FctSrc};

use dora_parser::ast::*;
use dora_parser::ast::Expr::*;
use dora_parser::ast::Stmt::*;
use dora_parser::ast::*;
use dora_parser::interner::Name;
use dora_parser::lexer::token::{FloatSuffix, IntSuffix};

// TODO - have types for registers
#[derive(PartialEq,Debug)]
pub struct Register(usize);
#[derive(PartialEq, Debug, Eq, Hash)]
pub struct Label(usize);

macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

pub struct Context {
    var_map: HashMap<Name, Register>,
}

impl Context {
    pub fn new() -> Context {
        Context {
            var_map: HashMap::new(),
        }
    }

    pub fn new_var(&mut self, var: Name, reg: Register) {
        self.var_map.insert(var, reg);
    }

    pub fn get_reg(&self, var: Name) -> Option<&Register> {
        let reg = self.var_map.get(&var);
        reg
    }
}

pub struct LoopLabels {
    cond: Label,
    end: Label,
}

#[derive(PartialEq, Debug)]
pub enum Bytecode {
    AddShort(Register), // TODO - used where?
    AddInt(Register),
    AddLong(Register),
    AddFloat(Register),
    AddDouble(Register),
    BitwiseAnd(Register),
    BitwiseOr(Register),
    BitwiseXor(Register),
    DivInt(Register),
    DivLong(Register),
    DivFloat(Register),
    DivDouble(Register),
    LdarShort(Register),
    LdarInt(Register),
    LdarLong(Register),
    LdarFloat(Register),
    LdarDouble(Register),
    LdaShort(u8),
    LdaInt(u32),
    LdaLong(u64),
    LdaFloat(f32),
    LdaDouble(f64),
    LdaZero,
    LogicalNot,
    StarShort(Register),
    StarInt(Register),
    StarLong(Register),
    StarFloat(Register),
    StarDouble(Register),
    JumpIfFalse(Label),
    Jump(Label),
    Mod(Register),
    MulInt(Register),
    MulLong(Register),
    MulFloat(Register),
    MulDouble(Register),
    Neg,
    ShiftLeft(Register),
    ShiftRight(Register),
    SubInt(Register),
    SubLong(Register),
    SubFloat(Register),
    SubDouble(Register),
    Return,
    ReturnVoid,
    TestEqual(Register),
    TestGreatherThan(Register),
    TestGreatherThanOrEqual(Register),
    TestLessThan(Register),
    TestLessThanOrEqual(Register),
    TestNotEqual(Register),
}

pub struct BytecodeGen<'ast> {
    vm: &'ast VM<'ast>,
    code: Vec<Bytecode>,
    ctxs: Vec<Context>,
    loops: Vec<LoopLabels>,
    labels: HashMap<Label, usize>,
    regs: usize,
}

impl<'ast> BytecodeGen<'ast> {

    pub fn generate(vm: &'ast VM<'ast>, id: FctId) -> BytecodeGen {
        let fct = vm.fcts[id].borrow();
        let src = fct.src();
        let mut src = src.borrow_mut();
        let mut generator = BytecodeGen {
            vm : vm,
            code: Vec::new(),
            ctxs: Vec::new(),
            loops: Vec::new(),
            labels: HashMap::new(),
            regs: 0,
        };
        generator.generate_fct(vm, &fct, &mut src);
        return generator;
    }

    pub fn generate_fct(&mut self, vm: &'ast VM<'ast>, fct: &Fct<'ast>, src: &mut FctSrc) {
        if  !fct.ast.has_optimize {
            return;
        }

        // TODO - handle types

        if let Some(ref block) = fct.ast.block {
            self.visit_stmt(src, block);
        }

        if self.code.len() == 0 || self.code.last().unwrap() != &Bytecode::Return {
            self.code.push(Bytecode::ReturnVoid);
        }
    }

    pub fn dump(&self) {
        let mut btidx = 0;
        for btcode in self.code.iter() {
            match btcode {
                Bytecode::AddShort(Register(register)) =>
                    println!("{}: AddI8 {}", btidx, register),
                Bytecode::AddInt(Register(register)) =>
                    println!("{}: AddI32 {}", btidx, register),
                Bytecode::AddLong(Register(register)) =>
                    println!("{}: AddI64 {}", btidx, register),
                Bytecode::AddFloat(Register(register)) =>
                    println!("{}: AddF32 {}", btidx, register),
                Bytecode::AddDouble(Register(register)) =>
                    println!("{}: AddF64 {}", btidx, register),
                Bytecode::BitwiseAnd(Register(register)) =>
                    println!("{}: BitwiseAnd {}", btidx, register),
                Bytecode::BitwiseOr(Register(register)) =>
                    println!("{}: BitwiseOr {}", btidx, register),
                Bytecode::BitwiseXor(Register(register)) =>
                    println!("{}: BitwiseXor {}", btidx, register),
                Bytecode::DivInt(Register(register)) =>
                    println!("{}: DivI32 {}", btidx, register),
                Bytecode::DivLong(Register(register)) =>
                    println!("{}: DivI64 {}", btidx, register),
                Bytecode::DivFloat(Register(register)) =>
                    println!("{}: DivF32 {}", btidx, register),
                Bytecode::DivDouble(Register(register)) =>
                    println!("{}: DivF64 {}", btidx, register),
                Bytecode::LdarShort(Register(register)) =>
                    println!("{}: LdarI8 {}", btidx, register),
                Bytecode::LdarInt(Register(register)) =>
                    println!("{}: LdarI32 {}", btidx, register),
                Bytecode::LdarLong(Register(register)) =>
                    println!("{}: LdarI64 {}", btidx, register),
                Bytecode::LdarFloat(Register(register)) =>
                    println!("{}: LdarF32 {}", btidx, register),
                Bytecode::LdarDouble(Register(register)) =>
                    println!("{}: LdarF64 {}", btidx, register),
                Bytecode::LdaShort(value) =>
                    println!("{}: LdaI8 {}", btidx, value),
                Bytecode::LdaInt(value) =>
                    println!("{}: LdaI32 {}", btidx, value),
                Bytecode::LdaLong(value) =>
                    println!("{}: LdaI64 {}", btidx, value),
                Bytecode::LdaFloat(value) =>
                    println!("{}: LdaF32 {}", btidx, value),
                Bytecode::LdaDouble(value) =>
                    println!("{}: LdaF64 {}", btidx, value),
                Bytecode::LdaZero =>
                    println!("{}: LdaZero", btidx),
                Bytecode::LogicalNot =>
                    println!("{}: LogicalNot", btidx),
                Bytecode::StarShort(Register(register)) =>
                    println!("{}: StarI8 {}", btidx, register),
                Bytecode::StarInt(Register(register)) =>
                    println!("{}: StarI32 {}", btidx, register),
                Bytecode::StarLong(Register(register)) =>
                    println!("{}: StarI64 {}", btidx, register),
                Bytecode::StarFloat(Register(register)) =>
                    println!("{}: StarF32 {}", btidx, register),
                Bytecode::StarDouble(Register(register)) =>
                    println!("{}: StarF64 {}", btidx, register),
                Bytecode::JumpIfFalse(label) =>
                    println!("{}: JumpIfFalse {}", btidx, self.labels.get(label).unwrap()),
                Bytecode::Jump(label) =>
                    println!("{}: Jump {}", btidx, self.labels.get(label).unwrap()),
                Bytecode::Mod(Register(register)) =>
                    println!("{}: Mod {}", btidx, register),
                Bytecode::MulInt(Register(register)) =>
                    println!("{}: MulI32 {}", btidx, register),
                Bytecode::MulLong(Register(register)) =>
                    println!("{}: MulI64 {}", btidx, register),
                Bytecode::MulFloat(Register(register)) =>
                    println!("{}: MulI32 {}", btidx, register),
                Bytecode::MulDouble(Register(register)) =>
                    println!("{}: MulI64 {}", btidx, register),
                Bytecode::Neg =>
                    println!("{}: Neg", btidx),
                Bytecode::ShiftLeft(Register(register)) =>
                    println!("{}: ShiftLeft {}", btidx, register),
                Bytecode::ShiftRight(Register(register)) =>
                    println!("{}: ShiftRight {}", btidx, register),
                Bytecode::SubInt(Register(register)) =>
                    println!("{}: SubI32 {}", btidx, register),
                Bytecode::SubLong(Register(register)) =>
                    println!("{}: SubI64 {}", btidx, register),
                Bytecode::SubFloat(Register(register)) =>
                    println!("{}: SubI32 {}", btidx, register),
                Bytecode::SubDouble(Register(register)) =>
                    println!("{}: SubI64 {}", btidx, register),
                Bytecode::Return =>
                    println!("{}: Return", btidx),
                Bytecode::ReturnVoid =>
                    println!("{}: ReturnVoid", btidx),
                Bytecode::TestEqual(Register(register)) =>
                    println!("{}: TestEqual {}", btidx, register),
                Bytecode::TestGreatherThan(Register(register)) =>
                    println!("{}: TestGreaterThan {}", btidx, register),
                Bytecode::TestGreatherThanOrEqual(Register(register)) =>
                    println!("{}: TestGreatherThanOrEqual {}", btidx, register),
                Bytecode::TestLessThan(Register(register)) =>
                    println!("{}: TestLessThan {}", btidx, register),
                Bytecode::TestLessThanOrEqual(Register(register)) =>
                    println!("{}: TestLessThanOrEqual {}", btidx, register),
                Bytecode::TestNotEqual(Register(register)) =>
                    println!("{}: TestNotEqual {}", btidx, register),
            }
            btidx = btidx + 1;
        }
    }

    pub fn get_reg(&self, var: Name) -> Option<&Register> {
        for ctx in self.ctxs.iter() {
            let opt = ctx.get_reg(var);
            if opt.is_some() {
                return opt;
            }
        }
        None
    }

    // TODO - implement other statements
    fn visit_stmt(&mut self, src: &mut FctSrc, stmt: &Stmt) {
        match *stmt {
            StmtBlock(ref block) => self.visit_block(src, block),
            StmtReturn(ref ret) => self.visit_stmt_return(src, ret),
            StmtBreak(ref stmt) => self.visit_stmt_break(src, stmt),
            StmtContinue(ref stmt) => self.visit_stmt_continue(src, stmt),
            StmtExpr(ref expr) => self.visit_stmt_expr(src, expr),
            StmtIf(ref stmt) => self.visit_stmt_if(src, stmt),
            StmtVar(ref stmt) => self.visit_stmt_var(src, stmt),
            StmtWhile(ref stmt) => self.visit_stmt_while(src, stmt),
            // StmtLoop(ref stmt) => {},
            // StmtThrow(ref stmt) => {},
            // StmtDefer(ref stmt) => {},
            // StmtDo(ref stmt) => {},
            // StmtSpawn(ref stmt) => {},
            // StmtFor(ref stmt) => {},
            _ => unimplemented!(),
        }
    }

    fn gen_star(&mut self, src: &mut FctSrc, nid: NodeId, reg: Register) {
        let expr_ty = src.ty(nid);
        match expr_ty {
            BuiltinType::Int => { self.code.push(Bytecode::StarInt(reg)) },
            BuiltinType::Long => { self.code.push(Bytecode::StarLong(reg)) },
            BuiltinType::Float => { self.code.push(Bytecode::StarFloat(reg)) },
            BuiltinType::Double => { self.code.push(Bytecode::StarDouble(reg)) },
            _ => unimplemented!(),
        }
    }

    fn visit_stmt_var(&mut self, src: &mut FctSrc, stmt: &StmtVarType) {
        let reg = self.regs;
        let varid = stmt.name;
        self.regs += 1;
        self.ctxs
            .last_mut()
            .unwrap()
            .new_var(varid as Name, Register(reg));

        if let Some(ref expr) = stmt.expr {
            self.visit_expr(src, expr);
            self.gen_star(src, expr.id(), Register(reg));
        } else {
            self.code.push(Bytecode::LdaZero);
            // TODO - check this, I dont know the type.
            self.code.push(Bytecode::StarInt(Register(reg)));
        };
    }

    fn visit_stmt_while(&mut self, src: &mut FctSrc, stmt: &StmtWhileType) {
        let cond_lbl = self.labels.len();
        let end_lbl = cond_lbl + 1;
        self.loops.push(LoopLabels {
            cond: Label(cond_lbl),
            end: Label(end_lbl),
        });

        self.labels.insert(Label(cond_lbl), self.code.len());
        self.labels.insert(Label(end_lbl), 0); // Just a place holder

        self.visit_expr(src, &stmt.cond);
        self.code.push(Bytecode::JumpIfFalse(Label(end_lbl)));
        self.visit_stmt(src, &stmt.block);
        self.code.push(Bytecode::Jump(Label(cond_lbl)));
        self.labels.insert(Label(end_lbl), self.code.len());
        self.loops.pop();
    }

    fn visit_stmt_if(&mut self, src: &mut FctSrc, stmt: &StmtIfType) {
        let else_lbl = self.labels.len();
        let end_lbl = else_lbl + 1;

        self.labels.insert(Label(else_lbl), 0); // Just a place holder
        self.labels.insert(Label(end_lbl), 0); // Just a place holder

        self.visit_expr(src, &stmt.cond);
        self.code.push(Bytecode::JumpIfFalse(Label(else_lbl)));
        self.visit_stmt(src, &stmt.then_block);
        self.code.push(Bytecode::Jump(Label(end_lbl)));
        self.labels.insert(Label(else_lbl), self.code.len());
        match &stmt.else_block {
            Some(else_block) => { self.visit_stmt(src, &else_block); },
            _ => {},
        }
        self.labels.insert(Label(end_lbl), self.code.len());
    }

    fn visit_stmt_expr(&mut self, src: &mut FctSrc, stmt: &StmtExprType) {
        self.visit_expr(src, &stmt.expr);
    }

    fn visit_block(&mut self, src: &mut FctSrc, block: &StmtBlockType) {
        let regs = self.regs;
        self.ctxs.push(Context::new());
        for stmt in &block.stmts {
            self.visit_stmt(src, stmt);
        }
        self.ctxs.pop();
        self.regs = regs;
    }

    fn visit_stmt_return(&mut self, src: &mut FctSrc, ret: &StmtReturnType) {
        if let Some(ref expr) = ret.expr {
            self.visit_expr(src, expr);
        }
        self.code.push(Bytecode::Return);
    }

    fn visit_stmt_break(&mut self, src: &mut FctSrc, stmt: &StmtBreakType) {
        let Label(end) = self.loops.pop().unwrap().end;
        self.code.push(Bytecode::Jump(Label(end)));
    }

    fn visit_stmt_continue(&mut self, src: &mut FctSrc, stmt: &StmtContinueType) {
        let Label(cond) = self.loops.last().unwrap().cond;
        self.code.push(Bytecode::Jump(Label(cond)));
    }

    // TODO - implement other expressions
    fn visit_expr(&mut self, src: &mut FctSrc, expr: &Expr) {
        match *expr {
            ExprUn(ref un) => self.visit_expr_un(src, un),
            ExprBin(ref bin) => self.visit_expr_bin(src, bin),
            // ExprField(ref field) => {},
            // ExprArray(ref array) => {},
            ExprLitChar(ref lit) => self.visit_expr_lit_char(src, lit),
            ExprLitInt(ref lit) => self.visit_expr_lit_int(src, lit),
            ExprLitFloat(ref lit) => self.visit_expr_lit_float(src, lit),
            // ExprLitStr(ref lit) => {},
            // ExprLitStruct(ref lit) => {},
            // ExprLitBool(ref lit) => {},
            ExprIdent(ref ident) => self.visit_expr_ident(src, ident),
            ExprAssign(ref assign) => self.visit_expr_assign(src, assign),
            // ExprCall(ref call) => {},
            // ExprDelegation(ref call) => {},
            // ExprSelf(ref selfie) => {},
            // ExprSuper(ref expr) => {},
            // ExprNil(ref nil) => {},
            // ExprConv(ref expr) => {},
            // ExprTry(ref expr) => {},
            // ExprLambda(ref expr) => {},
            _ => unimplemented!(),
        }
    }

    fn visit_expr_lit_char(&mut self, src: &mut FctSrc, lit: &ExprLitCharType) {
        self.code.push(Bytecode::LdaShort(lit.value as u8));
    }

    fn visit_expr_lit_int(&mut self, src: &mut FctSrc, lit: &ExprLitIntType) {
        if lit.value == 0 {
            self.code.push(Bytecode::LdaZero);
        } else {
            let bytecode = match lit.suffix {
                IntSuffix::Byte => Bytecode::LdaShort(lit.value as u8),
                IntSuffix::Int => Bytecode::LdaInt(lit.value as u32),
                IntSuffix::Long => Bytecode::LdaLong(lit.value as u64),
            };
            self.code.push(bytecode);
        }
    }

    fn visit_expr_lit_float(&mut self, src: &mut FctSrc, lit: &ExprLitFloatType) {
        let bytecode = match lit.suffix {
            FloatSuffix::Float => Bytecode::LdaFloat(lit.value as f32),
            FloatSuffix::Double => Bytecode::LdaDouble(lit.value as f64),
        };
        self.code.push(bytecode);
    }

    fn visit_expr_un(&mut self, src: &mut FctSrc, expr: &ExprUnType) {
        self.visit_expr(src, &expr.opnd);
        match expr.op {
            UnOp::Plus => {}
            UnOp::Neg => self.code.push(Bytecode::Neg),
            UnOp::Not => self.code.push(Bytecode::LogicalNot),
        }
    }

    fn gen_add(&mut self, src: &mut FctSrc, expr: &ExprBinType, reg: Register) {
        let expr_ty = src.ty(expr.id);
        match expr_ty {
            BuiltinType::Int => { self.code.push(Bytecode::AddInt(reg)) },
            BuiltinType::Long => { self.code.push(Bytecode::AddLong(reg)) },
            BuiltinType::Float => { self.code.push(Bytecode::AddFloat(reg)) },
            BuiltinType::Double => { self.code.push(Bytecode::AddDouble(reg)) },
            _ => unimplemented!(),
        }
    }

    fn gen_sub(&mut self, src: &mut FctSrc, expr: &ExprBinType, reg: Register) {
        let expr_ty = src.ty(expr.id);
        match expr_ty {
            BuiltinType::Int => { self.code.push(Bytecode::SubInt(reg)) },
            BuiltinType::Long => { self.code.push(Bytecode::SubLong(reg)) },
            BuiltinType::Float => { self.code.push(Bytecode::SubFloat(reg)) },
            BuiltinType::Double => { self.code.push(Bytecode::SubDouble(reg)) },
            _ => unimplemented!(),
        }
    }

    fn gen_mul(&mut self, src: &mut FctSrc, expr: &ExprBinType, reg: Register) {
        let expr_ty = src.ty(expr.id);
        match expr_ty {
            BuiltinType::Int => { self.code.push(Bytecode::MulInt(reg)) },
            BuiltinType::Long => { self.code.push(Bytecode::MulLong(reg)) },
            BuiltinType::Float => { self.code.push(Bytecode::MulFloat(reg)) },
            BuiltinType::Double => { self.code.push(Bytecode::MulDouble(reg)) },
            _ => unimplemented!(),
        }
    }

    fn gen_div(&mut self, src: &mut FctSrc, expr: &ExprBinType, reg: Register) {
        let expr_ty = src.ty(expr.id);
        match expr_ty {
            BuiltinType::Int => { self.code.push(Bytecode::DivInt(reg)) },
            BuiltinType::Long => { self.code.push(Bytecode::DivLong(reg)) },
            BuiltinType::Float => { self.code.push(Bytecode::DivFloat(reg)) },
            BuiltinType::Double => { self.code.push(Bytecode::DivDouble(reg)) },
            _ => unimplemented!(),
        }
    }

    fn visit_expr_bin(&mut self, src: &mut FctSrc, expr: &ExprBinType) {
        self.visit_expr(src, &expr.rhs);
        let rhs_reg = self.regs;
        self.regs += 1;
        self.gen_star(src, expr.rhs.id(), Register(rhs_reg));
        self.visit_expr(src, &expr.lhs);
        match expr.op {
            BinOp::Add => { self.gen_add(src, expr, Register(rhs_reg)) },
            BinOp::Sub => { self.gen_sub(src, expr, Register(rhs_reg)) },
            BinOp::Mul => { self.gen_mul(src, expr, Register(rhs_reg)) },
            BinOp::Div => { self.gen_div(src, expr, Register(rhs_reg)) },
            BinOp::Mod => { self.code.push(Bytecode::Mod(Register(rhs_reg))) },
            BinOp::BitOr => { self.code.push(Bytecode::BitwiseOr(Register(rhs_reg))) },
            BinOp::BitAnd => { self.code.push(Bytecode::BitwiseAnd(Register(rhs_reg))) },
            BinOp::BitXor => { self.code.push(Bytecode::BitwiseXor(Register(rhs_reg))) },
            BinOp::ShiftL => { self.code.push(Bytecode::ShiftLeft(Register(rhs_reg))) },
            BinOp::ShiftR => { self.code.push(Bytecode::ShiftRight(Register(rhs_reg))) },
            // BinOp::Or => { },
            // BinOp::And => { },
            // BinOp::UnShiftR => { },
            BinOp::Cmp(op) => {
                match op {
                    CmpOp::Eq => self.code.push(Bytecode::TestEqual(Register(rhs_reg))),
                    CmpOp::Ne => self.code.push(Bytecode::TestNotEqual(Register(rhs_reg))),
                    CmpOp::Lt => self.code.push(Bytecode::TestLessThan(Register(rhs_reg))),
                    CmpOp::Le => self
                        .code
                        .push(Bytecode::TestLessThanOrEqual(Register(rhs_reg))),
                    CmpOp::Gt => self
                        .code
                        .push(Bytecode::TestGreatherThan(Register(rhs_reg))),
                    CmpOp::Ge => self
                        .code
                        .push(Bytecode::TestGreatherThanOrEqual(Register(rhs_reg))),
                    // CmpOp::Is => { },
                    // CmpOp::IsNot => { },
                    _ => unimplemented!(),
                }
            }
            _ => unimplemented!(),
        }
        self.regs -= 1;
    }

    fn visit_expr_assign(&mut self, src: &mut FctSrc, expr: &ExprAssignType) {
        self.visit_expr(src, &expr.rhs);
        match *expr.lhs {
            ExprIdent(ref assign) => {
                let Register(reg) = *self.get_reg(assign.name).unwrap();
                self.gen_star(src, expr.rhs.id(), Register(reg));
            },
            _ => unimplemented!(),
        }
    }

    fn visit_expr_ident(&mut self, src: &mut FctSrc, ident: &ExprIdentType) {
        let Register(reg) = *self.get_reg(ident.name).unwrap();
        let expr_ty = src.ty(ident.id);
        match expr_ty {
            BuiltinType::Int => { self.code.push(Bytecode::LdarInt(Register(reg))) },
            BuiltinType::Long => { self.code.push(Bytecode::LdarLong(Register(reg))) },
            BuiltinType::Float => { self.code.push(Bytecode::LdarFloat(Register(reg))) },
            BuiltinType::Double => { self.code.push(Bytecode::LdarDouble(Register(reg))) },
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use driver::cmd::Args;
    use boots::bytecodegen::*;
    use boots::bytecodegen::Bytecode::*;
    use boots::bytecodegen::*;
    use dora_parser::interner::Interner;
    use dora_parser::lexer::reader::Reader;
    use os::mem;
    use semck;
    use driver::start;

    fn parse(code: &'static str) -> (Ast, Interner) {
        let id_generator = NodeIdGenerator::new();
        let mut interner = Interner::new();
        let mut ast = Ast::new();

        let reader = Reader::from_string(code);
        Parser::new(reader, &id_generator, &mut ast, &mut interner)
            .parse()
            .unwrap();

        (ast, interner)
    }

    fn run_test(
            tname: &'static str,
            code: &'static str,
            exp_code: Vec<Bytecode>,
            exp_labels: HashMap<Label, usize>)  {
        mem::init_page_size();
        let mut interner = Interner::new();
        let id_generator = NodeIdGenerator::new();
        let mut ast = Ast::new();

        if let Err(code) = start::parse_dir("stdlib", &id_generator, &mut ast, &mut interner) {
            println!("failed to load stdlib ({})", code);
        }

        if let Err(code) = Parser::new(
                Reader::from_string(code),
                &id_generator,
                &mut ast,
                &mut interner).parse() {
            println!("failed to load code from string ({})", code);
        }

        let mut vm = VM::new(Args::default(), &ast, interner);
        semck::check(&mut vm);

        let name = vm.interner.intern(tname);
        let fctid = vm.sym.borrow().get_fct(name).unwrap();
        let bytecodegen = BytecodeGen::generate(&vm, fctid);
        assert_eq!(exp_code, bytecodegen.code);
        assert_eq!(exp_labels, bytecodegen.labels);
    }

    #[test]
    fn gen_nooptimize() {
        run_test("f", "fun f() {1 + 2;}", vec![], hashmap![]);
    }

    #[test]
    fn gen_add() {
        run_test(
            "f",
            "optimize fun f() {1 + 2;}",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                AddInt(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_sub() {
        run_test(
            "f",
            "optimize fun f() {1 - 2;}",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                SubInt(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_div() {
        run_test(
            "f",
            "optimize fun f() {1 / 2;}",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                DivInt(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_mul() {
        run_test(
            "f",
            "optimize fun f() {1 * 2;}",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                MulInt(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_stmt_var_noinit() {
        run_test(
            "f",
            "optimize fun f() { let x; }",
            vec![LdaZero, StarInt(Register(0)), ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_stmt_var_init() {
        run_test(
            "f",
            "optimize fun f() { let x = 1; }",
            vec![LdaInt(1), StarInt(Register(0)), ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_stmt_while() {
        run_test(
            "f",
            "optimize fun f() { while 1 { 0; } }",
            vec![
                LdaInt(1),
                JumpIfFalse(Label(1)),
                LdaZero,
                Jump(Label(0)),
                ReturnVoid],
            hashmap![Label(0) => 0, Label(1) => 4]);
    }

    #[test]
    fn gen_stmt_if() {
        run_test(
            "f",
            "optimize fun f() { if 0 { 1; } }",
            vec![
                LdaZero,
                JumpIfFalse(Label(0)),
                LdaInt(1),
                Jump(Label(1)),
                ReturnVoid],
            hashmap![Label(0) => 4, Label(1) => 4]);
    }

    #[test]
    fn gen_stmt_if_else() {
        run_test(
            "f",
            "optimize fun f() { if 0 { 1; } else { 2; } }",
            vec![
                LdaZero,
                JumpIfFalse(Label(0)),
                LdaInt(1),
                Jump(Label(1)),
                LdaInt(2),
                ReturnVoid],
            hashmap![Label(0) => 4, Label(1) => 5]);
    }

    #[test]
    fn gen_stmt_break() {
        run_test(
            "f",
            "optimize fun f() { while 1 { break; } }",
            vec![
                LdaInt(1),
                JumpIfFalse(Label(1)),
                Jump(Label(1)),
                Jump(Label(0)),
                ReturnVoid],
            hashmap![Label(0) => 0, Label(1) => 4]);
    }

    #[test]
    fn gen_stmt_continue() {
        run_test(
            "f",
            "optimize fun f() { while 1 { continue; } }",
            vec![
                LdaInt(1),
                JumpIfFalse(Label(1)),
                Jump(Label(0)),
                Jump(Label(0)),
                ReturnVoid],
            hashmap![Label(0) => 0, Label(1) => 4]);
    }

    #[test]
    fn gen_expr_lit_int() {
        run_test(
            "f",
            "optimize fun f() { 1; }",
            vec![LdaInt(1), ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_lit_long() {
        run_test(
            "f",
            "optimize fun f() { 1L; }",
            vec![LdaLong(1), ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_lit_float() {
        run_test(
            "f",
            "optimize fun f() { 1.0F; }",
            vec![LdaFloat(1.0), ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_lit_double() {
        run_test(
            "f",
            "optimize fun f() { 1.0D; }",
            vec![LdaDouble(1.0), ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_lit_zero() {
        run_test(
            "f",
            "optimize fun f() { 0; }",
            vec![LdaZero, ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_puls() {
        run_test(
            "f",
            "optimize fun f() { +1; }",
            vec![ LdaInt(1), ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_neg() {
        run_test(
            "f",
            "optimize fun f() { -1; }",
            vec![ LdaInt(1), Neg, ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_not() {
        run_test(
            "f",
            "optimize fun f() { !1; }",
            vec![ LdaInt(1), LogicalNot, ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_mod() {
        run_test(
            "f",
            "optimize fun f() { 1 % 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                Mod(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_bit_or() {
        run_test(
            "f",
            "optimize fun f() { 1 | 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                BitwiseOr(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_bit_and() {
        run_test(
            "f",
            "optimize fun f() { 1 & 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                BitwiseAnd(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_bit_xor() {
        run_test(
            "f",
            "optimize fun f() { 1 ^ 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                BitwiseXor(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_bit_shiftl() {
        run_test(
            "f",
            "optimize fun f() { 1 << 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                ShiftLeft(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_bit_shiftr() {
        run_test(
            "f",
            "optimize fun f() { 1 >> 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                ShiftRight(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_test_equal() {
        run_test(
            "f",
            "optimize fun f() { 1 == 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                TestEqual(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_test_notequal() {
        run_test(
            "f",
            "optimize fun f() { 1 != 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                TestNotEqual(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_test_lessthan() {
        run_test(
            "f",
            "optimize fun f() { 1 < 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                TestLessThan(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_test_lessthanequal() {
        run_test(
            "f",
            "optimize fun f() { 1 <= 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                TestLessThanOrEqual(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_test_greaterthan() {
        run_test(
            "f",
            "optimize fun f() { 1 > 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                TestGreatherThan(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_test_greaterthanequall() {
        run_test(
            "f",
            "optimize fun f() { 1 >= 2; }",
            vec![
                LdaInt(2),
                StarInt(Register(0)),
                LdaInt(1),
                TestGreatherThanOrEqual(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_ident() {
        run_test(
            "f",
            "optimize fun f() { let x = 1; x; }",
            vec![
                LdaInt(1),
                StarInt(Register(0)),
                LdarInt(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_assign() {
        run_test(
            "f",
            "optimize fun f() { var x = 1; x = 2; }",
            vec![
                LdaInt(1),
                StarInt(Register(0)),
                LdaInt(2),
                StarInt(Register(0)),
                ReturnVoid],
            hashmap![]);
    }

    #[test]
    fn gen_expr_return() {
        run_test(
            "f",
            "optimize fun f() { return 1; }",
            vec![ LdaInt(1), Return ],
            hashmap![]);
    }

    #[test]
    fn gen_expr_returnvoid() {
        run_test(
            "f",
            "optimize fun f() { }",
            vec![ ReturnVoid ],
            hashmap![]);
    }
}

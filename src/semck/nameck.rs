use ctxt::*;
use error::msg::Msg;

use ast::*;
use ast::Expr::*;
use ast::Stmt::*;
use ast::visit::*;
use interner::Name;
use lexer::position::Position;

use sym::Sym;
use sym::Sym::*;
use ty::BuiltinType;

pub fn check<'ast>(ctxt: &Context<'ast>) {
    for fct in ctxt.fcts.iter() {
        let mut fct = fct.lock().unwrap();

        if fct.kind.is_src() {
            let ast = fct.ast();
            let mut nameck = NameCheck {
                ctxt: ctxt,
                fct: &mut fct,
                ast: ast,
            };

            nameck.check();
        }
    }
}

struct NameCheck<'a, 'ast: 'a> {
    ctxt: &'a Context<'ast>,
    fct: &'a mut Fct<'ast>,
    ast: &'ast Function,
}

impl<'a, 'ast> NameCheck<'a, 'ast> {
    fn check(&mut self) {
        self.ctxt.sym.borrow_mut().push_level();

        if self.fct.owner_class.is_some() {
            // add hidden this parameter for ctors and methods
            self.add_hidden_parameter_this();
        }

        for p in &self.ast.params { self.visit_param(p); }
        self.visit_stmt(&self.ast.block);

        self.ctxt.sym.borrow_mut().pop_level();
    }

    pub fn add_hidden_parameter_this(&mut self) {
        let var_id = VarId(self.fct.src().vars.len());
        let cls_id = self.fct.owner_class.unwrap();
        let ast_id = self.fct.src().ast.id;
        let name = self.ctxt.interner.intern("this");

        let var = Var {
            id: VarId(0),
            name: name,
            data_type: BuiltinType::Class(cls_id),
            mutable: false,
            node_id: ast_id,
            offset: 0
        };

        self.fct.src_mut().vars.push(var);
    }

    pub fn add_var<F>(&mut self, mut var: Var, replacable: F) ->
            Result<VarId, Sym> where F: FnOnce(&Sym) -> bool {
        let name = var.name;
        let var_id = VarId(self.fct.src().vars.len());

        var.id = var_id;

        let result = match self.ctxt.sym.borrow().get(name) {
            Some(sym) => if replacable(&sym) { Ok(var_id) } else { Err(sym) },
            None => Ok(var_id)
        };

        if result.is_ok() {
            self.ctxt.sym.borrow_mut().insert(name, SymVar(var_id));
            assert!(self.fct.src_mut().defs.insert(var.node_id, IdentType::Var(var_id)).is_none());
        }

        self.fct.src_mut().vars.push(var);

        result
    }

    fn check_stmt_let(&mut self, var: &'ast StmtLetType) {
        let var_ctxt = Var {
            id: VarId(0),
            name: var.name,
            data_type: BuiltinType::Unit,
            mutable: var.mutable,
            node_id: var.id,
            offset: 0
        };

        // variables are not allowed to replace types, other variables
        // and functions can be replaced
        if let Err(_) = self.add_var(var_ctxt, |sym| !sym.is_type()) {
            let name = str(self.ctxt, var.name);
            report(self.ctxt, var.pos, Msg::ShadowType(name));
        }

        if let Some(ref expr) = var.expr {
            self.visit_expr(expr);
        }
    }

    fn check_stmt_block(&mut self, block: &'ast StmtBlockType) {
        self.ctxt.sym.borrow_mut().push_level();
        for stmt in &block.stmts { self.visit_stmt(stmt); }
        self.ctxt.sym.borrow_mut().pop_level();
    }

    fn check_expr_ident(&mut self, ident: &'ast ExprIdentType) {
        if let Some(id) = self.ctxt.sym.borrow().get_var(ident.name) {
            self.fct.src_mut().defs.insert(ident.id, IdentType::Var(id));
            return;
        }

        if let Some(clsid) = self.fct.owner_class {
            let cls = self.ctxt.cls_by_id(clsid);

            for prop in &cls.props {
                if prop.name == ident.name {
                    let ident_type = IdentType::Prop(clsid, prop.id);
                    assert!(self.fct.src_mut().defs.insert(ident.id, ident_type).is_none());
                    return;
                }
            }
        }

        report(self.ctxt, ident.pos, Msg::UnknownIdentifier(ident.name));
    }

    fn check_expr_call(&mut self, call: &'ast ExprCallType) {
        let mut found = false;

        // do not check method calls yet
        if let Some(ref object) = call.object {
            self.visit_expr(object);

            for arg in &call.args {
                self.visit_expr(arg);
            }

            return;
        }

        if let Some(sym) = self.ctxt.sym.borrow().get(call.name) {
            if sym.is_fct() {
                let call_type = CallType::Fct(sym.to_fct().unwrap());
                self.fct.src_mut().calls.insert(call.id, call_type);
                found = true;

            } else if sym.is_type() && sym.to_type().unwrap().is_cls() {
                let clsid = sym.to_type().unwrap().cls();
                let cls = self.ctxt.cls_by_id(clsid);

                let call_type = CallType::Ctor(clsid, cls.ctor);
                self.fct.src_mut().calls.insert(call.id, call_type);
                found = true;
            }
        }

        if !found {
            let name = str(self.ctxt, call.name);
            report(self.ctxt, call.pos, Msg::UnknownFunction(name));
        }

        // also parse function arguments
        for arg in &call.args {
            self.visit_expr(arg);
        }
    }
}

impl<'a, 'ast> Visitor<'ast> for NameCheck<'a, 'ast> {
    fn visit_param(&mut self, p: &'ast Param) {
        let var_ctxt = Var {
            id: VarId(0),
            name: p.name,
            data_type: BuiltinType::Unit,
            mutable: p.mutable,
            node_id: p.id,
            offset: 0,
        };

        // params are only allowed to replace functions,
        // types and vars cannot be replaced
        if let Err(sym) = self.add_var(var_ctxt, |sym| sym.is_fct()) {
            let name = str(self.ctxt, p.name);
            let msg = if sym.is_type() {
                Msg::ShadowType(name)
            } else {
                Msg::ShadowParam(name)
            };

            report(self.ctxt, p.pos, msg);
        }
    }

    fn visit_stmt(&mut self, s: &'ast Stmt) {
        match *s {
            StmtLet(ref stmt) => self.check_stmt_let(stmt),
            StmtBlock(ref stmt) => self.check_stmt_block(stmt),

            // no need to handle rest of statements
            _ => visit::walk_stmt(self, s)
        }
    }

    fn visit_expr(&mut self, e: &'ast Expr) {
        match *e {
            ExprIdent(ref ident) => self.check_expr_ident(ident),
            ExprCall(ref call) => self.check_expr_call(call),

            // no need to handle rest of expressions
            _ => visit::walk_expr(self, e)
        }
    }
}

fn report(ctxt: &Context, pos: Position, msg: Msg) {
    ctxt.diag.borrow_mut().report(pos, msg);
}

fn str(ctxt: &Context, name: Name) -> String {
    ctxt.interner.str(name).to_string()
}

#[cfg(test)]
mod tests {
    use error::msg::Msg;
    use interner::Name;
    use semck::tests::*;

    #[test]
    fn multiple_functions() {
        ok("fn f() {}\nfn g() {}");
    }

    #[test]
    fn redefine_function() {
        err("fn f() {}\nfn f() {}", pos(2, 1),
            Msg::ShadowFunction("f".into()));
    }

    #[test]
    fn shadow_type_with_function() {
        err("fn int() {}", pos(1, 1),
            Msg::ShadowType("int".into()));
    }

    #[test]
    fn shadow_type_with_param() {
        err("fn test(bool: Str) {}", pos(1, 9),
            Msg::ShadowType("bool".into()));
    }

    #[test]
    fn shadow_type_with_var() {
        err("fn test() { let Str = 3; }", pos(1, 13),
            Msg::ShadowType("Str".into()));
    }

    #[test]
    fn shadow_function() {
        ok("fn f() { let f = 1; }");
        err("fn f() { let f = 1; f(); }", pos(1, 21),
            Msg::UnknownFunction("f".into()));
    }

    #[test]
    fn shadow_var() {
        ok("fn f() { let f = 1; let f = 2; }");
    }

    #[test]
    fn shadow_param() {
        err("fn f(a: int, b: int, a: Str) {}", pos(1, 22),
            Msg::ShadowParam("a".into()));
    }

    #[test]
    fn multiple_params() {
        ok("fn f(a: int, b: int, c:Str) {}");
    }

    #[test]
    fn undefined_variable() {
        err("fn f() { let b = a; }", pos(1, 18), Msg::UnknownIdentifier(Name(2)));
        err("fn f() { a; }", pos(1, 10), Msg::UnknownIdentifier(Name(1)));
    }

    #[test]
    fn undefined_function() {
        err("fn f() { foo(); }", pos(1, 10),
            Msg::UnknownFunction("foo".into()));
    }

    #[test]
    fn recursive_function_call() {
        ok("fn f() { f(); }");
    }

    #[test]
    fn function_call() {
        ok("fn a() {}\nfn b() { a(); }");

        // non-forward definition of functions
        ok("fn a() { b(); }\nfn b() {}");
    }

    #[test]
    fn variable_outside_of_scope() {
        err("fn f() -> int { { let a = 1; } return a; }", pos(1, 39),
            Msg::UnknownIdentifier(Name(2)));

        ok("fn f() -> int { let a = 1; { let a = 2; } return a; }");
    }
}

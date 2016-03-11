use std::cmp;

use ast::*;
use ast::Stmt::*;
use ast::Expr::*;
use ast::visit::*;
use cpu::{self, Reg};
use ctxt::{Arg, CallSite, Context, Fct, Store, Var};
use jit::expr::is_leaf;
use mem;
use ty::BuiltinType;

pub fn generate<'a, 'ast: 'a>(ctxt: &'a Context<'ast>, fct: &'a mut Fct<'ast>) {
    let ast = fct.ast();

    let mut ig = InfoGenerator {
        ctxt: ctxt,
        fct: fct,
        ast: ast,

        localsize: 0,
        max_tempsize: 0,
        cur_tempsize: 0,
        argsize: 0,

        param_offset: cpu::PARAM_OFFSET,
        leaf: true,
    };

    ig.generate();
}

struct InfoGenerator<'a, 'ast: 'a> {
    ctxt: &'a Context<'ast>,
    fct: &'a mut Fct<'ast>,
    ast: &'ast Function,

    localsize: i32,
    max_tempsize: i32,
    cur_tempsize: i32,
    argsize: i32,

    param_offset: i32,
    leaf: bool,
}

impl<'a, 'ast> Visitor<'ast> for InfoGenerator<'a, 'ast> {
    fn visit_param(&mut self, p: &'ast Param) {
        let idx = (p.idx as usize) + if self.fct.ctor { 1 } else { 0 };

        // only some parameters are passed in registers
        // these registers need to be stored into local variables
        if idx < cpu::REG_PARAMS.len() {
            self.reserve_stack_for_node(p.id);

        // the rest of the parameters are already stored on the stack
        // just use the current offset
        } else {
            let var = self.fct.var_by_node_id_mut(p.id);
            var.offset = self.param_offset;

            // determine next `param_offset`
            self.param_offset = cpu::next_param_offset(self.param_offset, var.data_type);
        }
    }

    fn visit_stmt(&mut self, s: &'ast Stmt) {
        if let StmtLet(ref var) = *s {
            self.reserve_stack_for_node(var.id);
        }

        visit::walk_stmt(self, s);
    }

    fn visit_expr_top(&mut self, e: &'ast Expr) {
        self.cur_tempsize = 0;
        self.visit_expr(e);
        self.max_tempsize = cmp::max(self.cur_tempsize, self.max_tempsize);
    }

    fn visit_expr(&mut self, e: &'ast Expr) {
        match *e {
            ExprCall(ref expr) => self.expr_call(expr),
            ExprArray(ref expr) => self.expr_array(expr),
            ExprAssign(ref expr) => self.expr_assign(expr),
            ExprBin(ref expr) => self.expr_bin(expr),

            _ => visit::walk_expr(self, e)
        }
    }
}

impl<'a, 'ast> InfoGenerator<'a, 'ast> {
    fn generate(&mut self) {
        if self.fct.owner_class.is_some() {
            self.reserve_stack_for_self();
        }

        self.visit_fct(self.ast);

        let src = self.fct.src_mut();
        src.localsize = self.localsize;
        src.tempsize = self.max_tempsize;
        src.argsize = self.argsize;
        src.leaf = self.leaf;
    }

    fn reserve_stack_for_self(&mut self) {
        let var = self.fct.var_self();

        let ty_size = var.data_type.size();
        self.localsize = mem::align_i32(self.localsize + ty_size, ty_size);
        var.offset = -self.localsize;
    }

    fn reserve_stack_for_node(&mut self, id: NodeId) {
        let var = self.fct.var_by_node_id_mut(id);

        let ty_size = var.data_type.size();
        self.localsize = mem::align_i32(self.localsize + ty_size, ty_size);
        var.offset = -self.localsize;

        // println!("local `{}` on {} with type {:?}", *self.ctxt.interner.str(var.name),
        //     var.offset, var.data_type);
    }

    fn expr_array(&mut self, expr: &'ast ExprArrayType) {
        self.visit_expr(&expr.object);
        self.visit_expr(&expr.index);

        let args = vec![
            Arg::Expr(&expr.object, 0),
            Arg::Expr(&expr.index, 0)
        ];

        self.universal_call(expr.id, args);
    }

    fn expr_call(&mut self, expr: &'ast ExprCallType) {
        let call_type = *self.fct.src().calls.get(&expr.id).unwrap();
        let ctor = call_type.is_ctor();

        let mut args = expr.args.iter().map(|arg| Arg::Expr(arg, 0)).collect::<Vec<_>>();

        if ctor {
            args.insert(0, Arg::Selfie(call_type.cls_id(), 0));
        }

        self.universal_call(expr.id, args);
    }

    fn universal_call(&mut self, id: NodeId, args: Vec<Arg<'ast>>) {
        // function invokes another function
        self.leaf = false;
        let mut with_ctor = false;

        for arg in &args {
            match *arg {
                Arg::Expr(ast, _) => self.visit_expr(ast),
                Arg::Selfie(_, _) => { with_ctor = true; }
            }
        }

        // reserve stack for arguments
        let no_args = args.len() as i32;
        let argsize = 8 * if no_args <= 6 { 0 } else { no_args - 6 };

        if argsize > self.argsize {
            self.argsize = argsize;
        }

        let fid = self.fct.src().calls.get(&id).unwrap().fct_id();
        let mut types = vec![];

        let args = args.iter().enumerate().map(|(ind, arg)| {
            match *arg {
                Arg::Expr(ast, _) => {
                    let ind = if with_ctor { ind-1 } else { ind };

                    let ty = if self.fct.id == fid {
                        self.fct.params_types[ind]
                    } else {
                        self.ctxt.fct_by_id(fid, |fct| fct.params_types[ind])
                    };

                    types.push(ty);
                    Arg::Expr(ast, self.reserve_temp_for_node_with_type(ast.id(), ty))
                }

                Arg::Selfie(cid, _) => {
                    types.push(BuiltinType::Class(cid));
                    Arg::Selfie(cid, self.reserve_temp_for_ctor(id))
                }
            }
        }).collect::<Vec<_>>();

        let return_type = if self.fct.id == fid {
            self.fct.return_type
        } else {
            self.ctxt.fct_by_id(fid, |fct| fct.return_type)
        };

        let csite = CallSite {
            callee: fid,
            args: args,
            types: types,
            return_type: return_type
        };

        // remember args
        self.fct.src_mut().call_sites.insert(id, csite);
    }

    fn expr_assign(&mut self, e: &'ast ExprAssignType) {
        if e.lhs.is_ident() {
            self.visit_expr(&e.rhs);

            let ident_type = *self.fct.src().defs.get(&e.lhs.id()).unwrap();

            if ident_type.is_prop() {
                self.reserve_temp_for_node_with_type(e.lhs.id(), BuiltinType::Ptr);
            }

        } else {
            assert!(e.lhs.is_prop());
            let lhs = e.lhs.to_prop().unwrap();

            self.visit_expr(&lhs.object);
            self.visit_expr(&e.rhs);

            self.reserve_temp_for_node(lhs.object.id());
        }
    }

    fn expr_bin(&mut self, expr: &'ast ExprBinType) {
        self.visit_expr(&expr.lhs);
        self.visit_expr(&expr.rhs);

        if !is_leaf(&expr.rhs) {
            self.reserve_temp_for_node(expr.lhs.id());
        }
    }

    fn reserve_temp_for_node(&mut self, id: NodeId) -> i32 {
        let ty = self.fct.src().get_type(id);
        self.reserve_temp_for_node_with_type(id, ty)
    }

    fn reserve_temp_for_ctor(&mut self, id: NodeId) -> i32 {
        self.reserve_temp_for_node_with_type(id, BuiltinType::Ptr)
    }

    fn reserve_temp_for_node_with_type(&mut self, id: NodeId, ty: BuiltinType) -> i32 {
        let ty_size = ty.size();
        self.cur_tempsize = mem::align_i32(self.cur_tempsize + ty_size, ty_size);

        self.fct.src_mut().storage.insert(id, Store::Temp(self.cur_tempsize));
        // println!("temp on {} with type {:?}", self.cur_tempsize, ty);

        self.cur_tempsize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ctxt::*;
    use test;

    fn info<F>(code: &'static str, f: F) where F: FnOnce(&Fct) {
        test::parse(code, |ctxt| {
            let ast = ctxt.ast.elements[0].to_function().unwrap();

            ctxt.fct_by_node_id_mut(ast.id, |fct| {
                generate(ctxt, fct);

                f(fct);
            });
        });
    }

    #[test]
    fn test_tempsize() {
        info("fn f() { 1+2*3; }", |fct| { assert_eq!(4, fct.src().tempsize); });
        info("fn f() { 2*3+4+5; }", |fct| { assert_eq!(0, fct.src().tempsize); });
        info("fn f() { 1+(2+(3+4)); }", |fct| { assert_eq!(8, fct.src().tempsize); })
    }

    #[test]
    fn test_tempsize_for_fct_call() {
        info("fn f() { g(1,2,3,4,5,6); }
              fn g(a:int, b:int, c:int, d:int, e:int, f:int) {}", |fct| {
            assert_eq!(24, fct.src().tempsize);
        });

        info("fn f() { g(1,2,3,4,5,6,7,8); }
              fn g(a:int, b:int, c:int, d:int, e:int, f:int, g:int, h:int) {}", |fct| {
            assert_eq!(32, fct.src().tempsize);
        });

        info("fn f() { g(1,2,3,4,5,6,7,8)+(1+2); }
              fn g(a:int, b:int, c:int, d:int, e:int, f:int, g:int, h:int) -> int {
                  return 0;
              }", |fct| {
            assert_eq!(36, fct.src().tempsize);
        });
    }

    #[test]
    fn test_invocation_flag() {
        info("fn f() { g(); } fn g() { }", |fct| {
            assert!(!fct.src().leaf);
        });

        info("fn f() { }", |fct| {
            assert!(fct.src().leaf);
        });
    }

    #[test]
    fn test_param_offset() {
        info("fn f(a: bool, b: int) { let c = 1; }", |fct| {
            assert_eq!(12, fct.src().localsize);

            for (var, offset) in fct.src().vars.iter().zip(&[-1, -8, -12]) {
                assert_eq!(*offset, var.offset);
            }
        });
    }

    #[test]
    fn test_params_over_6_offset() {
        info("fn f(a: int, b: int, c: int, d: int,
                   e: int, f: int, g: int, h: int) {
                  let i : int = 1;
              }", |fct| {
            assert_eq!(28, fct.src().localsize);
            let offsets = [-4, -8, -12, -16, -20, -24, 16, 24, -28];

            for (var, offset) in fct.src().vars.iter().zip(&offsets) {
                assert_eq!(*offset, var.offset);
            }
        });
    }

    #[test]
    fn test_var_offset() {
        info("fn f() { let a = true; let b = false; let c = 2; let d = \"abc\"; }", |fct| {
            assert_eq!(16, fct.src().localsize);

            for (var, offset) in fct.src().vars.iter().zip(&[-1, -2, -8, -16]) {
                assert_eq!(*offset, var.offset);
            }
        });
    }
}

use cpu::Reg;
use cpu::instr::*;
use cpu::Reg::*;
use cpu::trap;
use ctxt::*;
use jit::buffer::*;
use sym::BuiltinType;

pub fn prolog(buf: &mut Buffer, stacksize: i32) {
    emit_pushq_reg(buf, Reg::RBP);
    emit_movq_reg_reg(buf, Reg::RSP, Reg::RBP);

    if stacksize > 0 {
        emit_subq_imm_reg(buf, stacksize, RSP);
    }
}

pub fn epilog(buf: &mut Buffer, stacksize: i32) {
    if stacksize > 0 {
        emit_addq_imm_reg(buf, stacksize, RSP);
    }

    emit_popq_reg(buf, Reg::RBP);
    emit_retq(buf);
}

pub fn jump_if_zero(buf: &mut Buffer, reg: Reg, lbl: Label) {
    emit_testl_reg_reg(buf, reg, reg);
    emit_jz(buf, lbl);
}

pub fn jump(buf: &mut Buffer, lbl: Label) {
    emit_jmp(buf, lbl);
}

// emit debug instruction
pub fn debug(buf: &mut Buffer) {
    // emit int3 = 0xCC
    emit_op(buf, 0xCC);
}

// emit stub instruction
pub fn stub(buf: &mut Buffer) {
    let dest = R10;

    // mov r10, [trap::COMPILER]
    emit_rex(buf, 1, dest.msb(), 0, 0);
    emit_op(buf, 0x8b);
    emit_modrm(buf, 0, dest.and7(), 0b100);
    emit_sib(buf, 0, 0b100, 0b101);
    emit_u32(buf, trap::COMPILER);
}

pub fn var_store(buf: &mut Buffer, ctxt: &Context, src: Reg, var: VarInfoId) {
    let var_infos = ctxt.var_infos.borrow();
    let var = &var_infos[var.0];

    match var.data_type {
        BuiltinType::Bool => emit_movb_reg_memq(buf, src, RBP, var.offset),
        BuiltinType::Int => emit_movl_reg_memq(buf, src, RBP, var.offset),
        BuiltinType::Str => emit_movq_reg_memq(buf, src, RBP, var.offset),
        BuiltinType::Unit => {},
    }
}

pub fn var_load(buf: &mut Buffer, ctxt: &Context, var: VarInfoId, dest: Reg) {
    let var_infos = ctxt.var_infos.borrow();
    let var = &var_infos[var.0];

    match var.data_type {
        BuiltinType::Bool => emit_movb_memq_reg(buf, RBP, var.offset, dest),
        BuiltinType::Int => emit_movl_memq_reg(buf, RBP, var.offset, dest),
        BuiltinType::Str => emit_movq_memq_reg(buf, RBP, var.offset, dest),
        BuiltinType::Unit => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use jit::buffer::Buffer;

    #[test]
    fn test_stub() {
        let mut buf = Buffer::new();
        stub(&mut buf);

        assert_eq!(vec![0x4C, 0x8B, 0x14, 0x25, 0x11, 0x47, 0x00, 0x00], buf.finish());
    }
}
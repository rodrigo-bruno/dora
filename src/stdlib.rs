use libc;

use std::io::{self, Write};
use std::mem;
use std::process;
use std::ptr;
use std::str;
use std::thread;

use class::TypeParams;
use ctxt::{exception_set, get_ctxt};
use exception::{alloc_exception, stacktrace_from_last_dtn};
use gc::GcReason;
use object::{ByteArray, Handle, Obj, Str};
use os::signal::Trap;

use sym::Sym::SymFct;

pub extern "C" fn byte_to_string(val: u8) -> Handle<Str> {
    let buffer = val.to_string();
    let ctxt = get_ctxt();

    Str::from_buffer(ctxt, buffer.as_bytes())
}

pub extern "C" fn char_to_string(val: char) -> Handle<Str> {
    let buffer = val.to_string();
    let ctxt = get_ctxt();

    Str::from_buffer(ctxt, buffer.as_bytes())
}

pub extern "C" fn int_to_string(val: i32) -> Handle<Str> {
    let buffer = val.to_string();
    let ctxt = get_ctxt();

    Str::from_buffer(ctxt, buffer.as_bytes())
}

pub extern "C" fn long_to_string(val: i64) -> Handle<Str> {
    let buffer = val.to_string();
    let ctxt = get_ctxt();

    Str::from_buffer(ctxt, buffer.as_bytes())
}

pub extern "C" fn float_to_string(val: f32) -> Handle<Str> {
    let buffer = val.to_string();
    let ctxt = get_ctxt();

    Str::from_buffer(ctxt, buffer.as_bytes())
}

pub extern "C" fn double_to_string(val: f64) -> Handle<Str> {
    let buffer = val.to_string();
    let ctxt = get_ctxt();

    Str::from_buffer(ctxt, buffer.as_bytes())
}

pub extern "C" fn print(val: Handle<Str>) {
    io::stdout().write(val.content()).unwrap();
}

pub extern "C" fn addr(val: Handle<Obj>) -> u64 {
    val.raw() as usize as u64
}

pub extern "C" fn fatal_error(msg: Handle<Str>) {
    write!(&mut io::stderr(), "fatal error: ").expect("could not print to stderr");
    io::stderr().write(msg.content()).unwrap();
    writeln!(&mut io::stderr(), "").expect("could not print to stderr");
    process::exit(1);
}

pub extern "C" fn abort() {
    writeln!(&mut io::stderr(), "program aborted.").expect("could not print to stderr");
    process::exit(1);
}

pub extern "C" fn exit(status: i32) {
    process::exit(status);
}

pub extern "C" fn timestamp() -> u64 {
    use timer;

    timer::timestamp()
}

pub extern "C" fn println(val: Handle<Str>) {
    print(val);
    println!("");
}

pub extern "C" fn throw_native(val: bool) {
    if val {
        let ctxt = get_ctxt();
        let obj = alloc_exception(ctxt, Handle::null());
        let obj = ctxt.handles.root(obj);

        exception_set(obj.direct().raw() as *const u8);
    }
}

pub extern "C" fn call(fct: Handle<Str>) {
    let fct_name = fct.to_cstring();
    let fct_name = fct_name.to_str().unwrap();

    let ctxt = get_ctxt();
    let name = ctxt.interner.intern(fct_name);

    let sym = ctxt.sym.borrow().get(name);

    match sym {
        Some(SymFct(fct_id)) => {
            {
                let fct = ctxt.fcts[fct_id].borrow();

                if !fct.param_types.is_empty() {
                    writeln!(&mut io::stderr(), "fct `{}` takes arguments.", fct_name)
                        .expect("could not print to stderr");
                    process::exit(1);
                }
            }

            ctxt.run(fct_id);
        }

        _ => {
            writeln!(&mut io::stderr(), "fct `{}` not found.", fct_name)
                .expect("could not print to stderr");
            process::exit(1);
        }
    }
}

pub extern "C" fn strcmp(lhs: Handle<Str>, rhs: Handle<Str>) -> i32 {
    unsafe {
        libc::strcmp(
            lhs.data() as *const libc::c_char,
            rhs.data() as *const libc::c_char,
        )
    }
}

pub extern "C" fn strcat(lhs: Handle<Str>, rhs: Handle<Str>) -> Handle<Str> {
    let ctxt = get_ctxt();
    let lhs = ctxt.handles.root(lhs);
    let rhs = ctxt.handles.root(rhs);

    Str::concat(ctxt, lhs, rhs).direct()
}

pub extern "C" fn str_clone(val: Handle<Str>) -> Handle<Str> {
    let ctxt = get_ctxt();

    val.dup(ctxt)
}

pub extern "C" fn str_from_bytes(val: Handle<ByteArray>, offset: usize, len: usize) -> Handle<Str> {
    let ctxt = get_ctxt();
    let val: Handle<Str> = val.cast();
    let val = ctxt.handles.root(val);

    Str::from_str(ctxt, val, offset, len)
}

pub extern "C" fn gc_alloc(size: usize, array_ref: bool) -> *mut Obj {
    let ctxt = get_ctxt();
    ctxt.gc.alloc(ctxt, size, array_ref).to_mut_ptr()
}

pub extern "C" fn gc_collect() {
    let ctxt = get_ctxt();
    ctxt.gc.collect(ctxt, GcReason::ForceCollect);
}

pub extern "C" fn gc_minor_collect() {
    let ctxt = get_ctxt();
    ctxt.gc.minor_collect(ctxt, GcReason::ForceMinorCollect);
}

pub extern "C" fn str_len(s: Handle<Str>) -> i32 {
    s.len() as i32
}

pub extern "C" fn argc() -> i32 {
    let ctxt = get_ctxt();

    if let Some(ref args) = ctxt.args.arg_argument {
        args.len() as i32
    } else {
        0
    }
}

pub extern "C" fn argv(ind: i32) -> Handle<Str> {
    let ctxt = get_ctxt();

    if let Some(ref args) = ctxt.args.arg_argument {
        if ind >= 0 && ind < args.len() as i32 {
            let value = &args[ind as usize];

            return Str::from_buffer(ctxt, value.as_bytes());
        }
    }

    panic!("argument does not exist");
}

pub extern "C" fn str_parse_int(val: Handle<Str>) -> i32 {
    let slice = val.content();
    let val = str::from_utf8(slice).unwrap();

    val.parse::<i32>().unwrap_or(0)
}

pub extern "C" fn load_function(name: Handle<Str>) -> usize {
    let name = name.to_cstring();

    unsafe {
        let lib = libc::dlopen(ptr::null(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
        let addr = libc::dlsym(lib, name.as_ptr());

        libc::dlclose(lib);

        addr as usize
    }
}

pub extern "C" fn call0(addr: *const u8) -> usize {
    let fct: extern "C" fn() -> usize = unsafe { mem::transmute(addr) };

    fct()
}

pub extern "C" fn call1(addr: *const u8, arg1: usize) -> usize {
    let fct: extern "C" fn(usize) -> usize = unsafe { mem::transmute(addr) };

    fct(arg1)
}

pub extern "C" fn call2(addr: *const u8, arg1: usize, arg2: usize) -> usize {
    let fct: extern "C" fn(usize, usize) -> usize = unsafe { mem::transmute(addr) };

    fct(arg1, arg2)
}

pub extern "C" fn call3(addr: *const u8, arg1: usize, arg2: usize, arg3: usize) -> usize {
    let fct: extern "C" fn(usize, usize, usize) -> usize = unsafe { mem::transmute(addr) };

    fct(arg1, arg2, arg3)
}

pub extern "C" fn native_malloc(size: usize) -> *const u8 {
    unsafe { libc::malloc(size) as *const u8 }
}

pub extern "C" fn native_free(addr: *const u8) {
    unsafe { libc::free(addr as *mut libc::c_void) }
}

pub extern "C" fn trap(trap_id: u32) {
    let ctxt = get_ctxt();
    let trap = Trap::from(trap_id).expect("invalid trap id!");

    let msg = match trap {
        Trap::DIV0 => "division by 0",
        Trap::ASSERT => "assert failed",
        Trap::INDEX_OUT_OF_BOUNDS => "array index out of bounds",
        Trap::NIL => "nil check failed",
        Trap::CAST => "cast failed",
        Trap::THROW => "uncaught exception",
        Trap::UNEXPECTED => "unexpected exception",
        Trap::OOM => "out of memory",
    };

    println!("{}", msg);
    let stacktrace = stacktrace_from_last_dtn(ctxt);
    stacktrace.dump(ctxt);
    unsafe {
        libc::_exit(100 + trap_id as i32);
    }
}

pub extern "C" fn spawn_thread(obj: Handle<Obj>) {
    use baseline;
    use exception::DoraToNativeInfo;

    thread::spawn(move || {
        let ctxt = get_ctxt();

        let main = {
            let cls_id = obj.header().vtbl().class().cls_id;
            let cls = ctxt.classes[cls_id].borrow();
            let name = ctxt.interner.intern("run");
            cls.find_method(ctxt, name, false)
                .expect("run() method not found")
        };

        let fct_ptr = {
            let mut dtn = DoraToNativeInfo::new();
            let type_params = TypeParams::empty();
            ctxt.use_dtn(&mut dtn, || {
                baseline::generate(ctxt, main, &type_params, &type_params)
            })
        };

        let fct: extern "C" fn(Handle<Obj>) = unsafe { mem::transmute(fct_ptr) };
        fct(obj);
    });
}

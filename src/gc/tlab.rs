use ctxt::VM;
use gc::{fill_region, Address, Region};

pub const TLAB_SIZE: usize = 32 * 1024;
pub const TLAB_OBJECT_SIZE: usize = 8 * 1024;

pub fn initialize(vm: &VM, tlab: Region) {
    vm.tld.lock().tlab_initialize(tlab.start, tlab.end);
}

pub fn calculate_size(additional: usize) -> usize {
    assert!(additional < TLAB_OBJECT_SIZE);

    TLAB_SIZE + additional
}

pub fn allocate(vm: &VM, size: usize) -> Option<Address> {
    assert!(size < TLAB_OBJECT_SIZE);
    let mut tld = vm.tld.lock();
    let tlab = tld.tlab_region();

    if size <= tlab.size() {
        tld.tlab_initialize(tlab.start.offset(size), tlab.end);
        Some(tlab.start)
    } else {
        None
    }
}

pub fn make_iterable(vm: &VM) {
    let mut tld = vm.tld.lock();
    let tlab = tld.tlab_region();

    fill_region(vm, tlab.start, tlab.end);

    let n = Address::null();
    tld.tlab_initialize(n, n);
}

pub fn make_iterable_region(vm: &VM, start: Address, end: Address) {
    let mut tld = vm.tld.lock();

    fill_region(vm, start, end);

    let n = Address::null();
    tld.tlab_initialize(n, n);
}

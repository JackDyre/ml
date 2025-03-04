#include "lazy_alloc.h"
#include <cassert>
#include <cstdlib>
#include <ctime>

void LazyDeviceAllocator::set_host_valid_unchecked() {
  if (host_state == OWNED_INVALID) {
    host_state = OWNED_VALID;
  } else if (host_state == BORROWED_INVALID) {
    host_state = BORROWED_VALID;
  }
}

void LazyDeviceAllocator::set_host_invalid_unchecked() {
  if (host_state == OWNED_VALID) {
    host_state = OWNED_INVALID;
  } else if (host_state == BORROWED_VALID) {
    host_state = BORROWED_INVALID;
  }
}

void LazyDeviceAllocator::set_dev_valid_unchecked() {
  if (dev_state == OWNED_INVALID) {
    dev_state = OWNED_VALID;
  } else if (dev_state == BORROWED_INVALID) {
    dev_state = BORROWED_VALID;
  }
}

void LazyDeviceAllocator::set_dev_invalid_unchecked() {
  if (dev_state == OWNED_VALID) {
    dev_state = OWNED_INVALID;
  } else if (dev_state == BORROWED_VALID) {
    dev_state = BORROWED_INVALID;
  }
}

void LazyDeviceAllocator::set_host_state_unchecked(DeviceState state) {
  host_state = state;
}

void LazyDeviceAllocator::set_dev_state_unchecked(DeviceState state) {
  dev_state = state;
}

LazyDeviceAllocator::~LazyDeviceAllocator() {
  free_host();
  free_dev();
}

LazyDeviceAllocator LazyDeviceAllocator::new_no_alloc(size_t alloc_size) {
  LazyDeviceAllocator d;
  d.set_alloc_size_unchecked(alloc_size);
  return d;
}

LazyDeviceAllocator LazyDeviceAllocator::new_owned_host(void *host_ptr,
                                                        size_t size) {
  auto d = LazyDeviceAllocator::new_no_alloc(size);
  assert(host_ptr != NULL);
  d.set_host_ptr_unchecked(host_ptr);
  d.host_state = OWNED_VALID;
  return d;
}

LazyDeviceAllocator LazyDeviceAllocator::new_borrowed_host(void *host_ptr,
                                                           size_t size) {
  auto d = LazyDeviceAllocator::new_no_alloc(size);
  assert(host_ptr != NULL);
  d.set_host_ptr_unchecked(host_ptr);
  d.host_state = BORROWED_VALID;
  return d;
}

LazyDeviceAllocator LazyDeviceAllocator::new_owned_dev(void *dev_ptr,
                                                       size_t size) {
  auto d = LazyDeviceAllocator::new_no_alloc(size);
  assert(dev_ptr != NULL);
  d.set_dev_ptr_unchecked(dev_ptr);
  d.dev_state = OWNED_VALID;
  return d;
}

LazyDeviceAllocator LazyDeviceAllocator::new_borrowed_dev(void *dev_ptr,
                                                          size_t size) {
  auto d = LazyDeviceAllocator::new_no_alloc(size);
  assert(dev_ptr != NULL);
  d.set_dev_ptr_unchecked(dev_ptr);
  d.dev_state = BORROWED_VALID;
  return d;
}

void LazyDeviceAllocator::ensure_on_host() {
  ensure_host_alloced();

  if (host_is_valid()) {
    // Host is already valid, just return
    return;

  } else if (dev_is_valid()) {
    // We have a valid device ptr to copy
    // from, and an alloced but invalid
    // host ptr to copy to.
    copy_to_host_unchecked();
    set_host_valid_unchecked();
    set_dev_invalid_unchecked();
    return;

  } else {
    // We don't have a valid device ptr to
    // copy from, so we just label the host
    // as valid and return
    set_host_valid_unchecked();
    return;
  }
}

void LazyDeviceAllocator::ensure_on_dev() {
  ensure_dev_alloced();

  if (dev_is_valid()) {
    // Device is already valid, just return
    return;

  } else if (host_is_valid()) {
    // We have a valid host ptr to copy
    // from, and an alloced but invalid
    // device ptr to copy to.
    copy_to_dev_unchecked();
    set_dev_valid_unchecked();
    set_host_invalid_unchecked();
    return;

  } else {
    // We don't have a valid device ptr to
    // copy from, so we just label the host
    // as valid and return
    set_dev_valid_unchecked();
    return;
  }
}

void LazyDeviceAllocator::copy_to_host_unchecked() {
  assert(cudaSuccess ==
         cudaMemcpy(host_ptr, dev_ptr, alloc_size, cudaMemcpyDeviceToHost));
}

void LazyDeviceAllocator::copy_to_dev_unchecked() {
  assert(cudaSuccess ==
         cudaMemcpy(dev_ptr, host_ptr, alloc_size, cudaMemcpyHostToDevice));
}

void LazyDeviceAllocator::alloc_host_unchecked() {
  void *ptr = (float *)std::malloc(alloc_size);
  assert(ptr != NULL);
  set_host_ptr_unchecked(ptr);
}

void LazyDeviceAllocator::ensure_host_alloced() {
  if (host_state == NO_ALLOC) {
    alloc_host_unchecked();
    host_state = OWNED_INVALID;
  }
}

void LazyDeviceAllocator::alloc_dev_unchecked() {
  void *ptr;
  auto err = cudaMalloc(&ptr, alloc_size);
  assert(err == cudaSuccess);
  set_dev_ptr_unchecked(ptr);
}

void LazyDeviceAllocator::ensure_dev_alloced() {
  if (dev_state == NO_ALLOC) {
    alloc_dev_unchecked();
    dev_state = OWNED_INVALID;
  }
}

void LazyDeviceAllocator::free_host() {
  if (host_state == OWNED_VALID || host_state == OWNED_INVALID) {
    free(host_ptr);
  }
  host_ptr = NULL;
  host_state = NO_ALLOC;
}

void LazyDeviceAllocator::free_dev() {
  if (dev_state == OWNED_VALID || dev_state == OWNED_INVALID) {
    assert(cudaSuccess == cudaFree(dev_ptr));
  }
  dev_ptr = NULL;
  dev_state = NO_ALLOC;
}

bool LazyDeviceAllocator::host_is_valid() {
  return host_state == OWNED_VALID || host_state == BORROWED_VALID;
}

bool LazyDeviceAllocator::dev_is_valid() {
  return dev_state == OWNED_VALID || dev_state == BORROWED_VALID;
}

void LazyDeviceAllocator::set_alloc_size_unchecked(size_t alloc_size) {
  this->alloc_size = alloc_size;
}

void LazyDeviceAllocator::set_host_ptr_unchecked(void *host_ptr) {
  this->host_ptr = host_ptr;
}

void *LazyDeviceAllocator::get_host_ptr_unchecked() { return host_ptr; }

void *LazyDeviceAllocator::get_host_ptr() {
  ensure_on_host();
  return host_ptr;
}

void LazyDeviceAllocator::set_dev_ptr_unchecked(void *dev_ptr) {
  this->dev_ptr = dev_ptr;
}

void *LazyDeviceAllocator::get_dev_ptr_unchecked() { return dev_ptr; }

void *LazyDeviceAllocator::get_dev_ptr() {
  ensure_on_dev();
  return dev_ptr;
}

size_t LazyDeviceAllocator::get_alloc_size() { return alloc_size; }

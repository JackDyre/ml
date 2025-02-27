#ifndef LAZY_ALLOC_CU
#define LAZY_ALLOC_CU

#include <cassert>
#include <cstdlib>
#include <ctime>

typedef enum DeviceState {
  /*
   * The ptr is owned and the data is
   * on the current device
   */
  OWNED_VALID,
  /*
   * The ptr is borrowed and the data is
   * on the current device
   */
  BORROWED_VALID,
  /*
   * The ptr is owned and the data is
   * not on the current device
   */
  OWNED_INVALID,
  /*
   * The ptr is borrowed and the data is
   * not on the current device
   */
  BORROWED_INVALID,
  /*
   * We do not have any allocated memory
   * on the current device
   */
  NO_ALLOC,
} DeviceState;

/*
 * Helper class that lazily manages memory on the
 * host and device and provides a unified interface
 * for using the same data in both locations
 */
class LazyDeviceAllocator {
private:
  size_t alloc_size;

  void *host_ptr = NULL;
  DeviceState host_state = NO_ALLOC;

  void *dev_ptr = NULL;
  DeviceState dev_state = NO_ALLOC;

  LazyDeviceAllocator() {}

public:
  void set_host_valid_unchecked() {
    if (host_state == OWNED_INVALID) {
      host_state = OWNED_VALID;
    } else if (host_state == BORROWED_INVALID) {
      host_state = BORROWED_VALID;
    }
  }

  void set_host_invalid_unchecked() {
    if (host_state == OWNED_VALID) {
      host_state = OWNED_INVALID;
    } else if (host_state == BORROWED_VALID) {
      host_state = BORROWED_INVALID;
    }
  }

  void set_dev_valid_unchecked() {
    if (dev_state == OWNED_INVALID) {
      dev_state = OWNED_VALID;
    } else if (dev_state == BORROWED_INVALID) {
      dev_state = BORROWED_VALID;
    }
  }

  void set_dev_invalid_unchecked() {
    if (dev_state == OWNED_VALID) {
      dev_state = OWNED_INVALID;
    } else if (dev_state == BORROWED_VALID) {
      dev_state = BORROWED_INVALID;
    }
  }

  void set_host_state_unchecked(DeviceState state) { host_state = state; }
  void set_dev_state_unchecked(DeviceState state) { dev_state = state; }

  LazyDeviceAllocator &operator=(const LazyDeviceAllocator &) = delete;

  ~LazyDeviceAllocator() {
    free_host();
    free_dev();
  }

  static LazyDeviceAllocator new_no_alloc(size_t alloc_size) {
    LazyDeviceAllocator d;
    d.set_alloc_size_unchecked(alloc_size);
    return d;
  }

  static LazyDeviceAllocator new_owned_host(void *host_ptr, size_t size) {
    auto d = LazyDeviceAllocator::new_no_alloc(size);
    assert(host_ptr != NULL);
    d.set_host_ptr_unchecked(host_ptr);
    d.host_state = OWNED_VALID;
    return d;
  }

  static LazyDeviceAllocator new_borrowed_host(void *host_ptr, size_t size) {
    auto d = LazyDeviceAllocator::new_no_alloc(size);
    assert(host_ptr != NULL);
    d.set_host_ptr_unchecked(host_ptr);
    d.host_state = BORROWED_VALID;
    return d;
  }

  static LazyDeviceAllocator new_owned_dev(void *dev_ptr, size_t size) {
    auto d = LazyDeviceAllocator::new_no_alloc(size);
    assert(dev_ptr != NULL);
    d.set_dev_ptr_unchecked(dev_ptr);
    d.dev_state = OWNED_VALID;
    return d;
  }

  static LazyDeviceAllocator new_borrowed_dev(void *dev_ptr, size_t size) {
    auto d = LazyDeviceAllocator::new_no_alloc(size);
    assert(dev_ptr != NULL);
    d.set_dev_ptr_unchecked(dev_ptr);
    d.dev_state = BORROWED_VALID;
    return d;
  }

  void ensure_on_host() {
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

  void ensure_on_dev() {
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

  /*
   * NOTE: The caller is responseible for ensuring
   *       that the host data is not overwritten,
   *       and that both ptrs are allocated. This
   *       method does not set the DeviceState
   *       flags
   */
  void copy_to_host_unchecked() {
    assert(cudaSuccess ==
           cudaMemcpy(host_ptr, dev_ptr, alloc_size, cudaMemcpyDeviceToHost));
  }

  /*
   * NOTE: The caller is responseible for ensuring
   *       that the device data is not overwritten,
   *       and that both ptrs are allocated. This
   *       method does not set the DeviceState
   *       flags
   */
  void copy_to_dev_unchecked() {
    assert(cudaSuccess ==
           cudaMemcpy(dev_ptr, host_ptr, alloc_size, cudaMemcpyHostToDevice));
  }

  /*
   * NOTE: The caller is responsible for ensuring the
   *       overwritten ptr is not leaked. This method
   *       does not set DeviceState flags
   */
  void alloc_host_unchecked() {
    void *ptr = (float *)std::malloc(alloc_size);
    assert(ptr != NULL);
    set_host_ptr_unchecked(ptr);
  }

  void ensure_host_alloced() {
    if (host_state == NO_ALLOC) {
      alloc_host_unchecked();
      host_state = OWNED_INVALID;
    }
  }

  /*
   * NOTE: The caller is responsible for ensuring the
   *       overwritten ptr is not leaked. This method
   *       does not set DeviceState flags
   */
  void alloc_dev_unchecked() {
    void *ptr;
    auto err = cudaMalloc(&ptr, alloc_size);
    assert(err == cudaSuccess);
    set_dev_ptr_unchecked(ptr);
  }

  void ensure_dev_alloced() {
    if (dev_state == NO_ALLOC) {
      alloc_dev_unchecked();
      dev_state = OWNED_INVALID;
    }
  }

  void free_host() {
    if (host_state == OWNED_VALID || host_state == OWNED_INVALID) {
      free(host_ptr);
    }
    host_ptr = NULL;
    host_state = NO_ALLOC;
  }

  void free_dev() {
    if (dev_state == OWNED_VALID || dev_state == OWNED_INVALID) {
      assert(cudaSuccess == cudaFree(dev_ptr));
    }
    dev_ptr = NULL;
    dev_state = NO_ALLOC;
  }

  bool host_is_valid() {
    return host_state == OWNED_VALID || host_state == BORROWED_VALID;
  }

  bool dev_is_valid() {
    return dev_state == OWNED_VALID || dev_state == BORROWED_VALID;
  }

  /*
   * NOTE: The caller is responsible for ensuring the
   *       alloc size is not changed while any of the
   *       pointers are valid
   */
  void set_alloc_size_unchecked(size_t alloc_size) {
    this->alloc_size = alloc_size;
  }

  /*
   * NOTE: The caller is responsible for ensuring the
   *       ptr is the same size as the alloc size and
   *       that the overwritten ptr is not leaked. This
   *       method does not set DeviceState flags
   */
  void set_host_ptr_unchecked(void *host_ptr) { this->host_ptr = host_ptr; }

  /*
   * NOTE: The caller is responsible for ensuring the
   *       ptr is valid
   */
  void *get_host_ptr_unchecked() { return host_ptr; }

  void *get_host_ptr() {
    ensure_on_host();
    return host_ptr;
  }

  /*
   * NOTE: The caller is responsible for ensuring the
   *       ptr is the same size as the alloc size and
   *       that the overwritten ptr is not leaked. This
   *       method does not set DeviceState flags
   */
  void set_dev_ptr_unchecked(void *dev_ptr) { this->dev_ptr = dev_ptr; }

  /*
   * NOTE: The caller is responsible for ensuring the
   *       ptr is valid
   */
  void *get_dev_ptr_unchecked() { return dev_ptr; }

  void *get_dev_ptr() {
    ensure_on_dev();
    return dev_ptr;
  }

  size_t get_alloc_size() { return alloc_size; }
};

#endif // LAZY_ALLOC_CU

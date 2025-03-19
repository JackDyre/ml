#ifndef LAZY_ALLOC_HPP
#define LAZY_ALLOC_HPP

#include <cstddef>

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
  void set_host_valid_unchecked();
  void set_host_invalid_unchecked();
  void set_dev_valid_unchecked();
  void set_dev_invalid_unchecked();
  void set_host_state_unchecked(DeviceState state);
  void set_dev_state_unchecked(DeviceState state);

  LazyDeviceAllocator &operator=(const LazyDeviceAllocator &) = delete;

  ~LazyDeviceAllocator();

  static LazyDeviceAllocator new_no_alloc(size_t alloc_size);
  static LazyDeviceAllocator new_owned_host(void *host_ptr, size_t size);
  static LazyDeviceAllocator new_borrowed_host(void *host_ptr, size_t size);
  static LazyDeviceAllocator new_owned_dev(void *dev_ptr, size_t size);
  static LazyDeviceAllocator new_borrowed_dev(void *dev_ptr, size_t size);

  void ensure_on_host();
  void ensure_on_dev();
  void copy_to_host_unchecked();
  void copy_to_dev_unchecked();
  void alloc_host_unchecked();
  void ensure_host_alloced();
  void alloc_dev_unchecked();
  void ensure_dev_alloced();
  void free_host();
  void free_dev();
  bool host_is_valid();
  bool dev_is_valid();
  void set_alloc_size_unchecked(size_t alloc_size);
  void set_host_ptr_unchecked(void *host_ptr);
  void *get_host_ptr_unchecked();
  void *get_host_ptr();
  void set_dev_ptr_unchecked(void *dev_ptr);
  void *get_dev_ptr_unchecked();
  void *get_dev_ptr();
  size_t get_alloc_size();
};

#endif // LAZY_ALLOC_HPP
#pragma once

#include "tensorrt_flow/def/arch.hpp"

#include <atomic>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>

namespace tensorrt_flow {

namespace memory {

using MemoryDeleteFunction = std::function<void(void*, size_t)>;

class MemoryInterface {
public:
  MemoryInterface() { use_count_ = new std::atomic<int>(1); }

  virtual ~MemoryInterface() {
    if (use_count_ == nullptr) { return; }

    int expect_value = use_count_->load();

    if (expect_value == 1) {
      // destroy
      delete use_count_;
      return;
    }

    int new_value = expect_value - 1;
    while (!use_count_->compare_exchange_strong(expect_value, new_value, std::memory_order_release, std::memory_order_relaxed)) {}

    // if two more thread run compare_exchange_strong at the same time
    if (expect_value == 1) { delete use_count_; }
  }

  MemoryInterface(const MemoryInterface& other) {
    if (this != &other) {
      use_count_ = other.use_count_;
      use_count_->fetch_add(1);
    }
  }

  MemoryInterface(MemoryInterface&& other) noexcept {
    if (this != &other) {
      use_count_ = other.use_count_;
      use_count_->fetch_add(1);
    }
  }

  MemoryInterface& operator=(const MemoryInterface& other) {
    if (this != &other) {
      use_count_ = other.use_count_;
      use_count_->fetch_add(1);
    }
    return *this;
  }

  MemoryInterface& operator=(MemoryInterface&& other) noexcept {
    if (this != &other) {
      use_count_ = other.use_count_;
      use_count_->fetch_add(1);
    }
    return *this;
  }


  virtual void* host_ptr()   = 0;
  virtual void* device_ptr() = 0;

  virtual bool sync_to_host()   = 0;
  virtual bool sync_to_device() = 0;

  virtual bool sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device)   = 0;
  virtual bool sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) = 0;

  virtual bool sync_to_host(cudaStream_t& stream)   = 0;
  virtual bool sync_to_device(cudaStream_t& stream) = 0;

  virtual bool sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream)   = 0;
  virtual bool sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) = 0;

protected:
  bool is_last_object() { return use_count_->load(std::memory_order_relaxed) == 1; }

private:
  std::atomic<int>* use_count_{nullptr};
};

}   // namespace memory

}   // namespace tensorrt_flow
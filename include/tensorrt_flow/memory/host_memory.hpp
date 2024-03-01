#pragma once

#include "tensorrt_flow/memory/memory_interface.hpp"

#include <functional>
#include <iostream>
namespace tensorrt_flow {

namespace memory {

class HostMemory : public MemoryInterface {
public:
  HostMemory(size_t memory_size, bool is_pinned = false, size_t align_size = 64);
  HostMemory(void* memory_ptr, size_t memory_size, MemoryDeleteFunction deleter);

  virtual ~HostMemory();

  virtual void* host_ptr() override { return memory_ptr_; }
  virtual void* device_ptr() override { return nullptr; }

  virtual bool sync_to_host() override { return true; }
  virtual bool sync_to_device() override { return false; }

  virtual bool sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) override;
  virtual bool sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) override { return false; }

  virtual bool sync_to_host(cudaStream_t& stream) override { return true; }
  virtual bool sync_to_device(cudaStream_t& stream) override { return false; }

  virtual bool sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) override;
  virtual bool sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) override { return false; }

private:
  void*                memory_ptr_{nullptr};
  size_t               memory_size_{0};
  MemoryDeleteFunction deleter_function_{nullptr};
};

}   // namespace memory

}   // namespace tensorrt_flow
#pragma once

#include "tensorrt_flow/memory/memory_interface.hpp"

#include <functional>
#include <iostream>

namespace tensorrt_flow {

namespace memory {

class DeviceMemory : public MemoryInterface {
public:
  DeviceMemory(size_t memory_size, bool is_unified = false);
  DeviceMemory(void* memory_ptr, size_t memory_size, MemoryDeleteFunction deleter, bool is_unified = false);

  virtual ~DeviceMemory();

  virtual void* host_ptr() override { return is_unified_ ? memory_ptr_ : nullptr; }
  virtual void* device_ptr() override { return memory_ptr_; }

  virtual bool sync_to_host() override { return true; }
  virtual bool sync_to_device() override { return true; }

  virtual bool sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) override;
  virtual bool sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) override;

  virtual bool sync_to_host(cudaStream_t& stream) override { return true; }
  virtual bool sync_to_device(cudaStream_t& stream) override { return true; }


  virtual bool sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) override;
  virtual bool sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) override;

private:
  void*                memory_ptr_{nullptr};
  size_t               memory_size_{0};
  MemoryDeleteFunction deleter_function_{nullptr};
  bool                 is_unified_{false};
};


}   // namespace memory
}   // namespace tensorrt_flow
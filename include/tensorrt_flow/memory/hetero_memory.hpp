#pragma once

#include "tensorrt_flow/memory/memory_interface.hpp"
namespace tensorrt_flow {

namespace memory {

class HeteroMemory : public MemoryInterface {
public:
  HeteroMemory(size_t memory_size, bool is_pinned = false, bool is_unified = false, size_t align_size = 64);
  HeteroMemory(void* cpu_memory_ptr, MemoryDeleteFunction cpu_deleter_function, void* gpu_memory_ptr, MemoryDeleteFunction gpu_deleter_function, size_t memory_size, bool is_unified = false);

  virtual ~HeteroMemory();

  virtual void* host_ptr() override { return is_unified_ ? gpu_memory_ptr_ : cpu_memory_ptr_; }
  virtual void* device_ptr() override { return gpu_memory_ptr_; }

  virtual bool sync_to_host() override;
  virtual bool sync_to_device() override;

  virtual bool sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) override;
  virtual bool sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) override;

  virtual bool sync_to_host(cudaStream_t& stream) override;
  virtual bool sync_to_device(cudaStream_t& stream) override;

  virtual bool sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) override;
  virtual bool sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) override;

private:
  void*  cpu_memory_ptr_{nullptr};
  void*  gpu_memory_ptr_{nullptr};
  size_t memory_size_{0};
  bool   is_unified_{false};

  MemoryDeleteFunction cpu_deleter_function_{nullptr};
  MemoryDeleteFunction gpu_deleter_function_{nullptr};
};
}   // namespace memory
}   // namespace tensorrt_flow
#include "tensorrt_flow/memory/host_memory.hpp"

#include "tensorrt_flow/cuda/cuda_helper.hpp"

#include <assert.h>
#include <cstdlib>

namespace tensorrt_flow {

namespace memory {

HostMemory::HostMemory(size_t memory_size, bool is_pinned, size_t align_size)
  : memory_size_(memory_size) {
  if (!is_pinned) {
    memory_ptr_       = aligned_alloc(align_size, memory_size);
    deleter_function_ = [](void* ptr, size_t size) { free(ptr); };
  } else {
    CUDA_CHECK(cudaMallocHost((void**)&memory_ptr_, memory_size));
    deleter_function_ = [](void* ptr, size_t size) { CUDA_CHECK(cudaFreeHost(ptr)); };
  }
}

HostMemory::HostMemory(void* memory_ptr, size_t memory_size, MemoryDeleteFunction deleter)
  : memory_ptr_(memory_ptr)
  , memory_size_(memory_size)
  , deleter_function_(deleter) {}


bool HostMemory::sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) {
  assert(memory_size <= memory_size_);
  assert(arch_device != arch::ArchDevice::Unkonw);
  if (arch_device == arch::ArchDevice::CPU) {
    std::memcpy(memory_ptr_, memory_ptr, memory_size);
  } else if (arch_device == arch::ArchDevice::GPU) {
    CUDA_CHECK(cudaMemcpy(memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToHost));
  } else {
    assert(false);
  }
  return true;
}

bool HostMemory::sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) {
  assert(memory_size <= memory_size_);
  assert(arch_device != arch::ArchDevice::Unkonw);
  if (arch_device == arch::ArchDevice::CPU) {
    std::memcpy(memory_ptr_, memory_ptr, memory_size);
  } else if (arch_device == arch::ArchDevice::GPU) {
    CUDA_CHECK(cudaMemcpyAsync(memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToHost, stream));
  } else {
    assert(false);
  }
  return true;
}


HostMemory::~HostMemory() {
  if (is_last_object()) { deleter_function_(memory_ptr_, memory_size_); }
}

}   // namespace memory
}   // namespace tensorrt_flow
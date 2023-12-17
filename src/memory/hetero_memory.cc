#include "tensorrt_flow/memory/hetero_memory.hpp"

#include "tensorrt_flow/cuda/cuda_helper.hpp"

#include <assert.h>
namespace tensorrt_flow {

namespace memory {
HeteroMemory::HeteroMemory(size_t memory_size, bool is_pinned, bool is_unified, size_t align_size)
  : memory_size_(memory_size)
  , is_unified_(is_unified) {
  if (is_unified) {
    CUDA_CHECK(cudaMallocManaged((void**)&gpu_memory_ptr_, memory_size));
    gpu_deleter_function_ = [](void* ptr, size_t size) { CUDA_CHECK(cudaFreeHost(ptr)); };
    cpu_deleter_function_ = [](void* ptr, size_t size) {};
    return;
  }

  CUDA_CHECK(cudaMalloc((void**)&gpu_memory_ptr_, memory_size));
  gpu_deleter_function_ = [](void* ptr, size_t size) { CUDA_CHECK(cudaFree(ptr)); };

  if (is_pinned) {
    CUDA_CHECK(cudaMallocHost((void**)&cpu_memory_ptr_, memory_size));
    cpu_deleter_function_ = [](void* ptr, size_t size) { CUDA_CHECK(cudaFreeHost(ptr)); };
  } else {
    cpu_memory_ptr_       = aligned_alloc(align_size, memory_size);
    cpu_deleter_function_ = [](void* ptr, size_t size) { free(ptr); };
  }
}

HeteroMemory::HeteroMemory(void* cpu_memory_ptr, MemoryDeleteFunction cpu_deleter_function, void* gpu_memory_ptr, MemoryDeleteFunction gpu_deleter_function, size_t memory_size, bool is_unified)
  : cpu_memory_ptr_(cpu_memory_ptr)
  , gpu_memory_ptr_(gpu_memory_ptr)
  , memory_size_(memory_size)
  , is_unified_(is_unified)
  , cpu_deleter_function_(gpu_deleter_function)
  , gpu_deleter_function_(gpu_deleter_function) {}



HeteroMemory::~HeteroMemory() {
  if (is_last_object()) {
    if (cpu_deleter_function_) { cpu_deleter_function_(cpu_memory_ptr_, memory_size_); }
    if (gpu_deleter_function_) { gpu_deleter_function_(gpu_memory_ptr_, memory_size_); }
  }
}

bool HeteroMemory::sync_to_host() {
  if (is_unified_) { return true; }
  CUDA_CHECK(cudaMemcpy(cpu_memory_ptr_, gpu_memory_ptr_, memory_size_, cudaMemcpyDeviceToHost));
  return true;
}

bool HeteroMemory::sync_to_device() {
  if (is_unified_) { return true; }
  CUDA_CHECK(cudaMemcpy(gpu_memory_ptr_, cpu_memory_ptr_, memory_size_, cudaMemcpyHostToDevice));
  return true;
}

bool HeteroMemory::sync_to_host(cudaStream_t& stream) {
  if (is_unified_) { return true; }
  CUDA_CHECK(cudaMemcpyAsync(cpu_memory_ptr_, gpu_memory_ptr_, memory_size_, cudaMemcpyDeviceToHost, stream));
  return true;
}

bool HeteroMemory::sync_to_device(cudaStream_t& stream) {
  if (is_unified_) { return true; }
  CUDA_CHECK(cudaMemcpyAsync(gpu_memory_ptr_, cpu_memory_ptr_, memory_size_, cudaMemcpyHostToDevice, stream));
  return true;
}


bool HeteroMemory::sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) {
  assert(arch_device != arch::ArchDevice::Unkonw);
  assert(memory_size <= memory_size_);

  if (is_unified_) {
    assert(memory_size <= memory_size_);
    if (arch_device == arch::ArchDevice::CPU) {
      std::memcpy(gpu_memory_ptr_, memory_ptr, memory_size);
    } else if (arch_device == arch::ArchDevice::GPU) {
      CUDA_CHECK(cudaMemcpy(gpu_memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToHost));
    } else {
      assert(false);
    }
    return true;
  }

  if (arch_device == arch::ArchDevice::CPU) {
    std::memcpy(cpu_memory_ptr_, memory_ptr, memory_size);
  } else if (arch_device == arch::ArchDevice::GPU) {
    CUDA_CHECK(cudaMemcpy(cpu_memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToHost));
  } else {
    assert(false);
  }
  return true;
}

bool HeteroMemory::sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) {
  assert(arch_device != arch::ArchDevice::Unkonw);
  assert(memory_size <= memory_size_);
  if (arch_device == arch::ArchDevice::CPU) {
    CUDA_CHECK(cudaMemcpy(gpu_memory_ptr_, memory_ptr, memory_size, cudaMemcpyHostToDevice));
  } else if (arch_device == arch::ArchDevice::GPU) {
    CUDA_CHECK(cudaMemcpy(gpu_memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToDevice));
  } else {
    assert(false);
  }
  return true;
}


bool HeteroMemory::sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) {
  assert(arch_device != arch::ArchDevice::Unkonw);
  assert(memory_size <= memory_size_);

  if (is_unified_) {
    assert(memory_size <= memory_size_);
    if (arch_device == arch::ArchDevice::CPU) {
      std::memcpy(gpu_memory_ptr_, memory_ptr, memory_size);
    } else if (arch_device == arch::ArchDevice::GPU) {
      CUDA_CHECK(cudaMemcpyAsync(gpu_memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToHost, stream));
    } else {
      assert(false);
    }
    return true;
  }

  if (arch_device == arch::ArchDevice::CPU) {
    std::memcpy(cpu_memory_ptr_, memory_ptr, memory_size);
  } else if (arch_device == arch::ArchDevice::GPU) {
    CUDA_CHECK(cudaMemcpyAsync(cpu_memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToHost, stream));
  } else {
    assert(false);
  }
  return true;
}

bool HeteroMemory::sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) {
  assert(arch_device != arch::ArchDevice::Unkonw);
  assert(memory_size <= memory_size_);
  if (arch_device == arch::ArchDevice::CPU) {
    CUDA_CHECK(cudaMemcpyAsync(gpu_memory_ptr_, memory_ptr, memory_size, cudaMemcpyHostToDevice, stream));
  } else if (arch_device == arch::ArchDevice::GPU) {
    CUDA_CHECK(cudaMemcpyAsync(gpu_memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    assert(false);
  }
  return true;
}



}   // namespace memory
}   // namespace tensorrt_flow
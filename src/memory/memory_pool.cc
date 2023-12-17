#include "tensorrt_flow/memory/memory_pool.hpp"

#include "tensorrt_flow/cuda/cuda_helper.hpp"

#include <list>
#include <shared_mutex>
#include <unordered_map>
namespace tensorrt_flow {

namespace memory {


void* MemoryPool::GetMemoryPtrImp(std::shared_mutex& global_map_mutex, std::unordered_map<size_t, std::unique_ptr<std::mutex>>& memory_node_mutexs,
                                  std::unordered_map<size_t, std::list<void*>>& memory_nodes, size_t memory_size, MallocFunction function_) {
  bool is_has_node = true;
  {
    std::shared_lock<std::shared_mutex> lock(global_map_mutex);
    if (memory_nodes.find(memory_size) == memory_nodes.end()) { is_has_node; }
  }

  // double check
  if (!is_has_node) {
    std::unique_lock<std::shared_mutex> lock(global_map_mutex);
    if (memory_nodes.find(memory_size) == memory_nodes.end()) {
      memory_nodes.insert(std::make_pair(memory_size, std::list<void*>()));
      memory_node_mutexs.insert(std::make_pair(memory_size, std::make_unique<std::mutex>()));
    }
  }

  std::lock_guard<std::mutex> lk(*memory_node_mutexs[memory_size]);
  auto&                       mem_node_list = memory_nodes[memory_size];
  if (!mem_node_list.empty()) {
    void* reuslt_ptr = mem_node_list.front();
    mem_node_list.pop_front();
    return reuslt_ptr;
  }

  return function_(memory_size);
}



bool MemoryPool::ReturnMemoryPtrImp(std::shared_mutex& global_map_mutex, std::unordered_map<size_t, std::unique_ptr<std::mutex>>& memory_node_mutexs,
                                    std::unordered_map<size_t, std::list<void*>>& memory_nodes, void* memory_ptr, size_t memory_size) {
  std::lock_guard<std::mutex> lk(*memory_node_mutexs[memory_size]);
  auto&                       mem_node_list = memory_nodes[memory_size];
  // Inserting the mem_node at the Head
  mem_node_list.push_front(memory_ptr);
  return true;
}


bool MemoryPool::ReturnHostMemory(void* memory_ptr, size_t memory_size, bool is_pinned) {
  if (!is_pinned) {
    return ReturnMemoryPtrImp(host_memory_mutex_, host_memory_mutexs_, host_memory_, memory_ptr, memory_size);
  } else {
    return ReturnMemoryPtrImp(host_pinned_memory_mutex_, host_pinned_memory_mutexs_, host_pinned_memory_, memory_ptr, memory_size);
  }
}
bool MemoryPool::ReturnDeviceMemory(void* memory_ptr, size_t memory_size, bool is_unified) {
  if (!is_unified) {
    return ReturnMemoryPtrImp(device_memory_mutex_, device_memory_mutexs_, device_memory_, memory_ptr, memory_size);
  } else {
    return ReturnMemoryPtrImp(device_unified_memory_mutex_, device_unified_memory_mutexs_, device_unified_memory_, memory_ptr, memory_size);
  }
}

bool MemoryPool::ReturnHeteroMemory(void* cpu_memory_ptr, void* gpu_memory_ptr, size_t memory_size, bool is_pinned, bool is_unified) {

  bool is_success = true;

  if (!is_unified) {
    is_success &= ReturnMemoryPtrImp(device_memory_mutex_, device_memory_mutexs_, device_memory_, gpu_memory_ptr, memory_size);
    if (!is_pinned) {
      is_success &= ReturnMemoryPtrImp(host_memory_mutex_, host_memory_mutexs_, host_memory_, cpu_memory_ptr, memory_size);
    } else {
      is_success &= ReturnMemoryPtrImp(host_pinned_memory_mutex_, host_pinned_memory_mutexs_, host_pinned_memory_, cpu_memory_ptr, memory_size);
    }
  } else {
    is_success &= ReturnMemoryPtrImp(device_unified_memory_mutex_, device_unified_memory_mutexs_, device_unified_memory_, gpu_memory_ptr, memory_size);
  }
  return is_success;
}

HostMemory MemoryPool::GetHostMemory(size_t memory_size, bool is_pinned) {
  void* ptr = nullptr;
  if (!is_pinned) {
    ptr = GetMemoryPtrImp(host_memory_mutex_, host_memory_mutexs_, host_memory_, memory_size, [](size_t memory_size) { return std::malloc(memory_size); });
  } else {
    ptr = GetMemoryPtrImp(host_pinned_memory_mutex_, host_pinned_memory_mutexs_, host_pinned_memory_, memory_size, [](size_t memory_size) {
      void* mem_ptr = nullptr;
      CUDA_CHECK(cudaMallocHost((void**)&mem_ptr, memory_size));
      return mem_ptr;
    });
  }
  return HostMemory(ptr, memory_size, [this, is_pinned](void* mem_ptr, size_t mem_size) { ReturnHostMemory(mem_ptr, mem_size, is_pinned); });
}

// TODO
DeviceMemory MemoryPool::GetDeviceMemory(size_t memory_size, bool is_unified) {
  void* ptr = nullptr;
  if (is_unified) {
    ptr = GetMemoryPtrImp(device_unified_memory_mutex_, device_unified_memory_mutexs_, device_unified_memory_, memory_size, [](size_t memory_size) {
      void* mem_ptr = nullptr;
      CUDA_CHECK(cudaMallocManaged((void**)&mem_ptr, memory_size));
      return mem_ptr;
    });
  } else {
    ptr = GetMemoryPtrImp(device_memory_mutex_, device_memory_mutexs_, device_memory_, memory_size, [](size_t memory_size) {
      void* mem_ptr = nullptr;
      CUDA_CHECK(cudaMalloc((void**)&mem_ptr, memory_size));
      return mem_ptr;
    });
  }
  return DeviceMemory(ptr, memory_size, [this, is_unified](void* mem_ptr, size_t mem_size) { ReturnDeviceMemory(mem_ptr, mem_size, is_unified); });
}

HeteroMemory MemoryPool::GetHeteroMemory(size_t memory_size, bool is_pinned, bool is_unified) {
  void* cpu_ptr = nullptr;
  void* gpu_ptr = nullptr;
  if (!is_unified) {
    gpu_ptr = GetMemoryPtrImp(device_unified_memory_mutex_, device_unified_memory_mutexs_, device_unified_memory_, memory_size, [](size_t memory_size) {
      void* mem_ptr = nullptr;
      CUDA_CHECK(cudaMallocManaged((void**)&mem_ptr, memory_size));
      return mem_ptr;
    });
    return HeteroMemory(
      cpu_ptr, [this, is_pinned](void* mem_ptr, size_t mem_size) {}, gpu_ptr, [this, is_unified](void* mem_ptr, size_t mem_size) { ReturnDeviceMemory(mem_ptr, mem_size, is_unified); }, memory_size,
      is_unified);
  }

  if (!is_pinned) {
    cpu_ptr = GetMemoryPtrImp(host_memory_mutex_, host_memory_mutexs_, host_memory_, memory_size, [](size_t memory_size) { return std::malloc(memory_size); });
  } else {
    cpu_ptr = GetMemoryPtrImp(host_pinned_memory_mutex_, host_pinned_memory_mutexs_, host_pinned_memory_, memory_size, [](size_t memory_size) {
      void* mem_ptr = nullptr;
      CUDA_CHECK(cudaMallocHost((void**)&mem_ptr, memory_size));
      return mem_ptr;
    });
  }

  gpu_ptr = GetMemoryPtrImp(device_memory_mutex_, device_memory_mutexs_, device_memory_, memory_size, [](size_t memory_size) {
    void* mem_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&mem_ptr, memory_size));
    return mem_ptr;
  });

  return HeteroMemory(
    cpu_ptr, [this, is_pinned](void* mem_ptr, size_t mem_size) { ReturnHostMemory(mem_ptr, mem_size, is_pinned); }, gpu_ptr,
    [this, is_unified](void* mem_ptr, size_t mem_size) { ReturnDeviceMemory(mem_ptr, mem_size, is_unified); }, memory_size, is_unified);
}



MemoryPool::~MemoryPool() {
  for (auto& key : host_memory_) {
    for (auto& ptr : key.second) { std::free(ptr); }
  }

  for (auto& key : host_pinned_memory_) {
    for (auto& ptr : key.second) { CUDA_CHECK(cudaFreeHost(ptr)); }
  }

  for (auto& key : device_memory_) {
    for (auto& ptr : key.second) { CUDA_CHECK(cudaFree(ptr)); }
  }

  for (auto& key : device_unified_memory_) {
    for (auto& ptr : key.second) { CUDA_CHECK(cudaFree(ptr)); }
  }
}

}   // namespace memory
}   // namespace tensorrt_flow
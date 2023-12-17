#pragma once

#include "tensorrt_flow/memory/device_memory.hpp"
#include "tensorrt_flow/memory/hetero_memory.hpp"
#include "tensorrt_flow/memory/host_memory.hpp"

#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace tensorrt_flow {

namespace memory {


class MemoryPool {
  static constexpr size_t kMaxHostMemoryNum          = 10;
  static constexpr size_t kMaxHostPinnedMemoryNum    = 10;
  static constexpr size_t kMaxDeviceMemoryNum        = 10;
  static constexpr size_t kMaxDeviceUnifiedMemoryNum = 10;

  using MallocFunction = std::function<void*(size_t)>;

public:
  MemoryPool() = default;
  ~MemoryPool();

  HostMemory   GetHostMemory(size_t memory_size, bool is_pinned = false);
  DeviceMemory GetDeviceMemory(size_t memory_size, bool is_unified = false);
  HeteroMemory GetHeteroMemory(size_t memory_size, bool is_pinned = false, bool is_unified = false);


  bool ReturnHostMemory(void* memory_ptr, size_t memory_size, bool is_pinned = false);
  bool ReturnDeviceMemory(void* memory_ptr, size_t memory_size, bool is_unified = false);
  bool ReturnHeteroMemory(void* cpu_memory_ptr, void* gpu_memory_ptr, size_t memory_size, bool is_pinned = false, bool is_unified = false);

private:
  std::shared_mutex                                       host_memory_mutex_;
  std::unordered_map<size_t, std::unique_ptr<std::mutex>> host_memory_mutexs_;
  std::unordered_map<size_t, std::list<void*>>            host_memory_;

  std::shared_mutex                                       host_pinned_memory_mutex_;
  std::unordered_map<size_t, std::unique_ptr<std::mutex>> host_pinned_memory_mutexs_;
  std::unordered_map<size_t, std::list<void*>>            host_pinned_memory_;

  std::shared_mutex                                       device_memory_mutex_;
  std::unordered_map<size_t, std::unique_ptr<std::mutex>> device_memory_mutexs_;
  std::unordered_map<size_t, std::list<void*>>            device_memory_;

  std::shared_mutex                                       device_unified_memory_mutex_;
  std::unordered_map<size_t, std::unique_ptr<std::mutex>> device_unified_memory_mutexs_;
  std::unordered_map<size_t, std::list<void*>>            device_unified_memory_;

  void* GetMemoryPtrImp(std::shared_mutex&, std::unordered_map<size_t, std::unique_ptr<std::mutex>>&, std::unordered_map<size_t, std::list<void*>>&, size_t, MallocFunction);
  bool  ReturnMemoryPtrImp(std::shared_mutex&, std::unordered_map<size_t, std::unique_ptr<std::mutex>>&, std::unordered_map<size_t, std::list<void*>>&, void*, size_t);
};


}   // namespace memory
}   // namespace tensorrt_flow
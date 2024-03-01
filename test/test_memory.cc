#include "tensorrt_flow/memory/device_memory.hpp"
#include "tensorrt_flow/memory/host_memory.hpp"

#include <benchmark/benchmark.h>

static void BM_MemoryExternalCopy(benchmark::State& state) {
  size_t memory_size = 1024;
  void*  molloc_ptr  = std::malloc(memory_size);

  tensorrt_flow::memory::HostMemory host_memory{molloc_ptr, memory_size, [](void* ptr, size_t size) { std::free(ptr); }};

  for (auto _ : state) { tensorrt_flow::memory::HostMemory host_memory_temp = host_memory; }
}

BENCHMARK(BM_MemoryExternalCopy);

static void BM_MemoryInternalCopy(benchmark::State& state) {
  size_t memory_size = 1024;

  tensorrt_flow::memory::HostMemory host_memory(memory_size);

  for (auto _ : state) { tensorrt_flow::memory::HostMemory host_memory_temp = host_memory; }
}
BENCHMARK(BM_MemoryInternalCopy);


static void BM_MemoryInternalPinnedCopy(benchmark::State& state) {
  size_t memory_size = 1024;

  tensorrt_flow::memory::HostMemory host_memory(memory_size, true, 64);

  for (auto _ : state) { tensorrt_flow::memory::HostMemory host_memory_temp = host_memory; }
}
BENCHMARK(BM_MemoryInternalPinnedCopy);

static void BM_DeviceMemoryInternalCopy(benchmark::State& state) {
  size_t memory_size = 1024;

  tensorrt_flow::memory::DeviceMemory device_memory(memory_size, false);

  for (auto _ : state) { tensorrt_flow::memory::DeviceMemory device_memory_temp = device_memory; }
}
BENCHMARK(BM_DeviceMemoryInternalCopy);

static void BM_DeviceMemoryExternalCopy(benchmark::State& state) {
  size_t memory_size = 1024;
  void*  memory_ptr{nullptr};
  cudaMalloc((void**)&memory_ptr, memory_size);
  tensorrt_flow::memory::DeviceMemory device_memory(
    memory_ptr, memory_size, [](void* ptr, size_t size) { cudaFree(ptr); }, false);

  for (auto _ : state) { tensorrt_flow::memory::DeviceMemory device_memory_temp = device_memory; }
}
BENCHMARK(BM_DeviceMemoryExternalCopy);



BENCHMARK_MAIN();

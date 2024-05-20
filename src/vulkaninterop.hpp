#include <string>

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <tl/expected.hpp>

#include <core.hpp>

#ifdef _WIN64
#include <Windows.h>
#include <VersionHelpers.h>
#endif

#include <cuda.h>
#include <span>

#ifndef _WIN64
using HANDLE = int;
#endif

struct VulkanCudaCreateInfo {
	uint32_t width;
	uint32_t height;
    HANDLE inputSemaphoreHandle;
    HANDLE outputSemaphoreHandle;
    HANDLE bufferMemoryHandle;
    size_t bufferMemorySize;
};

struct ExternalBuffer {
	cudaExternalMemory_t bufferMemory;
	size_t bufferSize;
	CUdeviceptr dataPtr;
};

class VulkanCuda {
private:
	std::unique_ptr<Core> core;
	uint32_t width;
	uint32_t height;
	ExternalBuffer externalBuffer;
	cudaExternalSemaphore_t inputSemaphore;
	cudaExternalSemaphore_t outputSemaphore;
public:
	VulkanCuda(cudaExternalSemaphore_t inputSemaphore, cudaExternalSemaphore_t outputSemaphore, ExternalBuffer externalBuffer, size_t externalBufferSize, uint32_t width, uint32_t height);
	~VulkanCuda();
	static tl::expected<std::unique_ptr<VulkanCuda>, std::string> create(VulkanCudaCreateInfo info);

	void setDarkCorrection(std::span<unsigned short> darkMap, unsigned short offset);
	void setGainCorrection(std::span<unsigned short> gainMap);
	void setDefectCorrection(std::span<unsigned short> defectMap);
};

#ifdef _WIN64
tl::expected<cudaExternalSemaphore_t, cudaError_t> cudaVKImportSemaphore(HANDLE handle);
tl::expected<std::pair<cudaExternalMemory_t, CUdeviceptr>, cudaError_t> cudaVKImportMemory(HANDLE handle, size_t size);
#else
void cudaVKImportSemaphore(int handle, cudaExternalSemaphore_t extSemaphore);
void cudaVKImportImageMem(int handle, cudaExternalMemory_t extMemory);
#endif

tl::expected<void, cudaError_t> cudaVKSemaphoreSignal(cudaExternalSemaphore_t extSemaphore, cudaStream_t stream);
tl::expected<void, cudaError_t> cudaVKSemaphoreWait(cudaExternalSemaphore_t extSemaphore, cudaStream_t stream);
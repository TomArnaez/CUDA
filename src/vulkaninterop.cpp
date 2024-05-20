#include <vulkaninterop.hpp>

VulkanCuda::VulkanCuda(
    cudaExternalSemaphore_t inputSemaphore, 
    cudaExternalSemaphore_t outputSemaphore, 
    ExternalBuffer externalBuffer, 
    size_t externalBufferSize,
    uint32_t width,
    uint32_t height
) : 
    inputSemaphore(inputSemaphore), outputSemaphore(outputSemaphore), 
    externalBuffer(externalBuffer),
    width(width), height(height) {
}

VulkanCuda::~VulkanCuda() {
    cudaDestroyExternalSemaphore(inputSemaphore);
    cudaDestroyExternalSemaphore(outputSemaphore);
}

tl::expected<std::unique_ptr<VulkanCuda>, std::string> VulkanCuda::create(VulkanCudaCreateInfo info) {
    auto inputSemaphore = cudaVKImportSemaphore(info.inputSemaphoreHandle);
    if (!inputSemaphore.has_value())
        return tl::unexpected("Failed to import input semaphore");
    
    auto outputSemaphore = cudaVKImportSemaphore(info.outputSemaphoreHandle);
    if (!outputSemaphore.has_value())
        return tl::unexpected("Faild to import output semaphore");

    auto externalMemoryPair = cudaVKImportMemory(info.bufferMemoryHandle, info.bufferMemorySize);
    if (!externalMemoryPair.has_value())
        return tl::unexpected("Failed to import external memory");

    ExternalBuffer externalBuffer {
        .bufferMemory = externalMemoryPair.value().first,
        .bufferSize = info.bufferMemorySize,
        .dataPtr = externalMemoryPair.value().second
    };

    return std::make_unique<VulkanCuda>(new VulkanCuda (
        inputSemaphore.value(),
        outputSemaphore.value(),
        externalBuffer,
        info.bufferMemorySize,
        info.width,
        info.height
    ));
}

#ifdef _WIN64
tl::expected<cudaExternalSemaphore_t, cudaError_t> cudaVKImportSemaphore(HANDLE handle) {
	cudaExternalSemaphoreHandleDesc extSemaphoreHandleDesc;
	memset(&extSemaphoreHandleDesc, 0, sizeof(extSemaphoreHandleDesc));

	extSemaphoreHandleDesc.type =
		IsWindows8OrGreater() ? cudaExternalSemaphoreHandleTypeOpaqueWin32
		: cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
	extSemaphoreHandleDesc.handle.win32.handle = handle;
	extSemaphoreHandleDesc.flags = 0;

    cudaExternalSemaphore_t semaphore;
    cudaError_t err;

    if ((err = cudaImportExternalSemaphore(&semaphore, &extSemaphoreHandleDesc)) != cudaSuccess) {
        return tl::unexpected(err);
    }

	memset(&extSemaphoreHandleDesc, 0, sizeof(extSemaphoreHandleDesc));

    return semaphore;
}
#else
void cudaVKImportSemaphore(LINUX_HANDLE handle, cudaExternalSemaphore_t extSemaphore) {
	cudaExternalSemaphoreHandleDesc extSemaphoreHandleDesc;
	memset(&extSemaphoreHandleDesc, 0, sizeof(extSemaphoreHandleDesc));

	extSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
	extSemaphoreHandleDesc.handle.fd = handle;
	extSemaphoreHandleDesc.flags = 0;
	cudaErrorCheck(cudaImportExternalSemaphore(&semaphore,
		&extSemaphoreHandleDesc));
	memset(&extSemaphoreHandleDesc, 0, sizeof(extSemaphoreHandleDesc));
}
#endif

#ifdef _WIN64
tl::expected<std::pair<cudaExternalMemory_t, CUdeviceptr>, cudaError_t> cudaVKImportMemory(HANDLE handle, size_t size) {
	cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
	memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
	cudaExtMemHandleDesc.type =
		IsWindows8OrGreater() ? cudaExternalMemoryHandleTypeOpaqueWin32
		: cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
	cudaExtMemHandleDesc.handle.win32.handle = handle;
	cudaExtMemHandleDesc.size = size;

    cudaExternalMemory_t mem;
    cudaError_t err;

    if ((err = cudaImportExternalMemory(&mem, &cudaExtMemHandleDesc)) != cudaSuccess)
        return tl::unexpected(err);

	cudaExternalMemoryBufferDesc extMemBufferDesc = {};
	extMemBufferDesc.offset = 0;
	extMemBufferDesc.size = size;
	extMemBufferDesc.flags = 0;

    CUdeviceptr devPtr;

	// TODO: Proper casting safety
    if ((err = cudaExternalMemoryGetMappedBuffer((void**)devPtr, mem, &extMemBufferDesc)) != cudaSuccess)
        return tl::unexpected(err);

    return std::make_pair(mem, devPtr);
}
#else
void cudaVKImportImageMem(LINUXHANDLE handle, cudaExternalMemory_t extMemory) {
	cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
	memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
	cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	cudaExtMemHandleDesc.handle.fd = handle;
#endif

tl::expected<void, cudaError_t> cudaVKSemaphoreSignal(cudaExternalSemaphore_t extSemaphore, cudaStream_t stream) {
	cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
	memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));

	extSemaphoreSignalParams.params.fence.value = 0;
	extSemaphoreSignalParams.flags = 0;

    cudaError_t err;

	if ((err = cudaSignalExternalSemaphoresAsync(&extSemaphore, &extSemaphoreSignalParams, 1, stream)) != cudaSuccess)
        return tl::unexpected(err);
}

tl::expected<void, cudaError_t> cudaVKSemaphoreWait(cudaExternalSemaphore_t extSemaphore, cudaStream_t stream) {
	cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;
	memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));
	extSemaphoreWaitParams.params.fence.value = 0;
	extSemaphoreWaitParams.flags = 0;

    cudaError_t err;

	if ((err = cudaWaitExternalSemaphoresAsync(&extSemaphore, &extSemaphoreWaitParams, 1, stream)) != cudaSuccess)
        return tl::unexpected(err);
}
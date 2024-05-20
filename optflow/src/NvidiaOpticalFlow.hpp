#include <optional>
#include <vector>

#include <cuda.h>
#include <driver_types.h>

#include "NvOFDefines.hpp"
#include "NvOFUtils.hpp"

#include <tl/expected.hpp>

struct NvidiaOpticalFlowDescriptor {
    uint32_t width;
    uint32_t height;
	uint32_t gridSize;
	uint32_t hintGridSize;
	uint32_t numInputBuffers;
	uint32_t gpuId;
	std::optional<std::string> outputFileName;
};

class NvidiaOpticalFlow {
	uint32_t width;
    uint32_t height;
	uint32_t gridSize;
	uint32_t hintGridSize;
	uint32_t numInputBuffers;
	uint32_t gpuId;
	std::optional<std::string> outputFileName;
    
    uint32_t scaleFactor = 1;
    CUdevice device;
    CUcontext context;
    cudaStream_t inputStream;
    cudaStream_t outputStream;

	std::vector<NvOFBufferObj> upSampleBuffers;
	std::vector<NvOFBufferObj> inputBuffers;
	std::vector<NvOFBufferObj> outputBuffers;

    std::unique_ptr<NvOF> nvOpticalFlow;
	std::unique_ptr<NvOFFileWriter> flowFileWriter;
	std::unique_ptr<NV_OF_FLOW_VECTOR[]> flowVectors;
	std::unique_ptr<NvOFUtils> nvOFUtils;

public:
	static tl::expected<std::unique_ptr<NvidiaOpticalFlow>, std::string> create(NvidiaOpticalFlowDescriptor desc);
	tl::expected<void, std::string> run(std::vector<unsigned short*> data);

	// TODO: Make these private
	NvidiaOpticalFlow(NvidiaOpticalFlowDescriptor);
	tl::expected<void, std::string> initialise();
};
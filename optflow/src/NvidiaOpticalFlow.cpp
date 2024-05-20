#include <NvidiaOpticalFlow.hpp>
#include <NvOFCuda.hpp>

NvidiaOpticalFlow::NvidiaOpticalFlow(NvidiaOpticalFlowDescriptor desc)
    : width(desc.width), height(desc.height), gridSize(desc.gridSize), hintGridSize(desc.hintGridSize), numInputBuffers(desc.numInputBuffers), gpuId(gpuId), outputFileName(outputFileName) {
}

tl::expected<void, std::string> NvidiaOpticalFlow::initialise() {
    CUDA_DRVAPI_CALL(cuInit(0));
	CUDA_DRVAPI_CALL(cuDeviceGet(&device, gpuId));
	char szDeviceName[80];
	CUDA_DRVAPI_CALL(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), device));
	std::cout << "GPU in use: " << szDeviceName << std::endl;
	CUDA_DRVAPI_CALL(cuCtxCreate(&context, 0, device));

	CUDA_DRVAPI_CALL(cuStreamCreate(&inputStream, CU_STREAM_DEFAULT));
	CUDA_DRVAPI_CALL(cuStreamCreate(&outputStream, CU_STREAM_DEFAULT));

    nvOpticalFlow = NvOFCuda::Create(
		context,
        width,
        height,
		NV_OF_BUFFER_FORMAT_GRAYSCALE8,
		NV_OF_CUDA_BUFFER_TYPE_CUARRAY,
		NV_OF_CUDA_BUFFER_TYPE_CUARRAY,
		NV_OF_MODE_OPTICALFLOW,
		NV_OF_PERF_LEVEL_SLOW,
		inputStream,
		outputStream
	);

    uint32_t hwGridSize;

	if (!nvOpticalFlow->CheckGridSize(gridSize)) {
		if (!nvOpticalFlow->GetNextMinGridSize(gridSize, hwGridSize)) {
			return tl::unexpected("Invalid grid size parameter");
		}
		else {
			scaleFactor = hwGridSize / gridSize;
		}
	}
	else {
		hwGridSize = gridSize;
	}

    nvOpticalFlow->Init(gridSize, hintGridSize, false, false);

	const uint32_t numOutputBuffers = numInputBuffers - 1;
	inputBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_INPUT, numInputBuffers);
	outputBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, numOutputBuffers);

    nvOFUtils.reset(new NvOFUtilsCuda(NV_OF_MODE_OPTICALFLOW));

	if (scaleFactor > 1) {
		uint32_t outWidth = (width + gridSize - 1) / gridSize;
		uint32_t outHeight = (height + gridSize - 1) / gridSize;

		upSampleBuffers = nvOpticalFlow->CreateBuffers(outWidth, outHeight, NV_OF_BUFFER_USAGE_OUTPUT, 16);

		uint32_t outSize = outWidth * outHeight;
		flowVectors.reset(new NV_OF_FLOW_VECTOR[outSize]);
		if (flowVectors == nullptr) {
			return tl::unexpected(std::format("Failed to allocate output host memory of size {} bytes", outSize * sizeof(NV_OF_FLOW_VECTOR)));
		}

		flowFileWriter = NvOFFileWriter::Create(outWidth,
			outHeight,
			NV_OF_MODE_OPTICALFLOW,
			32.0f);
	}
	else {
		uint32_t outSize = outputBuffers[0]->getWidth() * outputBuffers[0]->getHeight();
		flowVectors.reset(new NV_OF_FLOW_VECTOR[outSize]);
		if (flowVectors == nullptr) {
			return tl::unexpected(std::format("Failed to allocate output host memory of size {} bytes", outSize * sizeof(NV_OF_FLOW_VECTOR)));
		}

		flowFileWriter = NvOFFileWriter::Create(outputBuffers[0]->getWidth(),
			outputBuffers[0]->getHeight(),
			NV_OF_MODE_OPTICALFLOW,
			32.0f);
	}

	return {};
}

tl::expected<std::unique_ptr<NvidiaOpticalFlow>, std::string> NvidiaOpticalFlow::create(NvidiaOpticalFlowDescriptor desc) {
	std::unique_ptr<NvidiaOpticalFlow> obj = std::make_unique<NvidiaOpticalFlow>(desc);

	auto result = obj->initialise();
	result.map_error([](std::string err) {
		std::cout << err << std::endl;
	});

	return obj;
}

void NvOFBatchExecute(NvOFObj& nvOpticalFlow,
	std::vector<NvOFBufferObj>& inputBuffers,
	std::vector<NvOFBufferObj>& outputBuffers,
	uint32_t batchSize,
	CUstream inputStream,
	CUstream outputStream)
{
	for (uint32_t i = 0; i < batchSize; i++)
	{
		nvOpticalFlow->Execute(inputBuffers[i].get(),
			inputBuffers[i + 1].get(),
			outputBuffers[i].get()
		);
	}
}
tl::expected<void, std::string> NvidiaOpticalFlow::run(std::vector<unsigned short*> data) {
	uint32_t	curFrameIdx = 0;
	uint32_t	frameCount	= 0;

	for (auto& dataPtr : data) {
		inputBuffers[curFrameIdx]->UploadData(dataPtr);

		if (curFrameIdx == numInputBuffers - 1) {
			NvOFBatchExecute(nvOpticalFlow, inputBuffers, outputBuffers, curFrameIdx, inputStream, outputStream);

			if (outputFileName.has_value()) {
				for (uint32_t i = 0; i < curFrameIdx; i++) {
					if (scaleFactor > 1) {
						nvOFUtils->Upsample(outputBuffers[i].get(), upSampleBuffers[i].get(), scaleFactor);
						upSampleBuffers[i]->DownloadData(flowVectors.get());
					}
					else {
						outputBuffers[i]->DownloadData(flowVectors.get());
						for (int x = 0; x < outputBuffers[i]->getWidth(); ++x) {
							for (int y = 0; y < outputBuffers[i]->getHeight(); ++y) {
								size_t idx = y * outputBuffers[i]->getWidth() + x;
								//auto flowVec = flowVectors[idx];
								//float floatX = (float)(flowVec.flowx / float(1 << 5));
								//float floatY = (float)(flowVec.flowy / float(1 << 5));
								//if (floatX > 0 || floatY > 0)
								//	std::cout << std::format("x: {}, y: {}, FlowX: {:.5f}, FlowY: {:.5f}", x, y, floatX, floatY) << std::endl;
							}
						}
					}
					flowFileWriter->SaveOutput(flowVectors.get(),
						outputFileName.value(), frameCount, true);
					frameCount++;
				}
			}
			else {
				frameCount += curFrameIdx;
			}
			swap(inputBuffers[curFrameIdx], inputBuffers[0]);
			curFrameIdx = 0;
		}
		std::cout << std::format("Frame: {}", curFrameIdx) << std::endl;
		curFrameIdx++;
	}
}
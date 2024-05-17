#include <corrections.hpp>

#include <driver_types.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

DarkCorrection::DarkCorrection(thrust::device_vector<u16> darkMap, u16 offset)
	: offset(offset), darkMap(darkMap) {
}

void DarkCorrection::run(thrust::device_vector<u16>& input) {
	u16 offset = this->offset;
	thrust::transform(
		input.begin(), input.end(),
		darkMap.begin(),
		input.begin(),
		[offset] __device__(u16 x, u16 y) {
		return (x - y) + offset;
	});
}

GainCorrection::GainCorrection(thrust::device_vector<u16> gainMap) {
    normaliseGainMap(gainMap);
}

void GainCorrection::run(thrust::device_vector<u16>& input) {
	thrust::transform(
		input.begin(), input.end(),
		normedGainMap.begin(),
		input.begin(),
		[] __device__(u16 val, double normedVal) {
		return val * normedVal;
	}
	);
}

void GainCorrection::normaliseGainMap(thrust::device_vector<u16> gainMap) {
	double sum = thrust::reduce(gainMap.begin(), gainMap.end(), unsigned long long(0), thrust::plus<unsigned long long>());
	double mean = sum / gainMap.size();

	normedGainMap = thrust::device_vector<float>(gainMap.size());

	thrust::transform(
		gainMap.begin(), gainMap.end(),
		normedGainMap.begin(),
		[mean] __device__(u16 val) {
		return  double(mean) / double(val);
	});
}

constexpr size_t DEFECT_CORRECTION_KERNEL_SIZE = 3;
__constant__ u16 defectCorrectionKernel[DEFECT_CORRECTION_KERNEL_SIZE * DEFECT_CORRECTION_KERNEL_SIZE];

__global__ static void averageNeighboursKernel(u16* input, const u16* defectMap, int width, int height, int kernelSize) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int kernelRadius = DEFECT_CORRECTION_KERNEL_SIZE / 2;

	if (x >= width || y >= height) return;

	int count = 0;
	int sum = 0;

	for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
		for (int dx = -kernelRadius; dx <= kernelRadius; dx++) {
			int nx = x + dx;
			int ny = y + dy;
			if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
				int idx = (dy + kernelRadius) * kernelSize + (dx + kernelRadius);
				int defectMapIdx = ny * width + nx;
				sum += input[ny * width + nx] * defectCorrectionKernel[idx] * (1 - defectMap[defectMapIdx]);
				count += defectCorrectionKernel[idx] * (1 - defectMap[defectMapIdx]);
			}
		}
	}

	if (count > 0)
		input[y * width + x] = sum / count;
}

DefectCorrection::DefectCorrection(Config config, thrust::device_vector<u16> defectMap)
	: defectMap(defectMap), config(config) {
	std::vector<u16> kernelTemp = {
		1, 1, 1,
		1, 0, 1,
		1, 1, 1
	};

	cudaMemcpyToSymbol(defectCorrectionKernel, kernelTemp.data(), kernelTemp.size() * sizeof(u16));
}

void DefectCorrection::run(thrust::device_vector<u16>& input) {
	dim3 blockSize(16, 16);
	dim3 gridSize((config.imageWidth + blockSize.x - 1) / blockSize.x,
		(config.imageHeight + blockSize.y - 1) / blockSize.y);

	u16* rawInputData = thrust::raw_pointer_cast(input.data());
	u16* rawDefectData = thrust::raw_pointer_cast(defectMap.data());

	averageNeighboursKernel << <gridSize, blockSize, 0 >> > (
		rawInputData,
		rawDefectData,
		config.imageWidth,
		config.imageHeight,
		3
	);
}
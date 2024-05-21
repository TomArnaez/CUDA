#include <corrections.hpp>

#include <device_launch_parameters.h>
#include <driver_types.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

DarkCorrection::DarkCorrection(std::span<unsigned short> darkMapSpan, unsigned short offset)
	: offset(offset), darkMap(darkMapSpan.size()) {
		thrust::copy(darkMapSpan.begin(), darkMapSpan.end(), darkMap.begin());
}

DarkCorrection::DarkCorrection(thrust::device_vector<unsigned short> darkMap, unsigned short offset)
	: offset(offset), darkMap(darkMap) {
}

void DarkCorrection::run(thrust::device_vector<unsigned short>& input) {
	unsigned short offset = this->offset;
	thrust::transform(
		input.begin(), input.end(),
		darkMap.begin(),
		input.begin(),
		[offset] __device__(unsigned short x, unsigned short y) {
		return (x - y) + offset;
	});
}

GainCorrection::GainCorrection(thrust::device_vector<unsigned short> gainMap) {
    normaliseGainMap(gainMap);
}

void GainCorrection::run(thrust::device_vector<unsigned short>& input) {
	thrust::transform(
		input.begin(), input.end(),
		normedGainMap.begin(),
		input.begin(),
		[] __device__(unsigned short val, double normedVal) {
		return val * normedVal;
	}
	);
}

void GainCorrection::normaliseGainMap(thrust::device_vector<unsigned short> gainMap) {
	double sum = thrust::reduce(gainMap.begin(), gainMap.end(), unsigned long long(0), thrust::plus<unsigned long long>());
	double mean = sum / gainMap.size();

	normedGainMap = thrust::device_vector<float>(gainMap.size());

	thrust::transform(
		gainMap.begin(), gainMap.end(),
		normedGainMap.begin(),
		[mean] __device__(unsigned short val) {
		return  double(mean) / double(val);
	});
}

constexpr size_t DEFECT_CORRECTION_KERNEL_SIZE = 3;
__constant__ unsigned short defectCorrectionKernel[DEFECT_CORRECTION_KERNEL_SIZE * DEFECT_CORRECTION_KERNEL_SIZE];

__global__ static void averageNeighboursKernel(unsigned short* input, const unsigned short* defectMap, int width, int height, int kernelSize) {
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

DefectCorrection::DefectCorrection(thrust::device_vector<unsigned short> defectMap, uint32_t width, uint32_t height)
	: defectMap(defectMap), width(width), height(height) {
	std::vector<unsigned short> kernelTemp = {
		1, 1, 1,
		1, 0, 1,
		1, 1, 1
	};

	cudaMemcpyToSymbol(defectCorrectionKernel, kernelTemp.data(), kernelTemp.size() * sizeof(unsigned short));
}

void DefectCorrection::run(thrust::device_vector<unsigned short>& input) {
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	unsigned short* rawInputData = thrust::raw_pointer_cast(input.data());
	unsigned short* rawDefectData = thrust::raw_pointer_cast(defectMap.data());

	averageNeighboursKernel << <gridSize, blockSize, 0 >> > (
		rawInputData,
		rawDefectData,
		width,
		height,
		3
	);
}

constexpr uint32_t HISTOGRAM_EQ_RANGE = 256;

HistogramEquilisation::HistogramEquilisation(uint32_t width, uint32_t height, uint32_t numBins)
	: width(width), height(height), numBins(numBins), histogram(numBins), normedHistogram(numBins), LUT(numBins) {
	// cub::DeviceHistogram::HistogramEven(
	// 	tempStorage, tempStorageBytes,
	// 	static_cast<unsigned short*>(nullptr), thrust::raw_pointer_cast(histogram.data()), numBins,
	// 	0, numBins - 1, width * height
	// );

	// cudaMalloc(&tempStorage, tempStorageBytes);
}

void HistogramEquilisation::run(thrust::device_vector<unsigned short>& input) {
	uint32_t totalPixels = width * height;
	// // TODO: Bug where the sum of the histogram doesn't equal number of pixels
	// cub::DeviceHistogram::HistogramEven(
	// 	tempStorage, tempStorageBytes,
	// 	thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(histogram.data()), numBins,
	// 	0, numBins - 1, totalPixels);

	// thrust::inclusive_scan(histogram.begin(), histogram.end(), histogram.begin());

	// thrust::transform(histogram.begin(), histogram.end(), normedHistogram.begin(),
	// 	[totalPixels] __device__(unsigned int x) -> float {
	// 	return static_cast<float>(x) / totalPixels;
	// });

	// thrust::transform(normedHistogram.begin(), normedHistogram.end(), LUT.begin(),
	// 	[] __device__(float x) -> u16 {
	// 	return static_cast<u16>(HISTOGRAM_EQ_RANGE * x);
	// });

	// thrust::gather(input.begin(), input.end(), LUT.begin(), input.begin());
}

HistogramEquilisation::~HistogramEquilisation() {
	if (tempStorage)
		cudaFree(tempStorage);
}
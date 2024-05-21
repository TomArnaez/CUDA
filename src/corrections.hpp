#pragma once

#include <span>
#include <thrust/device_vector.h>

class ICorrection {
public:
    virtual void run(thrust::device_vector<unsigned short>& input) = 0;
};

class DarkCorrection: public ICorrection {
private:
	thrust::device_vector<unsigned short> darkMap;
	unsigned short offset;
public:
    DarkCorrection(std::span<unsigned short> darkMap, unsigned short offset);
    DarkCorrection(thrust::device_vector<unsigned short> darkMap, unsigned short offset);
	void run(thrust::device_vector<unsigned short>& input) override;
};

class GainCorrection: public ICorrection {
private:
	thrust::device_vector<double> normedGainMap;
public:
	GainCorrection(thrust::device_vector<unsigned short> gainMap);
    void run(thrust::device_vector<unsigned short>& input);
    void normaliseGainMap(thrust::device_vector<unsigned short> gainMap);
};

class DefectCorrection : public ICorrection {
private:
	uint32_t width;
	uint32_t height;
	thrust::device_vector<unsigned short> defectMap;
public:
	DefectCorrection(thrust::device_vector<unsigned short> defectMap, uint32_t width, uint32_t height);
	void run(thrust::device_vector<unsigned short>& input) override;
};

class HistogramEquilisation : public ICorrection {
private:
	uint32_t width;
	uint32_t height;
	int numBins;
	void* tempStorage = nullptr;
	size_t tempStorageBytes = 0;
	thrust::device_vector<int> histogram;
	thrust::device_vector<float> normedHistogram;
	thrust::device_vector<unsigned short> LUT;
public:
	HistogramEquilisation(uint32_t width, uint32_t height, uint32_t numBins);
	~HistogramEquilisation();
	void run(thrust::device_vector<unsigned short>& input) override;
};
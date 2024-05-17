#include <thrust/device_vector.h>

#include <types.hpp>

class ICorrection {
public:
    virtual void run(thrust::device_vector<u16>& input) = 0;
};

class DarkCorrection: public ICorrection {
private:
	thrust::device_vector<u16> darkMap;
	u16 offset;
public:
    DarkCorrection(thrust::device_vector<u16> darkMap, u16 offset);
	void run(thrust::device_vector<u16>& input) override;
};

class GainCorrection: public ICorrection {
private:
	thrust::device_vector<double> normedGainMap;
public:
	GainCorrection(thrust::device_vector<u16> gainMap);
    void run(thrust::device_vector<u16>& input);
    void normaliseGainMap(thrust::device_vector<u16> gainMap);
};

class DefectCorrection : public ICorrection {
private:
	thrust::device_vector<u16> defectMap;
	Config config;
public:
	DefectCorrection(Config config, thrust::device_vector<u16> defectMap);
	void run(thrust::device_vector<u16>& input) override;
};
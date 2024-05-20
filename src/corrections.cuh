#include <thrust/device_vector.h>

#include <types.hpp>

class ICorrection {
public:
    virtual void run(thrust::device_vector<unsigned short>& input) = 0;
};

class DarkCorrection: public ICorrection {
private:
	thrust::device_vector<unsigned short> darkMap;
	unsigned short offset;
public:
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
	thrust::device_vector<unsigned short> defectMap;
	Config config;
public:
	DefectCorrection(Config config, thrust::device_vector<unsigned short> defectMap);
	void run(thrust::device_vector<unsigned short>& input) override;
};
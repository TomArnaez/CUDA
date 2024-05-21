#pragma once

#include <vector>
#include <corrections.hpp>

enum class CorrectionType {
    DarkCorrection,
    GainCorrection,
    DefectCorrection,
    Count
};

struct CorrectionEntry {
    std::shared_ptr<ICorrection> correction;
    bool enabled;
};

class Core {
    uint32_t width;
    uint32_t height;
    std::array<CorrectionEntry, static_cast<size_t>(CorrectionType::Count)> corrections;
public:
    Core(uint32_t width, uint32_t height);
    
    void addCorrection(CorrectionType type, std::shared_ptr<ICorrection> correction);
    void enableCorrection(CorrectionType type, bool enable);
    void run(thrust::device_vector<unsigned short>& input);
};
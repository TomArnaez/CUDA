#include "core.hpp"

Core::Core(uint32_t width, uint32_t height) {
    
}

void Core::addCorrection(CorrectionType type, std::unique_ptr<ICorrection> correction) {
    corrections[static_cast<size_t>(type)] = { std::move(correction), true };
}

void Core::enableCorrection(CorrectionType type, bool enable) {
    corrections[static_cast<size_t>(type)].enabled = enable;
}

void Core::run(thrust::device_vector<unsigned short>& input) {
    for (auto &entry : corrections)
        if (entry.enabled && entry.correction)
            entry.correction->run(input);
}
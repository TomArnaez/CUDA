//#include <NvidiaOpticalFlow.hpp>
//#include <NvOF.hpp>
#include <algorithm>
#include <ranges>

#include <iostream>
#include <opencv2/imgcodecs.hpp>

std::string DEMO_IMAGE_DIR = "C:\\dev\\data\\Test Images\\";
std::string clock_video_8bit_path = DEMO_IMAGE_DIR + "clock_video_8bit_maxed.tif";
std::string FLOW_RESULTS = DEMO_IMAGE_DIR + "FlowResults\\Flow";

int main() {
    std::vector<cv::Mat> mats;
    if (!cv::imreadmulti(clock_video_8bit_path, mats)) {
        std::cout << "Failed to read image" << std::endl;
        return -1;
    }

    // NvidiaOpticalFlowDescriptor descriptor {
    //     .width = mats[0].cols,
    //     .height = mats[0].rows,
    //     .gridSize = 1,
    //     .hintGridSize = 4,
    //     .numInputBuffers = 16,
    //     .gpuId = 0,
    //     .outputFileName = FLOW_RESULTS
    // };

    // std::vector<unsigned short*> data_ptrs;

    // std::ranges::transform(mats, std::back_inserter(data_ptrs), [](cv::Mat& mat) {
    //     return mat.data;
    // });

    // try {
    //     auto result = NvidiaOpticalFlow::create(descriptor);
    //     if (result.has_value()) {

    //         result.value()->run(data_ptrs);
    //     }
    // }
    // catch (NvOFException ex) {
    //     std::cout << "Got NvOF Exception: " << ex.getErrorString() << std::endl;
    // }
}
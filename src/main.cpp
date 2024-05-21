#include <iostream>
#include <opencv2/opencv.hpp>
#include <correctioncore.hpp>
#include <span>
// #include <thrust/host_vector.h>

const std::string TEST_IMAGES_DIR = "C:\\dev\\data\\Test Images\\";
const std::string DARK_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_Dark_2802_2400.tif";
const std::string GAIN_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_Gain_2802_2400.tif";
const std::string DEFECT_IMAGE_PATH = TEST_IMAGES_DIR + "DefectMap.tif";
const std::string PCB_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_PCB_2802_2400.tif";

#include <iostream>
#include <vector>
#include <cassert>

#define VK_USE_PLATFORM_WIN32_KHR
#define NOMINMAX
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <vulkaninterop.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

vk::Semaphore createExportableSemaphore(const vk::Device& device) {
    vk::SemaphoreCreateInfo semaphoreCreateInfo;
    vk::ExportSemaphoreCreateInfo exportSemaphoreCreateInfo;
    exportSemaphoreCreateInfo.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
    semaphoreCreateInfo.pNext = &exportSemaphoreCreateInfo;
    vk::Semaphore semaphore = device.createSemaphore(semaphoreCreateInfo);
    return semaphore;
}

HANDLE getWin32HandleFromSemaphore(const vk::Device& device, const vk::Semaphore& semaphore) {
    vk::SemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfo;
    semaphoreGetWin32HandleInfo.semaphore = semaphore;
    semaphoreGetWin32HandleInfo.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;

    HANDLE win32Handle = device.getSemaphoreWin32HandleKHR(semaphoreGetWin32HandleInfo);
    return win32Handle;
}

vk::DeviceMemory allocateExportableMemory(const vk::Device& device, vk::DeviceSize size, vk::PhysicalDevice& physicalDevice, const vk::Buffer& buffer) {
    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
    vk::MemoryAllocateInfo memAllocInfo;

    vk::ExportMemoryAllocateInfo exportAllocInfo;
    exportAllocInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;

    memAllocInfo.pNext = &exportAllocInfo;
    memAllocInfo.allocationSize = memRequirements.size;

    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) &&
            (memProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            memAllocInfo.memoryTypeIndex = i;
            break;
        }
    }

    vk::DeviceMemory memory = device.allocateMemory(memAllocInfo);
    return memory;
}

HANDLE getWin32HandleFromMemory(const vk::Device& device, const vk::DeviceMemory& memory) {
    vk::MemoryGetWin32HandleInfoKHR memoryGetWin32HandleInfo;
    memoryGetWin32HandleInfo.memory = memory;
    memoryGetWin32HandleInfo.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;

    HANDLE win32Handle = device.getMemoryWin32HandleKHR(memoryGetWin32HandleInfo);
    return win32Handle;
}

vk::Buffer createBuffer(const vk::Device& device, vk::DeviceSize size, vk::BufferUsageFlags usage) {
    vk::BufferCreateInfo bufferCreateInfo({}, size, usage, vk::SharingMode::eExclusive);
    vk::Buffer buffer = device.createBuffer(bufferCreateInfo);
    return buffer;
}

void copyDataToBuffer(const vk::Device& device, const vk::DeviceMemory& bufferMemory, const void* data, vk::DeviceSize size) {
    void* mappedMemory = device.mapMemory(bufferMemory, 0, size);
    std::memcpy(mappedMemory, data, static_cast<size_t>(size));
    device.unmapMemory(bufferMemory);
}

int main() {
    // Initialize the dynamic dispatcher
    VULKAN_HPP_DEFAULT_DISPATCHER.init();
    vk::DynamicLoader dl;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(dl);
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(getInstanceProcAddr);

    // Create a Vulkan instance with the required extensions
    vk::ApplicationInfo appInfo("External Semaphore Example", 1, "No Engine", 1, VK_API_VERSION_1_2);

    std::vector<const char*> instanceExtensions = { };
    vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo, 0, nullptr, instanceExtensions.size(), instanceExtensions.data());
    vk::Instance instance = vk::createInstance(instanceCreateInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

    std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();

    vk::PhysicalDevice physicalDevice = physicalDevices[1];

    std::vector<const char*> deviceExtensions = {
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
    };

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo({}, 0, 1, &queuePriority);
    vk::DeviceCreateInfo deviceCreateInfo({}, 1, &queueCreateInfo, 0, nullptr, deviceExtensions.size(), deviceExtensions.data());
    vk::Device device = physicalDevice.createDevice(deviceCreateInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    // Create an exportable semaphore
    vk::SemaphoreCreateInfo semaphoreCreateInfo;
    vk::ExportSemaphoreCreateInfo exportSemaphoreCreateInfo;
    exportSemaphoreCreateInfo.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
    semaphoreCreateInfo.pNext = &exportSemaphoreCreateInfo;

    vk::Semaphore inputSemaphore = createExportableSemaphore(device);
    vk::Semaphore outputSemaphore = createExportableSemaphore(device);

    HANDLE inputSemaphoreHandle = getWin32HandleFromSemaphore(device, inputSemaphore);
    HANDLE outputSemaphoreHandle = getWin32HandleFromSemaphore(device, outputSemaphore);

    cv::Mat darkMap = cv::imread(DARK_IMAGE_PATH, cv::IMREAD_UNCHANGED);
    cv::Mat gainMap = cv::imread(GAIN_IMAGE_PATH, cv::IMREAD_UNCHANGED);
    cv::Mat defectMap = cv::imread(DEFECT_IMAGE_PATH, cv::IMREAD_UNCHANGED);
    cv::Mat PCBImage = cv::imread(PCB_IMAGE_PATH, cv::IMREAD_UNCHANGED);

    vk::DeviceSize bufferSize = PCBImage.rows * PCBImage.cols * sizeof(unsigned short);
    vk::BufferUsageFlags usageFlags = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::Buffer buffer = createBuffer(device, bufferSize, usageFlags);

    vk::DeviceMemory bufferMemory = allocateExportableMemory(device, bufferSize, physicalDevice, buffer);
    HANDLE bufferMemoryHandle = getWin32HandleFromMemory(device, bufferMemory);
    device.bindBufferMemory(buffer, bufferMemory, 0);
    std::vector<char> data(bufferSize, 'A');
    copyDataToBuffer(device, bufferMemory, data.data(), bufferSize);

    VulkanCudaCreateInfo createInfo {
        .width = static_cast<uint32_t>(PCBImage.cols),
        .height = static_cast<uint32_t>(PCBImage.rows),
        .inputSemaphoreHandle = inputSemaphoreHandle,
        .outputSemaphoreHandle = outputSemaphoreHandle,
        .bufferMemoryHandle =  bufferMemoryHandle,
        .bufferMemorySize = bufferSize
    };

    auto res = VulkanCuda::create(createInfo);
    if (!res.has_value()) {
        std::cout << "Got err: " << res.error() << std::endl;
        return -1;
    }

    return 0;
}

// thrust::device_vector<unsigned short> getThrustVector(cv::Mat& image) {
//     size_t numPixels = image.total();
//     thrust::host_vector<unsigned short> vec(numPixels);
// 	std::memcpy(thrust::raw_pointer_cast(vec.data()), image.data, numPixels * sizeof(unsigned short));
//     return vec;
// }

// void saveDeviceVectorAsImage(const thrust::device_vector<unsigned short> d_vector, int rows, int cols, const std::string& filename) {
//     thrust::host_vector<unsigned short> h_vector = d_vector;
//     cv::Mat image(rows, cols, CV_16U, h_vector.data());

//     std::vector<int> params;
//     params.push_back(cv::IMWRITE_TIFF_COMPRESSION);
//     params.push_back(1);

//     bool result = cv::imwrite(filename, image, params);
//     if (result) {
//         std::cout << "Image successfully saved as " << filename << std::endl;
//     } else {
//         std::cerr << "Failed to save the image as " << filename << std::endl;
//     }
// }

// int main() {
//     testVulkan();
    // cv::Mat darkMap = cv::imread(DARK_IMAGE_PATH, cv::IMREAD_UNCHANGED);
    // cv::Mat gainMap = cv::imread(GAIN_IMAGE_PATH, cv::IMREAD_UNCHANGED);
    // cv::Mat defectMap = cv::imread(DEFECT_IMAGE_PATH, cv::IMREAD_UNCHANGED);
    // cv::Mat PCBImage = cv::imread(PCB_IMAGE_PATH, cv::IMREAD_UNCHANGED);

    // std::span<unsigned short> imageSpan(darkMap.ptr<unsigned short>(), darkMap.total());

    // std::shared_ptr<ICorrection> darkCorrection = std::make_shared<DarkCorrection>(imageSpan, 300);
    // std::shared_ptr<ICorrection> gainCorrection = std::make_shared<GainCorrection>(getThrustVector(gainMap));
    // std::shared_ptr<ICorrection> defectCorrection = std::make_shared<DefectCorrection>(getThrustVector(defectMap), darkMap.cols, darkMap.rows);

    // auto inputVec = getThrustVector(PCBImage);

    // Core core(PCBImage.cols, PCBImage.rows);
    // core.addCorrection(CorrectionType::DarkCorrection, darkCorrection);
    // core.addCorrection(CorrectionType::GainCorrection, gainCorrection);
    // core.addCorrection(CorrectionType::DefectCorrection, defectCorrection);

    // core.run(inputVec);

    // saveDeviceVectorAsImage(inputVec, PCBImage.rows, PCBImage.cols, TEST_IMAGES_DIR + "result.tif");
// }
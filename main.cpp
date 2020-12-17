#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan.h>
#define NOMINMAX
#include <windows.h>
#include <vector>
#include <optional>
#include <map>
#include <algorithm>
#include <tuple>

struct GpuMemoryProps
{
	uint32_t memoryTypeIndex;
	bool hostVisible;
	bool hostCoherent;
	bool hostCached;
};


struct CpuMemoryProps
{
	uint32_t memoryTypeIndex;
	bool hostCoherent;
	bool hostCached;
};


enum EQueueType
{
	kQueueGfx = 0,
	kQueueCompute,
	kQueueTransfer,
	kQueueTypesCount
};


struct Queue
{
	EQueueType type;
	VkQueue queue;
	VkCommandPool cmdPool;
	VkCommandBuffer cmdBuffer;
};


struct Buffer
{
	VkBuffer buffer;
	VkDeviceMemory memory;
	VkDeviceSize size;
	void* mappedAddr;
};


static VkInstance GVkInstance = VK_NULL_HANDLE;
static VkPhysicalDevice GVkPhysDevice = VK_NULL_HANDLE;
static VkDevice GVkDevice = VK_NULL_HANDLE;
static int32_t GQueueFamilyIndices[kQueueTypesCount] = {-1, -1, -1};
static std::vector<Queue> GQueues;
static std::vector<GpuMemoryProps> GGpuMemory;
static std::vector<CpuMemoryProps> GCpuMemory;
static VkFence GFence;
static double GInvPerfFrequency = 0.0f;
static const char* kQueueTypeStr[] = { "Graphics", "Compute", "Transfer" };


inline void LogStdOut(const char* format, ...)
{
	char buf[2048];
	va_list args;
	va_start(args, format);
	int len = vsprintf_s(buf, format, args);
	va_end(args);
	fwrite(buf, 1, len, stdout);
	OutputDebugStringA(buf);
}


inline void LogStdErr(const char* format, ...)
{
	char buf[2048];
	va_list args;
	va_start(args, format);
	int len = vsprintf_s(buf, format, args);
	va_end(args);
	fwrite(buf, 1, len, stderr);
	OutputDebugStringA(buf);
}


template <class T>
inline T SetBit(T& mask, uint32_t bitIndex)
{
	mask |= static_cast<T>(1) << bitIndex;
	return mask;
}


template <class T>
inline T ResetBit(T& mask, uint32_t bitIndex)
{
	mask &= ~(static_cast<T>(1) << bitIndex);
	return mask;
}


template <class T>
inline T ToggleBit(T& mask, uint32_t bitIndex)
{
	mask ^= static_cast<T>(1) << bitIndex;
	return mask;
}


template <class T>
inline bool TestBit(const T& mask, uint32_t bitIndex)
{
	return mask & (static_cast<T>(1) << bitIndex);
}


static void EnumerateQueues()
{
	uint32_t queueFamiliesCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(GVkPhysDevice, &queueFamiliesCount, 0);

	VkQueueFamilyProperties* queueFamilies = (VkQueueFamilyProperties*)alloca(queueFamiliesCount * sizeof(VkQueueFamilyProperties));
	vkGetPhysicalDeviceQueueFamilyProperties(GVkPhysDevice, &queueFamiliesCount, queueFamilies);

	uint32_t freeFamiliesMask = ~0u >> (32 - queueFamiliesCount);

	const VkQueueFlags requiredQueueFlags[kQueueTypesCount] = { VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
																VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 
		                                                        VK_QUEUE_TRANSFER_BIT };
	for (uint32_t typeIdx = 0; typeIdx < kQueueTypesCount; typeIdx++)
	{
		for (uint32_t familyIndex = 0; familyIndex < queueFamiliesCount; familyIndex++)
		{
			if (!TestBit(freeFamiliesMask, familyIndex))
				continue;

			const VkQueueFamilyProperties& famityProp = queueFamilies[familyIndex];
			if ((famityProp.queueFlags & requiredQueueFlags[typeIdx]) == requiredQueueFlags[typeIdx])
			{
				GQueueFamilyIndices[typeIdx] = familyIndex;
				ResetBit(freeFamiliesMask, familyIndex);
			}
		}
	}

	if (GQueueFamilyIndices[kQueueGfx] != -1)
		LogStdOut("Graphics queue family index: %d\n", GQueueFamilyIndices[kQueueGfx]);
	if (GQueueFamilyIndices[kQueueCompute] != -1)
		LogStdOut("Compute queue family index:  %d\n", GQueueFamilyIndices[kQueueCompute]);
	if (GQueueFamilyIndices[kQueueTransfer] != -1)
		LogStdOut("Transfer queue family index: %d\n", GQueueFamilyIndices[kQueueTransfer]);
}


static void EnumerateMemoryTypes()
{
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(GVkPhysDevice, &memProps);
	for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
	{
		LogStdOut("Memory type index %u:\n", i);
		const VkMemoryType& memType = memProps.memoryTypes[i];
		if (memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
		{
			GpuMemoryProps props;
			props.memoryTypeIndex = i;
			props.hostVisible = memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
			props.hostCoherent = memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			props.hostCached = memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
			GGpuMemory.push_back(props);
			LogStdOut("  Type:          GPU(device)\n");
			LogStdOut("  Host visible:  %s\n", props.hostVisible ? "true" : "false");
			LogStdOut("  Host coherent: %s\n", props.hostCoherent ? "true" : "false");
			LogStdOut("  Host cached:   %s\n", props.hostCached ? "true" : "false");
		}
		else if (memType.propertyFlags != 0)
		{
			CpuMemoryProps props;
			props.memoryTypeIndex = i;
			props.hostCoherent = memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			props.hostCached = memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
			GCpuMemory.push_back(props);
			LogStdOut("  Type:          CPU(host)\n");
			LogStdOut("  Host coherent: %s\n", props.hostCoherent ? "true" : "false");
			LogStdOut("  Host cached:   %s\n", props.hostCached ? "true" : "false");
		}
	}
}


static bool InitQueue(EQueueType type)
{
	Queue queue;
	queue.type = type;
	vkGetDeviceQueue(GVkDevice, GQueueFamilyIndices[type], 0, &queue.queue);

	VkCommandPoolCreateInfo poolCreateInfo = {};
	poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolCreateInfo.queueFamilyIndex = GQueueFamilyIndices[type];
	poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VkResult result = vkCreateCommandPool(GVkDevice, &poolCreateInfo, nullptr, &queue.cmdPool);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkCreateCommandPool failed\n");
		return {};
	}

	VkCommandBufferAllocateInfo allocateInfo = {};
	allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocateInfo.commandPool = queue.cmdPool;
	allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocateInfo.commandBufferCount = 1;
	result = vkAllocateCommandBuffers(GVkDevice, &allocateInfo, &queue.cmdBuffer);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkAllocateCommandBuffers failed\n");
		return {};
	}

	GQueues.push_back(queue);
	return true;
}


static bool InitVulkan()
{
	VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	appInfo.apiVersion = VK_API_VERSION_1_1;

	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.pApplicationInfo = &appInfo;

#define VK_VALIDATION 1

#if defined(_DEBUG) || VK_VALIDATION
	const char* debugLayers[] =
	{
	   "VK_LAYER_KHRONOS_validation"
	};

	createInfo.ppEnabledLayerNames = debugLayers;
	createInfo.enabledLayerCount = sizeof(debugLayers) / sizeof(debugLayers[0]);
#endif

	const char* instanceExtensions[] =
	{
 #if defined(_DEBUG) || VK_VALIDATION
	   VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
 #endif
	};

	createInfo.ppEnabledExtensionNames = instanceExtensions;
	createInfo.enabledExtensionCount = _countof(instanceExtensions);

	if (vkCreateInstance(&createInfo, 0, &GVkInstance) != VK_SUCCESS)
	{
		LogStdErr("vkCreateInstance failed\n");
		return false;
	}

	VkPhysicalDevice physicalDevices[16];
	uint32_t physicalDeviceCount = sizeof(physicalDevices) / sizeof(physicalDevices[0]);
	if (vkEnumeratePhysicalDevices(GVkInstance, &physicalDeviceCount, physicalDevices) != VK_SUCCESS)
	{
		LogStdErr("vkEnumeratePhysicalDevices failed\n");
		return false;
	}
	GVkPhysDevice = physicalDevices[0];

	EnumerateQueues();

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	float queuePriorities[] = { 1.0f };
	for (uint32_t i = 0; i < kQueueTypesCount; i++)
	{
		if (GQueueFamilyIndices[i] == -1)
			continue;
		VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
		queueCreateInfo.queueFamilyIndex = GQueueFamilyIndices[i];
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = queuePriorities;
		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	deviceCreateInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size();
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
	if (vkCreateDevice(GVkPhysDevice, &deviceCreateInfo, 0, &GVkDevice) != VK_SUCCESS)
	{
		LogStdErr("vkCreateDevice failed\n");
		return false;
	}

	EnumerateMemoryTypes();

	for (uint32_t i = 0; i < kQueueTypesCount; i++)
	{
		if (GQueueFamilyIndices[i] == -1)
			continue;

		if (!InitQueue((EQueueType)i))
			return false;
	}

	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	VkResult result = vkCreateFence(GVkDevice, &fenceCreateInfo, nullptr, &GFence);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkCreateFence failed\n");
		return {};
	}

	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	GInvPerfFrequency = 1.0 / frequency.QuadPart;

	return true;
}


static std::optional<Buffer> CreateBuffer(VkDeviceSize size, uint32_t memTypeIndex, bool canMapMemory)
{
	VkBufferCreateInfo bufferCreateInfo = {};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	Buffer buffer = {};
	VkResult result = vkCreateBuffer(GVkDevice, &bufferCreateInfo, nullptr, &buffer.buffer);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkCreateBuffer failed\n");
		return {};
	}

	VkMemoryRequirements memRequirements = {};
	vkGetBufferMemoryRequirements(GVkDevice, buffer.buffer, &memRequirements);
	if (!TestBit(memRequirements.memoryTypeBits, memTypeIndex))
	{
		LogStdErr("Buffer is unable to use memory type %u\n", memTypeIndex);
		return {};
	}

	VkMemoryAllocateInfo allocateInfo = {};
	allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocateInfo.memoryTypeIndex = memTypeIndex;
	allocateInfo.allocationSize = memRequirements.size;
	result = vkAllocateMemory(GVkDevice, &allocateInfo, nullptr, &buffer.memory);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkAllocateMemory failed\n");
		return {};
	}

	result = vkBindBufferMemory(GVkDevice, buffer.buffer, buffer.memory, 0);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkBindBufferMemory failed\n");
		return {};
	}

	if (canMapMemory)
	{
		result = vkMapMemory(GVkDevice, buffer.memory, 0, size, 0, &buffer.mappedAddr);
		if (result != VK_SUCCESS)
		{
			LogStdErr("vkMapMemory failed\n");
			return {};
		}
	}

	return buffer;
}


static uint64_t GetCpuTimestamp()
{
	LARGE_INTEGER count;
	QueryPerformanceCounter(&count);
	return count.QuadPart;
}


static std::tuple<double, VkDeviceSize> Copy(const Queue& queue, const Buffer& dstBuffer, const Buffer& srcBuffer, VkDeviceSize size)
{
	VkBufferCopy bufferCopy = {};
	if (dstBuffer.buffer != srcBuffer.buffer)
	{
		bufferCopy.size = size;
	}
	else
	{
		bufferCopy.dstOffset = size / 2;
		bufferCopy.size = size / 2;
	}

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VkResult result = vkBeginCommandBuffer(queue.cmdBuffer, &beginInfo);
	if(result != VK_SUCCESS)
	{
		LogStdErr("vkBeginCommandBuffer failed\n");
		return { -1.0f, 0 };
	}

	vkCmdCopyBuffer(queue.cmdBuffer, srcBuffer.buffer, dstBuffer.buffer, 1, &bufferCopy);
	result = vkEndCommandBuffer(queue.cmdBuffer);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkBeginCommandBuffer failed\n");
		return { -1.0f, 0 };
	}

	result = vkResetFences(GVkDevice, 1, &GFence);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkResetFences failed\n");
		return { -1.0f, 0 };
	}

	uint64_t cpuBeginTime = GetCpuTimestamp();
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &queue.cmdBuffer;
	result = vkQueueSubmit(queue.queue, 1, &submitInfo, GFence);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkQueueSubmit failed\n");
		return { -1.0f, 0 };
	}

	result = vkWaitForFences(GVkDevice, 1, &GFence, VK_TRUE, ~0llu);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkWaitForFences failed\n");
		return { -1.0f, 0 };
	}

	uint64_t cpuEndTime = GetCpuTimestamp();
	return { (cpuEndTime - cpuBeginTime) * GInvPerfFrequency, bufferCopy.size };
}


static std::tuple<double, VkDeviceSize> Copy(const Buffer& dstBuffer, const Buffer& srcBuffer, VkDeviceSize size)
{
	uint8_t* dst = (uint8_t*)dstBuffer.mappedAddr;
	uint8_t* src = (uint8_t*)srcBuffer.mappedAddr;
	if (dstBuffer.buffer == srcBuffer.buffer)
	{
		dst += size / 2;
		size /= 2;
	}

	uint64_t cpuBeginTime = GetCpuTimestamp();
	memcpy(dst, src, size);
	uint64_t cpuEndTime = GetCpuTimestamp();
	return { (cpuEndTime - cpuBeginTime) * GInvPerfFrequency, size };
}


static double ComputeSpeed(VkDeviceSize size, double time)
{
	return size / time / 1024.0 / 1024.0;
}


int main()
{
	if (!InitVulkan())
		return -1;

	const VkDeviceSize bufSize = 128llu * 1024llu * 1024llu;
	const uint32_t kNumOfMeasurements = 9;
	std::map<uint32_t, Buffer> buffers;

	for (const GpuMemoryProps& gpuMem : GGpuMemory)
	{
		std::optional<Buffer> buf = CreateBuffer(bufSize, gpuMem.memoryTypeIndex, gpuMem.hostVisible);
		if (!buf.has_value())
			return -1;
		buffers[gpuMem.memoryTypeIndex] = *buf;
	}

	for (const CpuMemoryProps& cpuMem : GCpuMemory)
	{
		std::optional<Buffer> buf = CreateBuffer(bufSize, cpuMem.memoryTypeIndex, true);
		if (!buf.has_value())
			return -1;
		buffers[cpuMem.memoryTypeIndex] = *buf;
	}

	for (const Queue& queue : GQueues)
	{
		LogStdOut("vkCmdCopyBuffer on %s queue:\n", kQueueTypeStr[queue.type]);
		for (auto& srcBufferIter : buffers)
		{
			const uint32_t srcMemIndex = srcBufferIter.first;
			const Buffer& srcBuffer = srcBufferIter.second;
			for (auto& dstBufferIter : buffers)
			{
				const uint32_t dstMemIndex = dstBufferIter.first;
				const Buffer& dstBuffer = dstBufferIter.second;

				double measurements[kNumOfMeasurements] = {};
				double minTime = FLT_MAX;
				double maxTime = -FLT_MAX;
				double averageTime = 0.0f;
				VkDeviceSize sizeCopied = 0;
				for (uint32_t i = 0; i < kNumOfMeasurements; i++)
				{
					double time = 0.0;
					std::tie(time, sizeCopied) = Copy(queue, dstBuffer, srcBuffer, bufSize);
					if (time < 0.0f)
						return -1;

					minTime = std::min(minTime, time);
					maxTime = std::max(maxTime, time);
					averageTime += time;
					measurements[i] = time;
				}
				averageTime /= kNumOfMeasurements;
				std::sort(measurements, measurements + kNumOfMeasurements);
				double medianTime = measurements[kNumOfMeasurements / 2];

				LogStdOut("  %u -> %u:\n", srcMemIndex, dstMemIndex);
				LogStdOut("    Min:     %.2fms %.2fmb/s\n", minTime * 1000.0f, ComputeSpeed(sizeCopied, minTime));
				LogStdOut("    Average: %.2fms %.2fmb/s\n", averageTime * 1000.0f, ComputeSpeed(sizeCopied, averageTime));
				LogStdOut("    Median:  %.2fms %.2fmb/s\n", medianTime * 1000.0f, ComputeSpeed(sizeCopied, medianTime));
				LogStdOut("    Max:     %.2fms %.2fmb/s\n", maxTime * 1000.0f, ComputeSpeed(sizeCopied, maxTime));
			}
		}
	}

	LogStdOut("memcpy:\n");
	for (auto& srcBufferIter : buffers)
	{
		const uint32_t srcMemIndex = srcBufferIter.first;
		const Buffer& srcBuffer = srcBufferIter.second;
		if (!srcBuffer.mappedAddr)
			continue;

		for (auto& dstBufferIter : buffers)
		{
			const uint32_t dstMemIndex = dstBufferIter.first;
			const Buffer& dstBuffer = dstBufferIter.second;
			if (!dstBuffer.mappedAddr)
				continue;

			double measurements[kNumOfMeasurements] = {};
			double minTime = FLT_MAX;
			double maxTime = -FLT_MAX;
			double averageTime = 0.0f;
			VkDeviceSize sizeCopied = 0;
			for (uint32_t i = 0; i < kNumOfMeasurements; i++)
			{
				double time = 0.0;
				std::tie(time, sizeCopied) = Copy(dstBuffer, srcBuffer, bufSize);
				if (time < 0.0f)
					return -1;

				minTime = std::min(minTime, time);
				maxTime = std::max(maxTime, time);
				averageTime += time;
				measurements[i] = time;
			}
			averageTime /= kNumOfMeasurements;
			std::sort(measurements, measurements + kNumOfMeasurements);
			double medianTime = measurements[kNumOfMeasurements / 2];

			LogStdOut("  %u -> %u:\n", srcMemIndex, dstMemIndex);
			LogStdOut("    Min:     %.2fms %.2fmb/s\n", minTime * 1000.0f, ComputeSpeed(sizeCopied, minTime));
			LogStdOut("    Average: %.2fms %.2fmb/s\n", averageTime * 1000.0f, ComputeSpeed(sizeCopied, averageTime));
			LogStdOut("    Median:  %.2fms %.2fmb/s\n", medianTime * 1000.0f, ComputeSpeed(sizeCopied, medianTime));
			LogStdOut("    Max:     %.2fms %.2fmb/s\n", maxTime * 1000.0f, ComputeSpeed(sizeCopied, maxTime));
		}
	}

    return 0;
}

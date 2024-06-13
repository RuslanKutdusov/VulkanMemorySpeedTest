#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan.h>
#ifdef _WIN64 
#define NOMINMAX
#include <windows.h>
#endif
#include <float.h>
#include <vector>
#include <optional>
#include <map>
#include <algorithm>
#include <tuple>

struct GpuMemoryProps
{
	uint32_t memoryTypeIndex;
	bool deviceCoherent;
	bool deviceUncached;
	bool hostVisible;
	bool hostCoherent;
	bool hostCached;
};


struct CpuMemoryProps
{
	uint32_t memoryTypeIndex;
	bool deviceCoherent;
	bool deviceUncached;
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
	bool supportTimestamps;
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
static bool GHasDeviceCoherentMemory = false;
static int32_t GQueueFamilyIndices[kQueueTypesCount] = {-1, -1, -1};
static bool GQueueFamilySupportTimestamp[kQueueTypesCount] = {};
static float GTimestampPeriod = 0.0f;
static std::vector<Queue> GQueues;
static std::vector<GpuMemoryProps> GGpuMemory;
static std::vector<CpuMemoryProps> GCpuMemory;
static VkFence GFence;
static VkQueryPool GQueryPool;
static double GInvPerfFrequency = 0.0f;
static const char* kQueueTypeStr[] = { "Graphics", "Compute", "Transfer" };


static void LogStdOut(const char* format, ...)
{
	char buf[2048];
	va_list args;
	va_start(args, format);
	int len = vsprintf(buf, format, args);
	va_end(args);
	fwrite(buf, 1, len, stdout);
#ifdef _WIN64
	OutputDebugStringA(buf);
#endif
}


static void LogStdErr(const char* format, ...)
{
	char buf[2048];
	va_list args;
	va_start(args, format);
	int len = vsprintf(buf, format, args);
	va_end(args);
	fwrite(buf, 1, len, stderr);
#ifdef _WIN64
	OutputDebugStringA(buf);
#endif
}


template <class T>
static T SetBit(T& mask, uint32_t bitIndex)
{
	mask |= static_cast<T>(1) << bitIndex;
	return mask;
}


template <class T>
static T ResetBit(T& mask, uint32_t bitIndex)
{
	mask &= ~(static_cast<T>(1) << bitIndex);
	return mask;
}


template <class T>
static T ToggleBit(T& mask, uint32_t bitIndex)
{
	mask ^= static_cast<T>(1) << bitIndex;
	return mask;
}


template <class T>
static bool TestBit(const T& mask, uint32_t bitIndex)
{
	return mask & (static_cast<T>(1) << bitIndex);
}


#define Countof(array) (sizeof(array)/sizeof(array[0]))


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
				GQueueFamilySupportTimestamp[typeIdx] = famityProp.timestampValidBits != 0;
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

		if (!GHasDeviceCoherentMemory && (memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD) != 0)
			continue;

		if (memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
		{
			GpuMemoryProps props;
			props.memoryTypeIndex = i;
			props.deviceCoherent = memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD;
			props.deviceUncached = memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
			props.hostVisible = memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
			props.hostCoherent = memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			props.hostCached = memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
			GGpuMemory.push_back(props);
			LogStdOut("  Type:            GPU(device)\n");
			if (GHasDeviceCoherentMemory)
			{
				LogStdOut("  Device coherent: %s\n", props.deviceCoherent ? "true" : "false");
				LogStdOut("  Device cached:   %s\n", props.deviceUncached ? "true" : "false");
			}
			LogStdOut("  Host visible:    %s\n", props.hostVisible ? "true" : "false");
			LogStdOut("  Host coherent:   %s\n", props.hostCoherent ? "true" : "false");
			LogStdOut("  Host cached:     %s\n", props.hostCached ? "true" : "false");
		}
		else if (memType.propertyFlags != 0)
		{
			CpuMemoryProps props;
			props.memoryTypeIndex = i;
			props.deviceCoherent = memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD;
			props.deviceUncached = memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
			props.hostCoherent = memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			props.hostCached = memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
			GCpuMemory.push_back(props);
			LogStdOut("  Type:            CPU(host)\n");
			if (GHasDeviceCoherentMemory)
			{
				LogStdOut("  Device coherent: %s\n", props.deviceCoherent ? "true" : "false");
				LogStdOut("  Device cached:   %s\n", props.deviceUncached ? "true" : "false");
			}
			LogStdOut("  Host coherent:   %s\n", props.hostCoherent ? "true" : "false");
			LogStdOut("  Host cached:     %s\n", props.hostCached ? "true" : "false");
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

	queue.supportTimestamps = GQueueFamilySupportTimestamp[type];

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
	createInfo.enabledExtensionCount = Countof(instanceExtensions);

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

	VkPhysicalDeviceProperties physDeviceProps = {};
	vkGetPhysicalDeviceProperties(GVkPhysDevice, &physDeviceProps);
	GTimestampPeriod = physDeviceProps.limits.timestampPeriod;

	VkPhysicalDeviceCoherentMemoryFeaturesAMD deviceCoherentFeature = {};
	deviceCoherentFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD;

	VkPhysicalDeviceFeatures2 features = {};
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	features.pNext = &deviceCoherentFeature;
	vkGetPhysicalDeviceFeatures2(GVkPhysDevice, &features);

	GHasDeviceCoherentMemory = deviceCoherentFeature.deviceCoherentMemory;

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
	if (GHasDeviceCoherentMemory)
		deviceCreateInfo.pNext = &deviceCoherentFeature;
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

#ifdef _WIN64
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	GInvPerfFrequency = 1.0 / frequency.QuadPart;
#elif __linux__
	// TODO
	GInvPerfFrequency = 0.0;
#endif

	VkQueryPoolCreateInfo queryPoolCreateInfo = {};
	queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
	queryPoolCreateInfo.queryCount = 2;
	result = vkCreateQueryPool(GVkDevice, &queryPoolCreateInfo, nullptr, &GQueryPool);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkCreateQueryPool failed\n");
		return {};
	}

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
	buffer.size = size;
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
#ifdef _WIN64
	LARGE_INTEGER count;
	QueryPerformanceCounter(&count);
	return count.QuadPart;
#elif __linux__
	return __rdtsc();
#else
	return 0;
#endif
}


static double SubmitAndMeasureTime(const Queue& queue)
{
	VkResult result = vkResetFences(GVkDevice, 1, &GFence);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkResetFences failed\n");
		return -1.0;
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
		return -1.0;
	}

	result = vkWaitForFences(GVkDevice, 1, &GFence, VK_TRUE, ~0llu);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkWaitForFences failed\n");
		return -1.0;
	}

	uint64_t cpuEndTime = GetCpuTimestamp();

	double time = 0.0f;
	if (queue.supportTimestamps)
	{
		uint64_t queryResult[2] = {};
		result = vkGetQueryPoolResults(GVkDevice, GQueryPool, 0, 2, sizeof(queryResult), queryResult, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		if (result != VK_SUCCESS)
		{
			LogStdErr("vkGetQueryPoolResults failed\n");
			return -1.0;
		}

		time = (queryResult[1] - queryResult[0]) * GTimestampPeriod * 1e-09;
	}
	else
	{
		time = (cpuEndTime - cpuBeginTime) * GInvPerfFrequency;
	}
	return time;
}


static bool ResetQueryPool(const Queue& queue)
{
	if (queue.type == kQueueTransfer)
	{
		const Queue& gfxQueue = GQueues[kQueueGfx];

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		VkResult result = vkBeginCommandBuffer(gfxQueue.cmdBuffer, &beginInfo);
		if (result != VK_SUCCESS)
		{
			LogStdErr("vkBeginCommandBuffer failed\n");
			return false;
		}

		vkCmdResetQueryPool(gfxQueue.cmdBuffer, GQueryPool, 0, 2);

		result = vkEndCommandBuffer(gfxQueue.cmdBuffer);
		if (result != VK_SUCCESS)
		{
			LogStdErr("vkEndCommandBuffer failed\n");
			return false;
		}

		result = vkResetFences(GVkDevice, 1, &GFence);
		if (result != VK_SUCCESS)
		{
			LogStdErr("vkResetFences failed\n");
			return false;
		}

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &gfxQueue.cmdBuffer;
		result = vkQueueSubmit(gfxQueue.queue, 1, &submitInfo, GFence);
		if (result != VK_SUCCESS)
		{
			LogStdErr("vkQueueSubmit failed\n");
			return false;
		}

		result = vkWaitForFences(GVkDevice, 1, &GFence, VK_TRUE, ~0llu);
		if (result != VK_SUCCESS)
		{
			LogStdErr("vkWaitForFences failed\n");
			return false;
		}
	}
	else
	{
		vkCmdResetQueryPool(queue.cmdBuffer, GQueryPool, 0, 2);
	}
	return true;
}


static std::tuple<double, VkDeviceSize> Copy(const Queue& queue, const Buffer& dstBuffer, const Buffer& srcBuffer)
{
	VkBufferCopy bufferCopy = {};
	if (dstBuffer.buffer != srcBuffer.buffer)
	{
		bufferCopy.size = dstBuffer.size;
	}
	else
	{
		bufferCopy.dstOffset = dstBuffer.size / 2;
		bufferCopy.size = dstBuffer.size / 2;
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

	if (queue.supportTimestamps)
	{
		if (!ResetQueryPool(queue))
			return { -1.0f, 0 };
		vkCmdWriteTimestamp(queue.cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, GQueryPool, 0);
	}

	vkCmdCopyBuffer(queue.cmdBuffer, srcBuffer.buffer, dstBuffer.buffer, 1, &bufferCopy);

	if (queue.supportTimestamps)
		vkCmdWriteTimestamp(queue.cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, GQueryPool, 1);

	result = vkEndCommandBuffer(queue.cmdBuffer);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkEndCommandBuffer failed\n");
		return { -1.0f, 0 };
	}

	double time = SubmitAndMeasureTime(queue);
	return { time, bufferCopy.size };
}


static std::tuple<double, VkDeviceSize> Fill(const Queue& queue, const Buffer& dstBuffer)
{
	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VkResult result = vkBeginCommandBuffer(queue.cmdBuffer, &beginInfo);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkBeginCommandBuffer failed\n");
		return { -1.0f, 0 };
	}

	if (queue.supportTimestamps)
	{
		ResetQueryPool(queue);
		vkCmdWriteTimestamp(queue.cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, GQueryPool, 0);
	}

	vkCmdFillBuffer(queue.cmdBuffer, dstBuffer.buffer, 0, dstBuffer.size, 0);

	if (queue.supportTimestamps)
		vkCmdWriteTimestamp(queue.cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, GQueryPool, 1);

	result = vkEndCommandBuffer(queue.cmdBuffer);
	if (result != VK_SUCCESS)
	{
		LogStdErr("vkEndCommandBuffer failed\n");
		return { -1.0f, 0 };
	}

	double time = SubmitAndMeasureTime(queue);
	return { time, dstBuffer.size };
}


static std::tuple<double, VkDeviceSize> Copy(const Buffer& dstBuffer, const Buffer& srcBuffer)
{
	uint8_t* dst = (uint8_t*)dstBuffer.mappedAddr;
	uint8_t* src = (uint8_t*)srcBuffer.mappedAddr;
	VkDeviceSize size = dstBuffer.size;
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


static std::tuple<double, VkDeviceSize> Fill(const Buffer& dstBuffer)
{
	uint64_t cpuBeginTime = GetCpuTimestamp();
	memset(dstBuffer.mappedAddr, 0, dstBuffer.size);
	uint64_t cpuEndTime = GetCpuTimestamp();
	return { (cpuEndTime - cpuBeginTime) * GInvPerfFrequency, dstBuffer.size };
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
			continue;
		buffers[gpuMem.memoryTypeIndex] = *buf;
	}

	for (const CpuMemoryProps& cpuMem : GCpuMemory)
	{
		std::optional<Buffer> buf = CreateBuffer(bufSize, cpuMem.memoryTypeIndex, true);
		if (!buf.has_value())
			continue;
		buffers[cpuMem.memoryTypeIndex] = *buf;
	}

	for (const Queue& queue : GQueues)
	{
		LogStdOut("vkCmdCopyBuffer on %s queue%s:\n", kQueueTypeStr[queue.type], queue.supportTimestamps ? "" : "(inaccurate)");
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
					std::tie(time, sizeCopied) = Copy(queue, dstBuffer, srcBuffer);
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
				std::tie(time, sizeCopied) = Copy(dstBuffer, srcBuffer);
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

	for (const Queue& queue : GQueues)
	{
		LogStdOut("vkCmdFillBuffer on %s queue%s:\n", kQueueTypeStr[queue.type], queue.supportTimestamps ? "" : "(inaccurate)");
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
				std::tie(time, sizeCopied) = Fill(queue, dstBuffer);
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

			LogStdOut("  %u:\n", dstMemIndex);
			LogStdOut("    Min:     %.2fms %.2fmb/s\n", minTime * 1000.0f, ComputeSpeed(sizeCopied, minTime));
			LogStdOut("    Average: %.2fms %.2fmb/s\n", averageTime * 1000.0f, ComputeSpeed(sizeCopied, averageTime));
			LogStdOut("    Median:  %.2fms %.2fmb/s\n", medianTime * 1000.0f, ComputeSpeed(sizeCopied, medianTime));
			LogStdOut("    Max:     %.2fms %.2fmb/s\n", maxTime * 1000.0f, ComputeSpeed(sizeCopied, maxTime));
		}
	}

	LogStdOut("memset:\n");
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
			std::tie(time, sizeCopied) = Fill(dstBuffer);
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

		LogStdOut("  %u:\n", dstMemIndex);
		LogStdOut("    Min:     %.2fms %.2fmb/s\n", minTime * 1000.0f, ComputeSpeed(sizeCopied, minTime));
		LogStdOut("    Average: %.2fms %.2fmb/s\n", averageTime * 1000.0f, ComputeSpeed(sizeCopied, averageTime));
		LogStdOut("    Median:  %.2fms %.2fmb/s\n", medianTime * 1000.0f, ComputeSpeed(sizeCopied, medianTime));
		LogStdOut("    Max:     %.2fms %.2fmb/s\n", maxTime * 1000.0f, ComputeSpeed(sizeCopied, maxTime));
	}

    return 0;
}

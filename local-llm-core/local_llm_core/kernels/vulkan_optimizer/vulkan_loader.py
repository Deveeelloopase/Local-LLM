"""
vulkan_loader.py — Full Vulkan compute pipeline for 1‑bit matmul.
=================================================================
Compiles SPIR‑V and dispatches the ternary matrix multiplication
to any Vulkan‑capable GPU.  Falls back to PyTorch on error.

Dependencies:
    pip install vulkan
    Vulkan SDK (glslc) must be installed.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import torch

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# SPIR‑V compilation (cached)
# -----------------------------------------------------------------------------
def _find_glslc() -> Optional[str]:
    for candidate in ["glslc", "/usr/local/bin/glslc"]:
        try:
            subprocess.run(
                [candidate, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return candidate
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return None

def _compile_shader() -> Optional[bytes]:
    glslc = _find_glslc()
    if not glslc:
        log.warning("glslc not found; cannot compile Vulkan shader")
        return None
    comp_path = Path(__file__).parent / "matmul_1bit.comp"
    spv_path = Path(tempfile.gettempdir()) / "ternary_matmul.spv"
    if spv_path.exists():
        log.debug("Reusing cached SPIR‑V: %s", spv_path)
        return spv_path.read_bytes()
    cmd = [glslc, "-o", str(spv_path), str(comp_path)]
    log.info("Compiling Vulkan shader: %s", " ".join(cmd))
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        log.error("Vulkan shader compile failed:\n%s", ret.stderr)
        return None
    return spv_path.read_bytes()

# -----------------------------------------------------------------------------
# Vulkan device / pipeline setup (lazy, cached)
# -----------------------------------------------------------------------------
_instance = None
_device = None
_queue = None
_pipeline = None
_pipeline_layout = None
_descriptor_set_layout = None
_command_pool = None

def _init_vulkan() -> bool:
    _VK_AVAIL = False
    try:
        import vulkan as vk  # inside a try block
        import numpy as _np
        _VK_AVAIL = True
    except ImportError:
        pass

    global _instance, _device, _queue, _pipeline, _pipeline_layout
    global _descriptor_set_layout, _command_pool

    if _pipeline is not None:
        return True

    spv = _compile_shader()
    if spv is None:
        return False

    try:
        # Instance
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="llm_vulkan",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_2,
        )
        instance_create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )
        _instance = vk.vkCreateInstance(instance_create_info, None)

        # Device (pick first GPU that supports compute)
        physical_devices = vk.vkEnumeratePhysicalDevices(_instance)
        if not physical_devices:
            raise RuntimeError("No Vulkan physical devices")
        _device = None
        for pd in physical_devices:
            props = vk.vkGetPhysicalDeviceProperties(pd)
            queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(pd)
            for i, qf in enumerate(queue_families):
                if qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                    queue_family_index = i
                    _device = pd
                    break
            if _device:
                log.info("Selected Vulkan device: %s", props.deviceName)
                break
        if not _device:
            raise RuntimeError("No compute‑capable Vulkan device")

        # Queue
        queue_priority = 1.0
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            queueCount=1,
            pQueuePriorities=[queue_priority],
        )
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
        )
        _device = vk.vkCreateDevice(_device, device_create_info, None)
        _queue = vk.vkGetDeviceQueue(_device, queue_family_index, 0)

        # Descriptor set layout (4 storage buffers + push constants)
        layout_bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=3,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        layout_create_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(layout_bindings),
            pBindings=layout_bindings,
        )
        _descriptor_set_layout = vk.vkCreateDescriptorSetLayout(_device, layout_create_info, None)

        # Pipeline layout (with push constants for M,N,K)
        push_constant_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=12,  # 3 uint32
        )
        pipeline_layout_create_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[_descriptor_set_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_constant_range],
        )
        _pipeline_layout = vk.vkCreatePipelineLayout(_device, pipeline_layout_create_info, None)

        # Shader module
        shader_module_create_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(spv),
            pCode=ctypes.cast(spv, ctypes.POINTER(ctypes.c_uint32)),
        )
        shader_module = vk.vkCreateShaderModule(_device, shader_module_create_info, None)

        # Compute pipeline
        pipeline_create_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                module=shader_module,
                pName="main",
            ),
            layout=_pipeline_layout,
        )
        _pipeline = vk.vkCreateComputePipelines(_device, None, 1, [pipeline_create_info], None)[0]
        vk.vkDestroyShaderModule(_device, shader_module, None)

        # Command pool
        pool_create_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        _command_pool = vk.vkCreateCommandPool(_device, pool_create_info, None)

        log.info("Vulkan pipeline initialised successfully")
        return True

    except Exception as e:
        log.error("Vulkan init failed: %s", e, exc_info=True)
        _cleanup()
        return False

def _cleanup():
    _VK_AVAIL = False
    try:
        import vulkan as vk  # inside a try block
        import numpy as _np
        _VK_AVAIL = True
    except ImportError:
        pass

    global _instance, _device, _queue, _pipeline, _pipeline_layout
    global _descriptor_set_layout, _command_pool
    if _pipeline:
        vk.vkDestroyPipeline(_device, _pipeline, None)
        _pipeline = None
    if _pipeline_layout:
        vk.vkDestroyPipelineLayout(_device, _pipeline_layout, None)
        _pipeline_layout = None
    if _descriptor_set_layout:
        vk.vkDestroyDescriptorSetLayout(_device, _descriptor_set_layout, None)
        _descriptor_set_layout = None
    if _command_pool:
        vk.vkDestroyCommandPool(_device, _command_pool, None)
        _command_pool = None
    if _device:
        vk.vkDestroyDevice(_device, None)
        _device = None
    if _instance:
        vk.vkDestroyInstance(_instance, None)
        _instance = None

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def vulkan_ternary_matmul(a: torch.Tensor, b_packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Execute ternary matmul on Vulkan compute.
    a: (M, K) float16 (on CPU or GPU)
    b_packed: (K, N) int8 packed (on CPU, 2 bits per element)
    scale: (N,) float32
    Returns: (M, N) float16 tensor on CPU.
    """
    try:
        import numpy as np
        import vulkan as vk
    except ImportError:
        log.warning("vulkan or numpy not installed; falling back to PyTorch")
        return torch.matmul(a.float(), (b_packed.float() * scale)).half()

    if not _init_vulkan():
        log.warning("Vulkan unavailable; falling back to PyTorch")
        return torch.matmul(a.float(), (b_packed.float() * scale)).half()

    M, K = a.shape
    _, N = b_packed.shape

    # Ensure tensors are contiguous and on CPU
    a_cpu = a.contiguous().cpu()
    b_cpu = b_packed.contiguous().cpu()
    scale_cpu = scale.contiguous().cpu()
    c_cpu = torch.empty(M, N, dtype=torch.float16, device='cpu')

    try:
        # Create buffers
        buf_a = vk.create_buffer_with_data(_device, a_cpu.numpy().tobytes(), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_b = vk.create_buffer_with_data(_device, b_cpu.numpy().tobytes(), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_scale = vk.create_buffer_with_data(_device, scale_cpu.numpy().tobytes(), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_c = vk.create_buffer(_device, c_cpu.numpy().nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_TRANSFER_DST_BIT)

        # Descriptor set
        descriptor_pool_sizes = [vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=4)]
        pool_create_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1,
            poolSizeCount=len(descriptor_pool_sizes),
            pPoolSizes=descriptor_pool_sizes,
        )
        descriptor_pool = vk.vkCreateDescriptorPool(_device, pool_create_info, None)
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[_descriptor_set_layout],
        )
        descriptor_set = vk.vkAllocateDescriptorSets(_device, alloc_info)[0]

        # Update descriptor set
        writes = [
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=0,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=buf_a, offset=0, range=vk.VK_WHOLE_SIZE)],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=1,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=buf_b, offset=0, range=vk.VK_WHOLE_SIZE)],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=2,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=buf_scale, offset=0, range=vk.VK_WHOLE_SIZE)],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=3,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=buf_c, offset=0, range=vk.VK_WHOLE_SIZE)],
            ),
        ]
        vk.vkUpdateDescriptorSets(_device, len(writes), writes, 0, None)

        # Command buffer
        alloc_info_cb = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=_command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd_buf = vk.vkAllocateCommandBuffers(_device, alloc_info_cb)[0]

        begin_info = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
        vk.vkBeginCommandBuffer(cmd_buf, begin_info)
        vk.vkCmdBindPipeline(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline)
        vk.vkCmdBindDescriptorSets(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline_layout, 0, 1, [descriptor_set], 0, None)

        # Push constants (M, N, K)
        push_data = np.array([M, N, K], dtype=np.uint32).tobytes()
        vk.vkCmdPushConstants(cmd_buf, _pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push_data)

        # Dispatch
        group_size_x = 16
        group_size_y = 16
        vk.vkCmdDispatch(cmd_buf, (M + group_size_x - 1) // group_size_x, (N + group_size_y - 1) // group_size_y, 1)
        vk.vkEndCommandBuffer(cmd_buf)

        # Submit
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd_buf],
        )
        fence_create_info = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = vk.vkCreateFence(_device, fence_create_info, None)
        vk.vkQueueSubmit(_queue, 1, [submit_info], fence)
        vk.vkWaitForFences(_device, 1, [fence], vk.VK_TRUE, 10_000_000_000)  # 10s timeout

        # Read result
        c_bytes = vk.read_buffer(_device, buf_c, c_cpu.numpy().nbytes)
        c_result = torch.frombuffer(bytearray(c_bytes), dtype=torch.float16).reshape(M, N)

        # Cleanup temporary objects
        vk.vkDestroyFence(_device, fence, None)
        vk.vkFreeCommandBuffers(_device, _command_pool, 1, [cmd_buf])
        vk.vkDestroyDescriptorPool(_device, descriptor_pool, None)
        vk.destroy_buffer(_device, buf_a)
        vk.destroy_buffer(_device, buf_b)
        vk.destroy_buffer(_device, buf_scale)
        vk.destroy_buffer(_device, buf_c)

        return c_result

    except Exception as e:
        log.error("Vulkan dispatch failed: %s", e, exc_info=True)
        log.warning("Falling back to PyTorch matmul")
        return torch.matmul(a.float(), (b_packed.float() * scale)).half()

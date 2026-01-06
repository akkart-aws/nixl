/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cuda_accelerator_device.h"
#include "common/nixl_log.h"

#ifdef HAVE_CUDA

CudaAcceleratorDevice::CudaAcceleratorDevice() 
    : context_(nullptr), device_id_(-1), address_workaround_enabled_(true) {
    
    // Check environment variable to disable workaround
    if (getenv("NIXL_DISABLE_CUDA_ADDR_WA")) {
        address_workaround_enabled_ = false;
        NIXL_DEBUG << "CUDA address workaround disabled via environment variable";
    }
}

CudaAcceleratorDevice::~CudaAcceleratorDevice() {
    finalizeContext();
}

nixl_status_t CudaAcceleratorDevice::initContext() {
    context_ = nullptr;
    device_id_ = -1;
    NIXL_DEBUG << "CUDA accelerator context initialized";
    return NIXL_SUCCESS;
}

nixl_status_t CudaAcceleratorDevice::cudaQueryAddr(void* address, bool& is_dev, 
                                                   CUdevice& dev, CUcontext& ctx) {
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
    CUpointer_attribute attr_type[4];
    void* attr_data[4];

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &dev;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &ctx;

    CUresult result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);
    is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);

    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        NIXL_ERROR << "CUDA pointer query failed: " << error_str;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t CudaAcceleratorDevice::updateContext(void* address, uint64_t device_id, 
                                                   bool& restart_required) {
    restart_required = false;

    if (device_id == static_cast<uint64_t>(-1)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    
    if (device_id_ != -1 && device_id != static_cast<uint64_t>(device_id_)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    bool is_dev;
    CUdevice dev;
    CUcontext ctx;
    
    nixl_status_t status = cudaQueryAddr(address, is_dev, dev, ctx);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    if (!is_dev) {
        return NIXL_SUCCESS;
    }

    if (dev != static_cast<CUdevice>(device_id)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (context_) {
        if (context_ != ctx) {
            return NIXL_ERR_INVALID_PARAM;
        }
        return NIXL_SUCCESS;
    }

    context_ = ctx;
    restart_required = true;
    device_id_ = device_id;

    NIXL_DEBUG << "CUDA context updated for device " << device_id_;
    return NIXL_SUCCESS;
}

nixl_status_t CudaAcceleratorDevice::applyContext() {
    if (!context_) {
        return NIXL_SUCCESS;
    }

    CUresult result = cuCtxSetCurrent(context_);
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        NIXL_ERROR << "Failed to set CUDA context: " << error_str;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

void CudaAcceleratorDevice::finalizeContext() {
    context_ = nullptr;
    device_id_ = -1;
    NIXL_DEBUG << "CUDA accelerator context finalized";
}

nixl_status_t CudaAcceleratorDevice::setDevice(int device_id) {
    cudaError_t result = cudaSetDevice(device_id);
    if (result != cudaSuccess) {
        NIXL_ERROR << "Failed to set CUDA device " << device_id 
                   << ": " << cudaGetErrorString(result);
        return NIXL_ERR_BACKEND;
    }

    device_id_ = device_id;
    NIXL_DEBUG << "Set CUDA device to " << device_id;
    return NIXL_SUCCESS;
}

nixl_status_t CudaAcceleratorDevice::queryAddress(void* address, bool& is_device_mem, 
                                                  int& device_id) {
    CUdevice dev;
    CUcontext ctx;
    
    nixl_status_t status = cudaQueryAddr(address, is_device_mem, dev, ctx);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    if (is_device_mem) {
        device_id = static_cast<int>(dev);
    }

    return NIXL_SUCCESS;
}

#endif // HAVE_CUDA

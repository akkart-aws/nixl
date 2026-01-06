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
#ifndef NIXL_SRC_UTILS_ACCELERATOR_CUDA_ACCELERATOR_DEVICE_H
#define NIXL_SRC_UTILS_ACCELERATOR_CUDA_ACCELERATOR_DEVICE_H

#include "accelerator_device.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief CUDA-specific accelerator device implementation
 * 
 * Handles CUDA context management, device queries, and memory operations.
 * Implements workarounds for CUDA-specific issues like context switching.
 */
class CudaAcceleratorDevice : public AcceleratorDevice {
private:
    CUcontext context_;
    int device_id_;
    bool address_workaround_enabled_;

    /**
     * @brief Query CUDA address attributes
     * @param address Memory address to query
     * @param is_dev Output: true if device memory
     * @param dev Output: CUDA device
     * @param ctx Output: CUDA context
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t cudaQueryAddr(void* address, bool& is_dev, CUdevice& dev, CUcontext& ctx);

public:
    CudaAcceleratorDevice();
    ~CudaAcceleratorDevice() override;

    // AcceleratorDevice interface implementation
    nixl_status_t initContext() override;
    nixl_status_t updateContext(void* address, uint64_t device_id, 
                               bool& restart_required) override;
    nixl_status_t applyContext() override;
    void finalizeContext() override;
    nixl_status_t setDevice(int device_id) override;
    nixl_status_t queryAddress(void* address, bool& is_device_mem, int& device_id) override;
    
    enum fi_hmem_iface getMemoryInterface() const override { return FI_HMEM_CUDA; }
    const char* getTypeName() const override { return "CUDA"; }
    bool needsAddressWorkaround() const override { return address_workaround_enabled_; }
};

#endif // HAVE_CUDA
#endif // NIXL_SRC_UTILS_ACCELERATOR_CUDA_ACCELERATOR_DEVICE_H

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
#ifndef NIXL_SRC_UTILS_ACCELERATOR_ACCELERATOR_DEVICE_H
#define NIXL_SRC_UTILS_ACCELERATOR_ACCELERATOR_DEVICE_H

#include "nixl.h"
#include <rdma/fi_domain.h>
#include <memory>

/**
 * @brief Abstract base class for accelerator device operations
 * 
 * Provides a unified interface for different accelerator types (CUDA, Neuron, ROCm, etc.)
 * to handle device context management, memory queries, and device-specific operations.
 */
class AcceleratorDevice {
public:
    virtual ~AcceleratorDevice() = default;

    /**
     * @brief Initialize device context management
     * @return NIXL_SUCCESS on success, error code on failure
     */
    virtual nixl_status_t initContext() = 0;

    /**
     * @brief Update context for a specific memory address and device
     * @param address Memory address to query
     * @param device_id Device ID
     * @param restart_required Output: true if context changed and restart needed
     * @return NIXL_SUCCESS on success, error code on failure
     */
    virtual nixl_status_t updateContext(void* address, uint64_t device_id, 
                                       bool& restart_required) = 0;

    /**
     * @brief Apply/set the current device context
     * @return NIXL_SUCCESS on success, error code on failure
     */
    virtual nixl_status_t applyContext() = 0;

    /**
     * @brief Finalize and cleanup device context
     */
    virtual void finalizeContext() = 0;

    /**
     * @brief Set the active device
     * @param device_id Device ID to set as active
     * @return NIXL_SUCCESS on success, error code on failure
     */
    virtual nixl_status_t setDevice(int device_id) = 0;

    /**
     * @brief Query if an address is device memory and get device info
     * @param address Memory address to query
     * @param is_device_mem Output: true if device memory
     * @param device_id Output: device ID if device memory
     * @return NIXL_SUCCESS on success, error code on failure
     */
    virtual nixl_status_t queryAddress(void* address, bool& is_device_mem, 
                                      int& device_id) = 0;

    /**
     * @brief Get the libfabric memory interface type for this accelerator
     * @return fi_hmem_iface type (FI_HMEM_CUDA, FI_HMEM_NEURON, etc.)
     */
    virtual enum fi_hmem_iface getMemoryInterface() const = 0;

    /**
     * @brief Get accelerator type name for logging
     * @return String representation of accelerator type
     */
    virtual const char* getTypeName() const = 0;

    /**
     * @brief Check if address workaround is needed for this accelerator
     * @return true if workaround needed, false otherwise
     */
    virtual bool needsAddressWorkaround() const = 0;
};

/**
 * @brief Factory for creating accelerator device instances
 */
class AcceleratorDeviceFactory {
public:
    /**
     * @brief Create accelerator device based on interface type
     * @param iface Libfabric memory interface type
     * @return Unique pointer to accelerator device, or nullptr if not supported
     */
    static std::unique_ptr<AcceleratorDevice> create(enum fi_hmem_iface iface);

    /**
     * @brief Check if accelerator type is available at compile time
     * @param iface Libfabric memory interface type
     * @return true if available, false otherwise
     */
    static bool isAvailable(enum fi_hmem_iface iface);
};

#endif // NIXL_SRC_UTILS_ACCELERATOR_ACCELERATOR_DEVICE_H

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
#ifndef NIXL_SRC_UTILS_ACCELERATOR_NULL_ACCELERATOR_DEVICE_H
#define NIXL_SRC_UTILS_ACCELERATOR_NULL_ACCELERATOR_DEVICE_H

#include "accelerator_device.h"

/**
 * @brief Null accelerator device for DRAM-only systems
 * 
 * All operations are no-ops since no accelerator is present.
 * Used as a fallback when no hardware accelerator is available.
 */
class NullAcceleratorDevice : public AcceleratorDevice {
public:
    NullAcceleratorDevice() = default;
    ~NullAcceleratorDevice() override = default;

    nixl_status_t initContext() override { return NIXL_SUCCESS; }
    
    nixl_status_t updateContext(void*, uint64_t, bool&) override { 
        return NIXL_SUCCESS; 
    }
    
    nixl_status_t applyContext() override { return NIXL_SUCCESS; }
    
    void finalizeContext() override {}
    
    nixl_status_t setDevice(int) override { return NIXL_SUCCESS; }
    
    nixl_status_t queryAddress(void*, bool& is_device_mem, int&) override {
        is_device_mem = false;
        return NIXL_SUCCESS;
    }
    
    enum fi_hmem_iface getMemoryInterface() const override { return FI_HMEM_SYSTEM; }
    const char* getTypeName() const override { return "SYSTEM"; }
    bool needsAddressWorkaround() const override { return false; }
};

#endif // NIXL_SRC_UTILS_ACCELERATOR_NULL_ACCELERATOR_DEVICE_H

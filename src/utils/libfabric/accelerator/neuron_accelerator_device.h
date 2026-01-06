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
#ifndef NIXL_SRC_UTILS_ACCELERATOR_NEURON_ACCELERATOR_DEVICE_H
#define NIXL_SRC_UTILS_ACCELERATOR_NEURON_ACCELERATOR_DEVICE_H

#include "accelerator_device.h"

/**
 * @brief AWS Neuron accelerator device implementation
 * 
 * Neuron devices don't require complex context management like CUDA,
 * so most operations are no-ops or simplified. Neuron uses a different
 * programming model where context management is handled by the runtime.
 */
class NeuronAcceleratorDevice : public AcceleratorDevice {
private:
    int current_device_id_;

public:
    NeuronAcceleratorDevice();
    ~NeuronAcceleratorDevice() override;

    // AcceleratorDevice interface implementation
    nixl_status_t initContext() override;
    nixl_status_t updateContext(void* address, uint64_t device_id, 
                               bool& restart_required) override;
    nixl_status_t applyContext() override;
    void finalizeContext() override;
    nixl_status_t setDevice(int device_id) override;
    nixl_status_t queryAddress(void* address, bool& is_device_mem, int& device_id) override;
    
    enum fi_hmem_iface getMemoryInterface() const override { return FI_HMEM_NEURON; }
    const char* getTypeName() const override { return "NEURON"; }
    bool needsAddressWorkaround() const override { return false; }
};

#endif // NIXL_SRC_UTILS_ACCELERATOR_NEURON_ACCELERATOR_DEVICE_H

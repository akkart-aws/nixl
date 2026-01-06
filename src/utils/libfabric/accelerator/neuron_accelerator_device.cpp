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
#include "neuron_accelerator_device.h"
#include "common/nixl_log.h"

NeuronAcceleratorDevice::NeuronAcceleratorDevice() : current_device_id_(-1) {
    NIXL_DEBUG << "Neuron accelerator device created";
}

NeuronAcceleratorDevice::~NeuronAcceleratorDevice() {
    finalizeContext();
}

nixl_status_t NeuronAcceleratorDevice::initContext() {
    current_device_id_ = -1;
    NIXL_DEBUG << "Neuron accelerator context initialized (no-op)";
    return NIXL_SUCCESS;
}

nixl_status_t NeuronAcceleratorDevice::updateContext(void* address, uint64_t device_id, 
                                                     bool& restart_required) {
    restart_required = false;
    // Neuron doesn't require context updates
    NIXL_TRACE << "Neuron context update (no-op) for device " << device_id;
    return NIXL_SUCCESS;
}

nixl_status_t NeuronAcceleratorDevice::applyContext() {
    // Neuron doesn't require explicit context application
    NIXL_TRACE << "Neuron apply context (no-op)";
    return NIXL_SUCCESS;
}

void NeuronAcceleratorDevice::finalizeContext() {
    current_device_id_ = -1;
    NIXL_DEBUG << "Neuron accelerator context finalized (no-op)";
}

nixl_status_t NeuronAcceleratorDevice::setDevice(int device_id) {
    current_device_id_ = device_id;
    NIXL_DEBUG << "Neuron device set to " << device_id << " (tracked for reference)";
    return NIXL_SUCCESS;
}

nixl_status_t NeuronAcceleratorDevice::queryAddress(void* address, bool& is_device_mem, 
                                                    int& device_id) {
    // For Neuron, we assume the caller knows if it's device memory
    // This is a simplified implementation - may need Neuron-specific APIs
    is_device_mem = false;
    device_id = -1;
    
    NIXL_TRACE << "Neuron address query (simplified implementation)";
    return NIXL_SUCCESS;
}

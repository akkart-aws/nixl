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
#include "accelerator_device.h"
#include "null_accelerator_device.h"
#include "neuron_accelerator_device.h"

#ifdef HAVE_CUDA
#include "cuda_accelerator_device.h"
#endif

#include "common/nixl_log.h"

std::unique_ptr<AcceleratorDevice> 
AcceleratorDeviceFactory::create(enum fi_hmem_iface iface) {
    switch (iface) {
#ifdef HAVE_CUDA
        case FI_HMEM_CUDA:
            NIXL_DEBUG << "Creating CUDA accelerator device";
            return std::make_unique<CudaAcceleratorDevice>();
#endif
        
        case FI_HMEM_NEURON:
            NIXL_DEBUG << "Creating Neuron accelerator device";
            return std::make_unique<NeuronAcceleratorDevice>();
        
        case FI_HMEM_SYSTEM:
            NIXL_DEBUG << "Creating null accelerator device (system memory)";
            return std::make_unique<NullAcceleratorDevice>();
        
        default:
            NIXL_WARN << "Unsupported accelerator interface type: " << iface 
                      << ", falling back to null device";
            return std::make_unique<NullAcceleratorDevice>();
    }
}

bool AcceleratorDeviceFactory::isAvailable(enum fi_hmem_iface iface) {
    switch (iface) {
#ifdef HAVE_CUDA
        case FI_HMEM_CUDA:
            return true;
#endif
        case FI_HMEM_NEURON:
            return true;
        
        case FI_HMEM_SYSTEM:
            return true;
        
        default:
            return false;
    }
}

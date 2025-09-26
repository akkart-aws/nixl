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

#include "libfabric_common.h"
#include "common/nixl_log.h"

#include <iomanip>
#include <sstream>
#include <atomic>
#include <cstring>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

namespace LibfabricUtils {


std::pair<std::string, std::vector<std::string>>
getAvailableNetworkDevices() {
    std::vector<std::string> all_devices;
    std::string fabric_name;

    // Priority order: EFA-Direct > EFA > TCP > Sockets
    std::vector<std::string> provider_priority = {"efa-direct", "efa", "tcp", "sockets"};

    for (const std::string &provider : provider_priority) {
        NIXL_TRACE << "Trying provider: " << provider;

        struct fi_info *hints, *info;
        hints = fi_allocinfo();
        if (!hints) {
            NIXL_ERROR << "Failed to allocate fi_info for " << provider << " discovery";
            continue;
        }

        // Configure hints based on provider capabilities
        if (provider == "tcp" || provider == "sockets") {
            // TCP/sockets providers use message endpoints and have different requirements
            hints->mode = 0; // Let provider set appropriate mode
            hints->caps = FI_MSG | FI_RMA; // Basic messaging and RMA
            hints->ep_attr->type = FI_EP_MSG; // Message endpoint for TCP
        } else {
            // EFA providers use reliable datagram endpoints
            hints->mode = 0;
            hints->caps = FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE;
            hints->ep_attr->type = FI_EP_RDM; // Reliable datagram for EFA
        }

        hints->fabric_attr->prov_name = strdup(provider.c_str());

        int ret = fi_getinfo(FI_VERSION(1, 9), NULL, NULL, 0, hints, &info);
        if (ret) {
            NIXL_TRACE << "fi_getinfo failed for " << provider << ": " << fi_strerror(-ret);
            fi_freeinfo(hints);
            continue; // Try next provider
        }

        NIXL_TRACE << "fi_getinfo succeeded for " << provider << ", processing results...";

        // Process devices for this provider
        std::vector<std::string> provider_devices;
        for (struct fi_info *cur = info; cur; cur = cur->next) {
            if (cur->domain_attr && cur->domain_attr->name && cur->fabric_attr &&
                cur->fabric_attr->name) {

                std::string device_name = cur->domain_attr->name;
                std::string fabric_name_returned = cur->fabric_attr->name;
                std::string provider_name_returned =
                    cur->fabric_attr->prov_name ? cur->fabric_attr->prov_name : "unknown";

                NIXL_TRACE << "Found device - domain: " << device_name
                           << ", fabric: " << fabric_name_returned
                           << ", provider: " << provider_name_returned
                           << ", ep_type: " << cur->ep_attr->type << ", caps: 0x" << std::hex
                           << cur->caps << std::dec;

                // Check if this matches the provider we're looking for
                if (provider_name_returned == provider) {
                    // For TCP provider, skip loopback devices
                    if (provider == "tcp" && device_name == "lo") {
                        NIXL_TRACE << "Skipping loopback device: " << device_name;
                        continue;
                    }

                    provider_devices.push_back(device_name);
                    NIXL_INFO << "Matched " << provider << " device: " << device_name
                              << " (fabric: " << fabric_name_returned << ")";
                }
            }
        }

        fi_freeinfo(info);
        fi_freeinfo(hints);

        // If we found devices with this provider, use them
        if (!provider_devices.empty()) {
            // For TCP/sockets providers, just use the first device to keep it simple
            if (provider == "tcp" || provider == "sockets") {
                all_devices = {provider_devices[0]}; // Only use first device
                NIXL_INFO << "Using " << provider
                          << " provider with first device: " << provider_devices[0];
            } else {
                all_devices = provider_devices; // Use all devices for EFA
                NIXL_INFO << "Using " << provider << " provider with " << all_devices.size()
                          << " devices";
            }
            fabric_name = provider;
            return {fabric_name, all_devices};
        }
    }

    // No devices found with any provider
    NIXL_WARN << "No network devices found with any provider";
    return {"none", {}};
}

std::string
hexdump(const void *data) {
    static constexpr uint HEXDUMP_MAX_LENGTH = 56;
    std::stringstream ss;
    ss.str().reserve(HEXDUMP_MAX_LENGTH * 3);
    const unsigned char *bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < HEXDUMP_MAX_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]) << " ";
    }
    return ss.str();
}

// Thread-safe atomic counter for dynamic allocation to ensure uniqueness
static std::atomic<uint32_t> g_xfer_id_counter{1}; // Start from 1, 0 reserved for special cases

uint32_t
getNextXferId() {
    uint32_t xfer_id = g_xfer_id_counter.fetch_add(1);

    // Handle wraparound: 20-bit field can hold 0 to 1,048,575
    if (xfer_id > NIXL_XFER_ID_MASK) {
        // Reset counter atomically and get a fresh ID
        uint32_t expected = xfer_id;
        while (expected > NIXL_XFER_ID_MASK &&
               !g_xfer_id_counter.compare_exchange_weak(expected, 1)) {
            expected = g_xfer_id_counter.load();
        }
        xfer_id = g_xfer_id_counter.fetch_add(1);
        // Ensure we don't exceed the mask after reset
        if (xfer_id > NIXL_XFER_ID_MASK) {
            xfer_id = 1;
        }
    }

    return xfer_id;
}

} // namespace LibfabricUtils

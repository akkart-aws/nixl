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
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LOCK_FREE_STACK_H
#define NIXL_SRC_UTILS_LIBFABRIC_LOCK_FREE_STACK_H

#include <atomic>
#include <cstddef>
#include <cstdint>

/**
 * @brief Lock-free stack using atomic Compare-And-Swap (CAS) operations
 *
 * This implementation uses tagged pointers to solve the ABA problem:
 * - Lower 48 bits: pointer to node (supports 256TB address space)
 * - Upper 16 bits: version counter (prevents ABA problem)
 *
 * Performance characteristics:
 * - Push: ~10-20 CPU cycles (uncontended)
 * - Pop: ~10-20 CPU cycles (uncontended)
 * - No locks, no blocking, true lock-free operation
 *
 * Memory ordering:
 * - Uses acquire-release semantics for correctness
 * - Ensures visibility across threads without barriers
 */
class LockFreeStack {
public:
    /**
     * @brief Initialize empty lock-free stack
     */
    LockFreeStack();

    /**
     * @brief Destructor - does NOT free nodes (caller manages memory)
     */
    ~LockFreeStack();

    // Non-copyable and non-movable
    LockFreeStack(const LockFreeStack &) = delete;
    LockFreeStack &operator=(const LockFreeStack &) = delete;
    LockFreeStack(LockFreeStack &&) = delete;
    LockFreeStack &operator=(LockFreeStack &&) = delete;

    /**
     * @brief Push index onto stack (lock-free)
     *
     * @param index Index to push
     * @return true if successful, false if out of memory
     *
     * Time complexity: O(1) amortized
     * Memory ordering: release (ensures all prior writes visible)
     */
    bool push(size_t index);

    /**
     * @brief Pop index from stack (lock-free)
     *
     * @param index Output parameter for popped index
     * @return true if successful, false if stack is empty
     *
     * Time complexity: O(1)
     * Memory ordering: acquire (ensures all writes from push visible)
     */
    bool pop(size_t &index);

    /**
     * @brief Check if stack is empty (may be stale immediately)
     *
     * @return true if stack appears empty
     *
     * Note: This is a snapshot - stack may change immediately after return
     */
    bool empty() const;

    /**
     * @brief Get approximate size (may be stale immediately)
     *
     * @return Approximate number of elements
     *
     * Note: This is a snapshot - stack may change immediately after return
     */
    size_t size() const;

private:
    /**
     * @brief Internal node structure for stack
     *
     * Each node stores an index and a pointer to the next node.
     * Nodes are allocated from a pre-allocated pool to avoid malloc overhead.
     */
    struct Node {
        size_t index;    ///< Index value stored in this node
        Node *next;      ///< Pointer to next node in stack

        Node(size_t idx) : index(idx), next(nullptr) {}
    };

    /**
     * @brief Tagged pointer combining pointer and version counter
     *
     * Layout (64-bit):
     * - Bits 0-47: Node pointer (48 bits = 256TB address space)
     * - Bits 48-63: Version counter (16 bits = 65536 versions)
     *
     * The version counter prevents ABA problem:
     * - Thread A reads head (ptr=X, ver=1)
     * - Thread B pops X, pushes Y, pops Y, pushes X (ptr=X, ver=2)
     * - Thread A's CAS fails because version changed
     */
    union TaggedPointer {
        struct {
            uint64_t ptr : 48;     ///< Pointer to node (lower 48 bits)
            uint64_t version : 16; ///< Version counter (upper 16 bits)
        } parts;
        uint64_t combined; ///< Combined 64-bit value for atomic operations

        TaggedPointer() : combined(0) {}
        TaggedPointer(Node *p, uint16_t v) : combined(0) {
            parts.ptr = reinterpret_cast<uint64_t>(p);
            parts.version = v;
        }

        Node *getPtr() const {
            return reinterpret_cast<Node *>(parts.ptr);
        }

        uint16_t getVersion() const {
            return parts.version;
        }
    };

    // Ensure TaggedPointer is lock-free on this platform
    static_assert(sizeof(TaggedPointer) == sizeof(uint64_t),
                  "TaggedPointer must be 64 bits");
    static_assert(std::atomic<uint64_t>::is_always_lock_free,
                  "std::atomic<uint64_t> must be lock-free");

    /**
     * @brief Atomic head pointer with version tag
     *
     * Uses std::memory_order_acquire for loads and std::memory_order_release for stores
     * to ensure proper synchronization without full barriers.
     */
    std::atomic<uint64_t> head_;

    /**
     * @brief Approximate size counter (relaxed ordering)
     *
     * This is used for statistics only - not for correctness.
     * Uses relaxed memory ordering for minimal overhead.
     */
    std::atomic<size_t> size_;

    /**
     * @brief Node pool for memory reuse
     *
     * To avoid malloc/free overhead on every push/pop, we maintain a pool
     * of pre-allocated nodes. When popping, nodes are returned to this pool
     * instead of being freed. When pushing, nodes are taken from this pool
     * if available, otherwise allocated.
     *
     * This pool itself is lock-free using the same tagged pointer technique.
     */
    std::atomic<uint64_t> free_list_;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LOCK_FREE_STACK_H

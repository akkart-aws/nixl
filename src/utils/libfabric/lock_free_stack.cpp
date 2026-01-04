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

#include "lock_free_stack.h"
#include <new>

LockFreeStack::LockFreeStack() : head_(0), size_(0), free_list_(0) {
    // Initialize with empty stack
    // head_ = 0 means nullptr with version 0
}

LockFreeStack::~LockFreeStack() {
    // Clean up all nodes in the main stack
    TaggedPointer current;
    current.combined = head_.load(std::memory_order_relaxed);
    
    while (current.getPtr() != nullptr) {
        Node *node = current.getPtr();
        Node *next = node->next;
        delete node;
        
        current.parts.ptr = reinterpret_cast<uint64_t>(next);
    }
    
    // Clean up all nodes in the free list
    current.combined = free_list_.load(std::memory_order_relaxed);
    
    while (current.getPtr() != nullptr) {
        Node *node = current.getPtr();
        Node *next = node->next;
        delete node;
        
        current.parts.ptr = reinterpret_cast<uint64_t>(next);
    }
}

bool LockFreeStack::push(size_t index) {
    // Try to get a node from the free list first
    Node *new_node = nullptr;
    
    while (true) {
        TaggedPointer old_free;
        old_free.combined = free_list_.load(std::memory_order_acquire);
        
        if (old_free.getPtr() == nullptr) {
            // Free list is empty, allocate new node
            break;
        }
        
        Node *free_node = old_free.getPtr();
        TaggedPointer new_free(free_node->next, old_free.getVersion() + 1);
        
        // Try to pop from free list
        if (free_list_.compare_exchange_weak(
                old_free.combined,
                new_free.combined,
                std::memory_order_release,
                std::memory_order_acquire)) {
            // Successfully got node from free list
            new_node = free_node;
            new_node->index = index;
            new_node->next = nullptr;
            break;
        }
        // CAS failed, retry
    }
    
    // If free list was empty, allocate new node
    if (new_node == nullptr) {
        try {
            new_node = new Node(index);
        } catch (const std::bad_alloc &) {
            return false; // Out of memory
        }
    }
    
    // Push node onto main stack using CAS loop
    while (true) {
        TaggedPointer old_head;
        old_head.combined = head_.load(std::memory_order_acquire);
        
        // Link new node to current head
        new_node->next = old_head.getPtr();
        
        // Create new head with incremented version
        TaggedPointer new_head(new_node, old_head.getVersion() + 1);
        
        // Try to update head atomically
        if (head_.compare_exchange_weak(
                old_head.combined,
                new_head.combined,
                std::memory_order_release,
                std::memory_order_acquire)) {
            // Success! Increment size counter
            size_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        
        // CAS failed (another thread modified head), retry
        // The weak version may spuriously fail, so we loop
    }
}

bool LockFreeStack::pop(size_t &index) {
    while (true) {
        TaggedPointer old_head;
        old_head.combined = head_.load(std::memory_order_acquire);
        
        if (old_head.getPtr() == nullptr) {
            // Stack is empty
            return false;
        }
        
        Node *head_node = old_head.getPtr();
        
        // Create new head pointing to next node with incremented version
        TaggedPointer new_head(head_node->next, old_head.getVersion() + 1);
        
        // Try to update head atomically
        if (head_.compare_exchange_weak(
                old_head.combined,
                new_head.combined,
                std::memory_order_release,
                std::memory_order_acquire)) {
            // Success! Extract the index
            index = head_node->index;
            
            // Decrement size counter
            size_.fetch_sub(1, std::memory_order_relaxed);
            
            // Return node to free list for reuse
            while (true) {
                TaggedPointer old_free;
                old_free.combined = free_list_.load(std::memory_order_acquire);
                
                head_node->next = old_free.getPtr();
                TaggedPointer new_free(head_node, old_free.getVersion() + 1);
                
                if (free_list_.compare_exchange_weak(
                        old_free.combined,
                        new_free.combined,
                        std::memory_order_release,
                        std::memory_order_acquire)) {
                    break;
                }
                // CAS failed, retry adding to free list
            }
            
            return true;
        }
        
        // CAS failed (another thread modified head), retry
    }
}

bool LockFreeStack::empty() const {
    TaggedPointer current;
    current.combined = head_.load(std::memory_order_acquire);
    return current.getPtr() == nullptr;
}

size_t LockFreeStack::size() const {
    // Return approximate size (may be stale)
    return size_.load(std::memory_order_relaxed);
}

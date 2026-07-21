/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "kernel_operator.h"

namespace {

class DSAPrepareSparseIndicesKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* requestBlockTable,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* selectedCounts,
        __gm__ int64_t* targetSlots,
        __gm__ int32_t* hashWorkspace,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t requestCount,
        uint32_t blockTableWidth,
        uint32_t scratchCapacity,
        uint32_t hashCapacity,
        uint32_t blockSize,
        uint32_t needPacked,
        uint32_t clearInvalidRows)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        requestCount_ = requestCount;
        blockTableWidth_ = blockTableWidth;
        scratchCapacity_ = scratchCapacity;
        hashCapacity_ = hashCapacity;
        blockSize_ = blockSize;
        needPacked_ = needPacked != 0;
        clearInvalidRows_ = clearInvalidRows != 0;
        topkIndices_.SetGlobalBuffer(topkIndices);
        splitBoundary_.SetGlobalBuffer(splitBoundary, rowCount);
        rowReqIndices_.SetGlobalBuffer(rowReqIndices, rowCount);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable,
            static_cast<uint64_t>(requestCount) * blockTableWidth);
        selectedPacked_.SetGlobalBuffer(
            selectedPacked,
            static_cast<uint64_t>(requestCount) * scratchCapacity);
        selectedCounts_.SetGlobalBuffer(selectedCounts, requestCount);
        targetSlots_.SetGlobalBuffer(
            targetSlots,
            static_cast<uint64_t>(requestCount) * scratchCapacity);
        hashWorkspace_.SetGlobalBuffer(
            hashWorkspace,
            static_cast<uint64_t>(requestCount) * hashCapacity);
    }

    __aicore__ inline void Process()
    {
        const uint32_t req = AscendC::GetBlockIdx();
        if (req >= requestCount_) {
            return;
        }
        const uint64_t hashOffset = static_cast<uint64_t>(req) * hashCapacity_;
        for (uint32_t i = 0; i < hashCapacity_; ++i) {
            hashWorkspace_.SetValue(hashOffset + i, -1);
        }
        const uint64_t packedOffset =
            static_cast<uint64_t>(req) * scratchCapacity_;
        uint32_t uniqueCount = 0;

        // One AIV owns one request. Serial row-major insertion gives stable
        // first-occurrence union order across all MTP rows of that request.
        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) != static_cast<int32_t>(req)) {
                continue;
            }
            const int32_t boundary = splitBoundary_.GetValue(row);
            const uint64_t rowOffset = static_cast<uint64_t>(row) * rowWidth_;
            for (uint32_t col = 0; col < rowWidth_; ++col) {
                const uint64_t indexOffset = rowOffset + col;
                const int32_t token = topkIndices_.GetValue(indexOffset);
                if (token < 0 || token >= boundary) {
                    continue;
                }
                uint32_t bucket =
                    static_cast<uint32_t>(token) & (hashCapacity_ - 1);
                int32_t scratchSlot = -1;
                for (uint32_t probe = 0; probe < hashCapacity_; ++probe) {
                    const uint64_t hashIndex =
                        hashOffset + ((bucket + probe) & (hashCapacity_ - 1));
                    const int32_t existing = hashWorkspace_.GetValue(hashIndex);
                    if (existing < 0) {
                        scratchSlot = static_cast<int32_t>(uniqueCount++);
                        hashWorkspace_.SetValue(hashIndex, scratchSlot);
                        selectedPacked_.SetValue(
                            packedOffset + scratchSlot, token);
                        const uint32_t logicalBlock =
                            static_cast<uint32_t>(scratchSlot) / blockSize_;
                        const uint32_t blockOffset =
                            static_cast<uint32_t>(scratchSlot) % blockSize_;
                        const int32_t physicalBlock =
                            requestBlockTable_.GetValue(
                                static_cast<uint64_t>(req) * blockTableWidth_
                                + logicalBlock);
                        targetSlots_.SetValue(
                            packedOffset + scratchSlot,
                            static_cast<int64_t>(physicalBlock) * blockSize_
                                + blockOffset);
                        break;
                    }
                    if (selectedPacked_.GetValue(
                            packedOffset + existing) == token) {
                        scratchSlot = existing;
                        break;
                    }
                }
                topkIndices_.SetValue(indexOffset, scratchSlot);
            }
        }
        selectedCounts_.SetValue(
            req, needPacked_ ? static_cast<int32_t>(uniqueCount) : 0);

        if (req == 0 && clearInvalidRows_) {
            for (uint32_t row = 0; row < rowCount_; ++row) {
                if (rowReqIndices_.GetValue(row) >= 0) {
                    continue;
                }
                const uint64_t rowOffset = static_cast<uint64_t>(row) * rowWidth_;
                for (uint32_t col = 0; col < rowWidth_; ++col) {
                    topkIndices_.SetValue(rowOffset + col, 0);
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> selectedCounts_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::GlobalTensor<int32_t> hashWorkspace_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestCount_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t hashCapacity_ = 0;
    uint32_t blockSize_ = 0;
    bool needPacked_ = false;
    bool clearInvalidRows_ = false;
};

}  // namespace

extern "C" __global__ __aicore__ void dsa_prepare_sparse_indices_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* requestBlockTable,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* selectedCounts,
    __gm__ int64_t* targetSlots,
    __gm__ int32_t* hashWorkspace,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t requestCount,
    uint32_t blockTableWidth,
    uint32_t scratchCapacity,
    uint32_t hashCapacity,
    uint32_t blockSize,
    uint32_t needPacked,
    uint32_t clearInvalidRows)
{
    DSAPrepareSparseIndicesKernel op;
    op.Init(topkIndices, splitBoundary, rowReqIndices, requestBlockTable,
            selectedPacked, selectedCounts, targetSlots, hashWorkspace,
            rowCount, rowWidth, requestCount, blockTableWidth, scratchCapacity,
            hashCapacity, blockSize, needPacked, clearInvalidRows);
    op.Process();
}

namespace vllm_ascend {

void dsa_prepare_sparse_indices_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* rowReqIndices, void* requestBlockTable, void* selectedPacked,
    void* selectedCounts, void* targetSlots, void* hashWorkspace,
    uint32_t rowCount, uint32_t rowWidth, uint32_t requestCount,
    uint32_t blockTableWidth, uint32_t scratchCapacity,
    uint32_t hashCapacity, uint32_t blockSize, bool needPacked,
    bool clearInvalidRows)
{
    dsa_prepare_sparse_indices_kernel<<<requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(selectedCounts),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(hashWorkspace), rowCount, rowWidth,
        requestCount, blockTableWidth, scratchCapacity, hashCapacity,
        blockSize, needPacked, clearInvalidRows);
}

}  // namespace vllm_ascend

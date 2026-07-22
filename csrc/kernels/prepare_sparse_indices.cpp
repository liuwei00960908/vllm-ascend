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
        __gm__ int32_t* bitmapWorkspace,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t requestCount,
        uint32_t blockTableWidth,
        uint32_t scratchCapacity,
        uint32_t workspaceWidth,
        uint32_t blockSize,
        uint32_t needPacked,
        uint32_t clearInvalidRows)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        requestCount_ = requestCount;
        blockTableWidth_ = blockTableWidth;
        scratchCapacity_ = scratchCapacity;
        workspaceWidth_ = workspaceWidth;
        bitmapWords_ = workspaceWidth / 2;
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
        bitmapWorkspace_.SetGlobalBuffer(
            bitmapWorkspace,
            static_cast<uint64_t>(requestCount) * workspaceWidth);
    }

    __aicore__ inline void Process()
    {
        const uint32_t req = AscendC::GetBlockIdx();
        if (req >= requestCount_) {
            return;
        }
        bool hasPositiveBoundary = false;
        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) == static_cast<int32_t>(req)
                && splitBoundary_.GetValue(row) > 0) {
                hasPositiveBoundary = true;
                break;
            }
        }
        if (!hasPositiveBoundary) {
            selectedCounts_.SetValue(req, 0);
            if (req == 0 && clearInvalidRows_) {
                ClearInvalidRows();
            }
            return;
        }
        const uint64_t workspaceOffset =
            static_cast<uint64_t>(req) * workspaceWidth_;
        const uint64_t prefixOffset = workspaceOffset + bitmapWords_;
        for (uint32_t i = 0; i < bitmapWords_; ++i) {
            bitmapWorkspace_.SetValue(workspaceOffset + i, 0);
        }
        const uint64_t packedOffset =
            static_cast<uint64_t>(req) * scratchCapacity_;

        // One AIV owns one request, so setting bitmap words does not need
        // atomics even when several MTP rows select the same position.
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
                const uint32_t word = static_cast<uint32_t>(token) >> 5;
                const uint32_t bit = static_cast<uint32_t>(token) & 31;
                const uint64_t bitmapIndex = workspaceOffset + word;
                const uint32_t value = static_cast<uint32_t>(
                    bitmapWorkspace_.GetValue(bitmapIndex));
                bitmapWorkspace_.SetValue(
                    bitmapIndex,
                    static_cast<int32_t>(value | (1U << bit)));
            }
        }

        // Prefix popcount gives every selected position a deterministic rank
        // in ascending token-position order.
        uint32_t uniqueCount = 0;
        for (uint32_t word = 0; word < bitmapWords_; ++word) {
            bitmapWorkspace_.SetValue(
                prefixOffset + word, static_cast<int32_t>(uniqueCount));
            uniqueCount += Popcount32(static_cast<uint32_t>(
                bitmapWorkspace_.GetValue(workspaceOffset + word)));
        }

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
                const uint32_t word = static_cast<uint32_t>(token) >> 5;
                const uint32_t bit = static_cast<uint32_t>(token) & 31;
                const uint32_t bitmapValue = static_cast<uint32_t>(
                    bitmapWorkspace_.GetValue(workspaceOffset + word));
                const uint32_t lowerMask = bit == 0 ? 0 : ((1U << bit) - 1);
                const uint32_t scratchSlot = static_cast<uint32_t>(
                    bitmapWorkspace_.GetValue(prefixOffset + word))
                    + Popcount32(bitmapValue & lowerMask);
                topkIndices_.SetValue(
                    indexOffset, static_cast<int32_t>(scratchSlot));
                if (needPacked_) {
                    selectedPacked_.SetValue(
                        packedOffset + scratchSlot, token);
                    const uint32_t logicalBlock = scratchSlot / blockSize_;
                    const uint32_t blockOffset = scratchSlot % blockSize_;
                    const int32_t physicalBlock =
                        requestBlockTable_.GetValue(
                            static_cast<uint64_t>(req) * blockTableWidth_
                            + logicalBlock);
                    targetSlots_.SetValue(
                        packedOffset + scratchSlot,
                        static_cast<int64_t>(physicalBlock) * blockSize_
                            + blockOffset);
                }
            }
        }
        selectedCounts_.SetValue(
            req, needPacked_ ? static_cast<int32_t>(uniqueCount) : 0);

        if (req == 0 && clearInvalidRows_) {
            ClearInvalidRows();
        }
    }

private:
    __aicore__ inline void ClearInvalidRows()
    {
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

    __aicore__ inline uint32_t Popcount32(uint32_t value) const
    {
        value = value - ((value >> 1) & 0x55555555U);
        value = (value & 0x33333333U) + ((value >> 2) & 0x33333333U);
        value = (value + (value >> 4)) & 0x0F0F0F0FU;
        value = value + (value >> 8);
        value = value + (value >> 16);
        return value & 0x3FU;
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> selectedCounts_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::GlobalTensor<int32_t> bitmapWorkspace_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestCount_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t workspaceWidth_ = 0;
    uint32_t bitmapWords_ = 0;
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
    __gm__ int32_t* bitmapWorkspace,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t requestCount,
    uint32_t blockTableWidth,
    uint32_t scratchCapacity,
    uint32_t workspaceWidth,
    uint32_t blockSize,
    uint32_t needPacked,
    uint32_t clearInvalidRows)
{
    DSAPrepareSparseIndicesKernel op;
    op.Init(topkIndices, splitBoundary, rowReqIndices, requestBlockTable,
            selectedPacked, selectedCounts, targetSlots, bitmapWorkspace,
            rowCount, rowWidth, requestCount, blockTableWidth, scratchCapacity,
            workspaceWidth, blockSize, needPacked, clearInvalidRows);
    op.Process();
}

namespace vllm_ascend {

void dsa_prepare_sparse_indices_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* rowReqIndices, void* requestBlockTable, void* selectedPacked,
    void* selectedCounts, void* targetSlots, void* bitmapWorkspace,
    uint32_t rowCount, uint32_t rowWidth, uint32_t requestCount,
    uint32_t blockTableWidth, uint32_t scratchCapacity,
    uint32_t workspaceWidth, uint32_t blockSize, bool needPacked,
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
        static_cast<int32_t*>(bitmapWorkspace), rowCount, rowWidth,
        requestCount, blockTableWidth, scratchCapacity, workspaceWidth,
        blockSize, needPacked, clearInvalidRows);
}

}  // namespace vllm_ascend

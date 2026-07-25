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

constexpr uint32_t kSortGroup = 32;
constexpr uint32_t kPairWidth = 2;
constexpr uint32_t kMergeWays = 4;

template <AscendC::HardEvent event>
__aicore__ inline void Sync()
{
    const int32_t id =
        static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(id);
    AscendC::WaitFlag<event>(id);
}

class DSAStagedBitmapUnionKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* rowPacked,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* localToUnion,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int64_t* targetSlots,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t maxTokens,
        uint32_t blockTableWidth,
        uint32_t blockSize)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        total_ = rowCount * rowWidth;
        maxTokens_ = maxTokens;
        bitmapWords_ = (maxTokens + 31) / 32;
        bufferWords_ = ((bitmapWords_ + 7) / 8) * 8;
        blockTableWidth_ = blockTableWidth;
        blockSize_ = blockSize;
        rowPacked_.SetGlobalBuffer(rowPacked, total_);
        selectedPacked_.SetGlobalBuffer(selectedPacked, total_);
        localToUnion_.SetGlobalBuffer(localToUnion, total_);
        selectedCount_.SetGlobalBuffer(selectedCount, 1);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable, blockTableWidth);
        targetSlots_.SetGlobalBuffer(targetSlots, total_);
        pipe_.InitBuffer(bitmapBuf_, bufferWords_ * sizeof(int32_t));
        pipe_.InitBuffer(prefixBuf_, bufferWords_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        auto bitmap = bitmapBuf_.Get<int32_t>();
        auto prefix = prefixBuf_.Get<int32_t>();
        AscendC::Duplicate(bitmap, static_cast<int32_t>(0), bufferWords_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_S>();

        for (uint32_t i = 0; i < total_; ++i) {
            const int32_t token = rowPacked_.GetValue(i);
            if (token < 0 || token >= static_cast<int32_t>(maxTokens_)) {
                continue;
            }
            const uint32_t word = static_cast<uint32_t>(token) >> 5;
            const uint32_t bit = static_cast<uint32_t>(token) & 31;
            const uint32_t value =
                static_cast<uint32_t>(bitmap.GetValue(word));
            bitmap.SetValue(
                word, static_cast<int32_t>(value | (1U << bit)));
        }

        uint32_t count = 0;
        for (uint32_t word = 0; word < bitmapWords_; ++word) {
            prefix.SetValue(word, static_cast<int32_t>(count));
            count += Popcount(
                static_cast<uint32_t>(bitmap.GetValue(word)));
        }

        for (uint32_t i = 0; i < total_; ++i) {
            const uint32_t token =
                static_cast<uint32_t>(rowPacked_.GetValue(i));
            const uint32_t word = token >> 5;
            const uint32_t bit = token & 31;
            const uint32_t rank = Rank(bitmap, prefix, word, bit);
            localToUnion_.SetValue(i, static_cast<int32_t>(rank));
            // Duplicate tokens overwrite the same values at the same rank.
            WriteUnion(rank, static_cast<int32_t>(token));
        }
        selectedCount_.SetValue(0, static_cast<int32_t>(count));
    }

private:
    __aicore__ inline uint32_t Popcount(uint32_t value) const
    {
        value -= (value >> 1) & 0x55555555U;
        value = (value & 0x33333333U)
            + ((value >> 2) & 0x33333333U);
        value = (value + (value >> 4)) & 0x0F0F0F0FU;
        value += value >> 8;
        value += value >> 16;
        return value & 0x3FU;
    }

    __aicore__ inline uint32_t Rank(
        const AscendC::LocalTensor<int32_t>& bitmap,
        const AscendC::LocalTensor<int32_t>& prefix,
        uint32_t word,
        uint32_t bit) const
    {
        const uint32_t lower =
            bit == 0 ? 0 : ((1U << bit) - 1);
        return static_cast<uint32_t>(prefix.GetValue(word))
            + Popcount(
                static_cast<uint32_t>(bitmap.GetValue(word)) & lower);
    }

    __aicore__ inline void WriteUnion(uint32_t rank, int32_t token)
    {
        selectedPacked_.SetValue(rank, token);
        const uint32_t logicalBlock = rank / blockSize_;
        const uint32_t blockOffset = rank % blockSize_;
        const int32_t physicalBlock =
            requestBlockTable_.GetValue(logicalBlock);
        targetSlots_.SetValue(
            rank,
            static_cast<int64_t>(physicalBlock) * blockSize_
                + blockOffset);
    }

    AscendC::GlobalTensor<int32_t> rowPacked_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bitmapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prefixBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t total_ = 0;
    uint32_t maxTokens_ = 0;
    uint32_t bitmapWords_ = 0;
    uint32_t bufferWords_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t blockSize_ = 0;
};

class DSAStagedSortUnionKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* rowPacked,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* localToUnion,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int64_t* targetSlots,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t blockTableWidth,
        uint32_t blockSize)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        total_ = rowCount * rowWidth;
        blockTableWidth_ = blockTableWidth;
        blockSize_ = blockSize;
        rowPacked_.SetGlobalBuffer(rowPacked, total_);
        selectedPacked_.SetGlobalBuffer(selectedPacked, total_);
        localToUnion_.SetGlobalBuffer(localToUnion, total_);
        selectedCount_.SetGlobalBuffer(selectedCount, 1);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable, blockTableWidth);
        targetSlots_.SetGlobalBuffer(targetSlots, total_);
        pipe_.InitBuffer(
            sortSrcBuf_, total_ * kPairWidth * sizeof(float));
        pipe_.InitBuffer(
            sortTmpBuf_, total_ * kPairWidth * sizeof(float));
        pipe_.InitBuffer(sortInputBuf_, total_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        auto src = sortSrcBuf_.Get<float>();
        auto tmp = sortTmpBuf_.Get<float>();
        auto input = sortInputBuf_.Get<int32_t>();
        auto srcInt = src.ReinterpretCast<int32_t>();
        AscendC::DataCopy(input, rowPacked_, total_);
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(
            src, input, AscendC::RoundMode::CAST_NONE, total_);
        AscendC::Muls(src, src, -1.0F, total_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_S>();
        for (uint32_t i = 0; i < total_; ++i) {
            srcInt.SetValue(total_ + i, static_cast<int32_t>(i));
        }
        Sync<AscendC::HardEvent::S_V>();
        SortAll(src, tmp);
        Sync<AscendC::HardEvent::V_S>();

        auto sortedInt = src.ReinterpretCast<int32_t>();
        int32_t previous = -1;
        uint32_t rank = 0;
        for (uint32_t i = 0; i < total_; ++i) {
            const int32_t token =
                -static_cast<int32_t>(src.GetValue(kPairWidth * i));
            const uint32_t original = static_cast<uint32_t>(
                sortedInt.GetValue(kPairWidth * i + 1));
            if (i == 0 || token != previous) {
                WriteUnion(rank, token);
                previous = token;
                ++rank;
            }
            localToUnion_.SetValue(
                original, static_cast<int32_t>(rank - 1));
        }
        selectedCount_.SetValue(0, static_cast<int32_t>(rank));
    }

private:
    __aicore__ inline void SortAll(
        AscendC::LocalTensor<float>& src,
        AscendC::LocalTensor<float>& tmp)
    {
        const uint32_t repeats = total_ / kSortGroup;
        AscendC::Sort32(
            tmp,
            src,
            src[total_].ReinterpretCast<uint32_t>(),
            repeats);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t groups = repeats;
        uint32_t elements = kSortGroup;
        uint32_t pass = 0;
        while (groups > 1) {
            auto input = pass % 2 == 0 ? tmp : src;
            auto output = pass % 2 == 0 ? src : tmp;
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = elements;
            params.elementLengths[1] = elements;
            params.elementLengths[2] = elements;
            params.elementLengths[3] = elements;
            params.ifExhaustedSuspension = false;
            params.validBit = 0b1111;
            if (groups <= kMergeWays) {
                params.repeatTimes = 1;
                params.validBit =
                    groups == 2 ? 0b0011
                    : groups == 3 ? 0b0111
                                  : 0b1111;
            } else {
                params.repeatTimes = groups / kMergeWays;
            }
            AscendC::MrgSortSrcList<float> list;
            list.src1 = input;
            list.src2 = input[kPairWidth * elements];
            list.src3 = input[2 * kPairWidth * elements];
            list.src4 = input[3 * kPairWidth * elements];
            AscendC::MrgSort<float>(output, list, params);
            AscendC::PipeBarrier<PIPE_V>();
            groups = groups <= kMergeWays ? 1 : groups / kMergeWays;
            elements *= kMergeWays;
            ++pass;
        }
        if (pass % 2 == 0) {
            AscendC::DataCopy(
                src, tmp, total_ * kPairWidth);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void WriteUnion(uint32_t rank, int32_t token)
    {
        selectedPacked_.SetValue(rank, token);
        const uint32_t logicalBlock = rank / blockSize_;
        const uint32_t blockOffset = rank % blockSize_;
        const int32_t physicalBlock =
            requestBlockTable_.GetValue(logicalBlock);
        targetSlots_.SetValue(
            rank,
            static_cast<int64_t>(physicalBlock) * blockSize_
                + blockOffset);
    }

    AscendC::GlobalTensor<int32_t> rowPacked_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortSrcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortInputBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t total_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t blockSize_ = 0;
};

class DSAStagedRemapRowsKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* localIndices,
        __gm__ int32_t* localToUnion,
        uint32_t rowCount,
        uint32_t rowWidth)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        localIndices_.SetGlobalBuffer(
            localIndices, rowCount * rowWidth);
        localToUnion_.SetGlobalBuffer(
            localToUnion, rowCount * rowWidth);
        pipe_.InitBuffer(inputBuf_, rowWidth * sizeof(int32_t));
        pipe_.InitBuffer(mapBuf_, rowWidth * sizeof(int32_t));
        pipe_.InitBuffer(outputBuf_, rowWidth * sizeof(int32_t));
        pipe_.InitBuffer(offsetBuf_, rowWidth * sizeof(uint32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t row = AscendC::GetBlockIdx();
        if (row >= rowCount_) {
            return;
        }
        const uint32_t offset = row * rowWidth_;
        auto input = inputBuf_.Get<int32_t>();
        auto mapping = mapBuf_.Get<int32_t>();
        auto output = outputBuf_.Get<int32_t>();
        auto offsets = offsetBuf_.Get<uint32_t>();
        AscendC::DataCopy(input, localIndices_[offset], rowWidth_);
        AscendC::DataCopy(mapping, localToUnion_[offset], rowWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Muls(
            offsets.ReinterpretCast<int32_t>(),
            input,
            static_cast<int32_t>(sizeof(int32_t)),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(
            output, mapping, offsets, static_cast<uint32_t>(0), rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(localIndices_[offset], output, rowWidth_);
    }

private:
    AscendC::GlobalTensor<int32_t> localIndices_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void dsa_staged_bitmap_union_kernel(
    __gm__ int32_t* rowPacked,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* localToUnion,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t maxTokens,
    uint32_t blockTableWidth,
    uint32_t blockSize)
{
    if ASCEND_IS_AIV {
        DSAStagedBitmapUnionKernel op;
        op.Init(rowPacked, selectedPacked, localToUnion, selectedCount,
                requestBlockTable, targetSlots, rowCount, rowWidth,
                maxTokens, blockTableWidth, blockSize);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_sort_union_kernel(
    __gm__ int32_t* rowPacked,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* localToUnion,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t blockTableWidth,
    uint32_t blockSize)
{
    if ASCEND_IS_AIV {
        DSAStagedSortUnionKernel op;
        op.Init(rowPacked, selectedPacked, localToUnion, selectedCount,
                requestBlockTable, targetSlots, rowCount, rowWidth,
                blockTableWidth, blockSize);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_remap_rows_kernel(
    __gm__ int32_t* localIndices,
    __gm__ int32_t* localToUnion,
    uint32_t rowCount,
    uint32_t rowWidth)
{
    if ASCEND_IS_AIV {
        DSAStagedRemapRowsKernel op;
        op.Init(localIndices, localToUnion, rowCount, rowWidth);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_copy_rows_kernel(
    __gm__ int32_t* output,
    __gm__ int32_t* localIndices,
    uint32_t rowCount,
    uint32_t rowWidth)
{
    if ASCEND_IS_AIV {
        const uint32_t row = AscendC::GetBlockIdx();
        if (row >= rowCount) {
            return;
        }
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> rowBuf;
        pipe.InitBuffer(rowBuf, rowWidth * sizeof(int32_t));
        auto rowLocal = rowBuf.Get<int32_t>();
        AscendC::GlobalTensor<int32_t> outputGm;
        AscendC::GlobalTensor<int32_t> localIndicesGm;
        const uint64_t total =
            static_cast<uint64_t>(rowCount) * rowWidth;
        outputGm.SetGlobalBuffer(output, total);
        localIndicesGm.SetGlobalBuffer(localIndices, total);
        const uint64_t offset = static_cast<uint64_t>(row) * rowWidth;
        AscendC::DataCopy(rowLocal, localIndicesGm[offset], rowWidth);
        Sync<AscendC::HardEvent::MTE2_MTE3>();
        AscendC::DataCopy(outputGm[offset], rowLocal, rowWidth);
    }
}

namespace vllm_ascend {

void dsa_staged_bitmap_union_impl(
    void* stream, void* rowPacked, void* selectedPacked,
    void* localToUnion, void* selectedCount, void* requestBlockTable,
    void* targetSlots, uint32_t rowCount, uint32_t rowWidth,
    uint32_t maxTokens, uint32_t blockTableWidth, uint32_t blockSize)
{
    dsa_staged_bitmap_union_kernel<<<1, nullptr, stream>>>(
        static_cast<int32_t*>(rowPacked),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        rowCount, rowWidth, maxTokens, blockTableWidth, blockSize);
}

void dsa_staged_sort_union_impl(
    void* stream, void* rowPacked, void* selectedPacked,
    void* localToUnion, void* selectedCount, void* requestBlockTable,
    void* targetSlots, uint32_t rowCount, uint32_t rowWidth,
    uint32_t blockTableWidth, uint32_t blockSize)
{
    dsa_staged_sort_union_kernel<<<1, nullptr, stream>>>(
        static_cast<int32_t*>(rowPacked),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        rowCount, rowWidth, blockTableWidth, blockSize);
}

void dsa_staged_remap_rows_impl(
    void* stream, void* localIndices, void* localToUnion,
    uint32_t rowCount, uint32_t rowWidth)
{
    dsa_staged_remap_rows_kernel<<<rowCount, nullptr, stream>>>(
        static_cast<int32_t*>(localIndices),
        static_cast<int32_t*>(localToUnion), rowCount, rowWidth);
}

void dsa_staged_copy_rows_impl(
    void* stream, void* output, void* localIndices,
    uint32_t rowCount, uint32_t rowWidth)
{
    dsa_staged_copy_rows_kernel<<<rowCount, nullptr, stream>>>(
        static_cast<int32_t*>(output),
        static_cast<int32_t*>(localIndices), rowCount, rowWidth);
}

}  // namespace vllm_ascend

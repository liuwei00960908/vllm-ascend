/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Provenance: vllm-ascend-sparse@c7c4a4ac
 * csrc/kernels/prepare_sparse_indices_staged.cpp (production slice only):
 * the experimental hash-union / sharded-sort / sharded-vector /
 * unique-finalize / copy-rows kernels and their impl wrappers stay
 * unported - the production dispatch only launches compact_rows ->
 * single_row_finalize (MTP=1) or production_sort_union +
 * boundary_remap (MTP=2).
 */

#include "kernel_operator.h"

namespace {

constexpr uint32_t kSortGroup = 32;
constexpr uint32_t kPairWidth = 2;
constexpr uint32_t kMergeWays = 4;
constexpr uint32_t kCumSumTileWidth = 512;
constexpr uint32_t kCumSumTransposeRows = 16;
constexpr uint32_t kCumSumWorkspaceBytes =
    2 * kCumSumTransposeRows * kCumSumTileWidth * sizeof(float);
constexpr uint32_t kDataBlockBytes = 32;
constexpr uint32_t kInt32PerDataBlock =
    kDataBlockBytes / sizeof(int32_t);
constexpr AscendC::CumSumConfig kCumSumConfig{true, false, false};

template <AscendC::HardEvent event>
__aicore__ inline void Sync()
{
    const int32_t id =
        static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(id);
    AscendC::WaitFlag<event>(id);
}

template <typename T>
__aicore__ inline void CopyLocalToGlobalExact(
    AscendC::GlobalTensor<T> dst,
    AscendC::LocalTensor<T> src,
    uint32_t count)
{
    if (count == 0) {
        return;
    }
    const uint32_t bytes = count * sizeof(T);
    if ((bytes & (kDataBlockBytes - 1)) == 0) {
        AscendC::DataCopy(dst, src, count);
        return;
    }
    // The count-based DataCopy rounds an unaligned byte length down.
    const AscendC::DataCopyParams params{
        1,
        static_cast<uint16_t>(bytes),
        0,
        0};
    AscendC::DataCopyPad(dst, src, params);
}

__aicore__ inline void CopyLocalToGlobalExact(
    AscendC::GlobalTensor<int64_t> dst,
    AscendC::LocalTensor<int64_t> src,
    uint32_t count)
{
    if (count == 0) {
        return;
    }
    AscendC::GlobalTensor<int32_t> dstWords;
    dstWords.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(
            const_cast<__gm__ int64_t*>(dst.GetPhyAddr())),
        2 * count);
    CopyLocalToGlobalExact(
        dstWords, src.ReinterpretCast<int32_t>(), 2 * count);
}

template <typename T>
__aicore__ inline void CopyGlobalToLocalExact(
    AscendC::LocalTensor<T> dst,
    AscendC::GlobalTensor<T> src,
    uint32_t count)
{
    if (count == 0) {
        return;
    }
    const uint32_t bytes = count * sizeof(T);
    if ((bytes & (kDataBlockBytes - 1)) == 0) {
        AscendC::DataCopy(dst, src, count);
        return;
    }
    // DataCopyPad keeps the dynamic tail and pads only the local destination.
    const AscendC::DataCopyParams params{
        1,
        static_cast<uint16_t>(bytes),
        0,
        0};
    AscendC::DataCopyPad(dst, src, params, {});
}

// Production pre-union stage for fixed-width pure-decode batches. Each AIV
// owns one complete top-k row, so all GM writes are naturally cacheline
// disjoint. Selected tokens are compacted into the caller-owned output buffer;
// the same row is remapped in place to row-local ranks. Unselected positions
class DSAStagedCompactRowsKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* rowPacked,
        __gm__ int32_t* rowCounts,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t requestCount,
        uint32_t rowsPerRequest,
        uint32_t scratchCapacity,
        uint32_t coreCount,
        bool clearInvalidRows)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        requestCount_ = requestCount;
        rowsPerRequest_ = rowsPerRequest;
        scratchCapacity_ = scratchCapacity;
        coreCount_ = coreCount;
        clearInvalidRows_ = clearInvalidRows;
        topkIndices_.SetGlobalBuffer(
            topkIndices,
            static_cast<uint64_t>(rowCount_) * rowWidth_);
        splitBoundary_.SetGlobalBuffer(splitBoundary, rowCount_);
        rowReqIndices_.SetGlobalBuffer(rowReqIndices, rowCount_);
        rowPacked_.SetGlobalBuffer(
            rowPacked,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        rowCounts_.SetGlobalBuffer(
            rowCounts,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);

        const uint32_t rowBytes = rowWidth_ * sizeof(int32_t);
        const uint32_t maskBytes = rowWidth_ / 8;
        pipe_.InitBuffer(inputBuf_, rowBytes);
        pipe_.InitBuffer(clampedBuf_, rowBytes);
        pipe_.InitBuffer(packedBuf_, rowBytes);
        pipe_.InitBuffer(flagsBuf_, rowBytes);
        pipe_.InitBuffer(prefixBuf_, rowBytes);
        pipe_.InitBuffer(cumSumWorkspaceBuf_, kCumSumWorkspaceBytes);
        pipe_.InitBuffer(nonNegativeMaskBuf_, maskBytes);
        pipe_.InitBuffer(beforeBoundaryMaskBuf_, maskBytes);
        pipe_.InitBuffer(selectedMaskBuf_, maskBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        for (uint32_t row = core; row < rowCount_; row += coreCount_) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const uint32_t request = row / rowsPerRequest_;
        const uint32_t requestRow = row % rowsPerRequest_;
        const uint64_t inputOffset =
            static_cast<uint64_t>(row) * rowWidth_;
        const uint64_t packedOffset =
            static_cast<uint64_t>(request) * scratchCapacity_
            + static_cast<uint64_t>(requestRow) * rowWidth_;
        const int32_t rowRequest = rowReqIndices_.GetValue(row);

        auto input = inputBuf_.Get<int32_t>();
        auto clamped = clampedBuf_.Get<int32_t>();
        auto packed = packedBuf_.Get<int32_t>();
        auto flags = flagsBuf_.Get<float>();
        auto prefix = prefixBuf_.Get<float>();
        auto workspace = cumSumWorkspaceBuf_.Get<uint8_t>();
        auto nonNegativeMask = nonNegativeMaskBuf_.Get<uint8_t>();
        auto beforeBoundaryMask = beforeBoundaryMaskBuf_.Get<uint8_t>();
        auto selectedMask = selectedMaskBuf_.Get<uint8_t>();

        AscendC::DataCopy(
            input, topkIndices_[inputOffset], rowWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();

        if (rowRequest < 0) {
            AscendC::Duplicate(
                packed, static_cast<int32_t>(0x7FFFFFFF), rowWidth_);
            if (clearInvalidRows_) {
                AscendC::Duplicate(
                    input, static_cast<int32_t>(0), rowWidth_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(
                rowPacked_[packedOffset], packed, rowWidth_);
            if (clearInvalidRows_) {
                AscendC::DataCopy(
                    topkIndices_[inputOffset], input, rowWidth_);
            }
            rowCounts_.SetValue(packedOffset, 0);
            return;
        }

        const int32_t boundary = splitBoundary_.GetValue(row);
        if (boundary <= 0) {
            AscendC::Duplicate(
                packed, static_cast<int32_t>(0x7FFFFFFF), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(
                rowPacked_[packedOffset], packed, rowWidth_);
            rowCounts_.SetValue(packedOffset, 0);
            return;
        }
        AscendC::Maxs(
            clamped, input, static_cast<int32_t>(0), rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            nonNegativeMask,
            clamped,
            input,
            AscendC::CMPMODE::EQ,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(
            clamped, input, boundary - 1, rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            beforeBoundaryMask,
            clamped,
            input,
            AscendC::CMPMODE::EQ,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(
            selectedMask.ReinterpretCast<uint16_t>(),
            nonNegativeMask.ReinterpretCast<uint16_t>(),
            beforeBoundaryMask.ReinterpretCast<uint16_t>(),
            rowWidth_ / 16);
        AscendC::PipeBarrier<PIPE_V>();

        // INT32_MAX converts to the smallest sort key after negation, keeping
        // every padded element behind all valid non-negative token positions.
        AscendC::Duplicate(
            packed, static_cast<int32_t>(0x7FFFFFFF), rowWidth_);
        AscendC::Duplicate(prefix, 1.0F, rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            flags,
            selectedMask,
            prefix,
            0.0F,
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::GatherMaskParams gatherParams;
        gatherParams.repeatTimes = 1;
        gatherParams.src0BlockStride = 1;
        gatherParams.src0RepeatStride = 8;
        gatherParams.src1RepeatStride = 8;
        uint64_t selectedCount = 0;
        AscendC::GatherMask(
            packed,
            input,
            selectedMask.ReinterpretCast<uint32_t>(),
            true,
            rowWidth_,
            gatherParams,
            selectedCount);
        AscendC::PipeBarrier<PIPE_V>();

        float carry = 0.0F;
        auto lastRow = clamped.ReinterpretCast<float>();
        for (uint32_t tileOffset = 0; tileOffset < rowWidth_;
             tileOffset += kCumSumTileWidth) {
            const AscendC::CumSumInfo info{1, kCumSumTileWidth};
            auto tilePrefix = prefix[tileOffset];
            auto tileFlags = flags[tileOffset];
            AscendC::CumSum<float, kCumSumConfig>(
                tilePrefix,
                lastRow,
                tileFlags,
                workspace,
                info);
            AscendC::PipeBarrier<PIPE_V>();
            if (tileOffset != 0) {
                Sync<AscendC::HardEvent::S_V>();
                AscendC::Adds(
                    tilePrefix, tilePrefix, carry, kCumSumTileWidth);
                AscendC::PipeBarrier<PIPE_V>();
            }
            Sync<AscendC::HardEvent::V_S>();
            carry = prefix.GetValue(
                tileOffset + kCumSumTileWidth - 1);
        }

        AscendC::Cast(
            clamped,
            prefix,
            AscendC::RoundMode::CAST_ROUND,
            rowWidth_);
        AscendC::Adds(clamped, clamped, -1, rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            input.ReinterpretCast<float>(),
            selectedMask,
            clamped.ReinterpretCast<float>(),
            input.ReinterpretCast<float>(),
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            topkIndices_[inputOffset], input, rowWidth_);
        AscendC::DataCopy(
            rowPacked_[packedOffset], packed, rowWidth_);
        rowCounts_.SetValue(
            packedOffset, static_cast<int32_t>(selectedCount));
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> rowPacked_;
    AscendC::GlobalTensor<int32_t> rowCounts_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> clampedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> flagsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prefixBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cumSumWorkspaceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> nonNegativeMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> beforeBoundaryMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestCount_ = 0;
    uint32_t rowsPerRequest_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t coreCount_ = 0;
    bool clearInvalidRows_ = false;
};
class DSAStagedSortUnionKernel {
public:
    __aicore__ inline void InitProduction(
        __gm__ int32_t* rowPacked,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* localToUnion,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int64_t* targetSlots,
        __gm__ int32_t* rowCounts,
        uint32_t requestCount,
        uint32_t rowWidth,
        uint32_t scratchCapacity,
        uint32_t blockTableWidth,
        uint32_t selectedCountStride,
        uint32_t blockSize,
        bool needPacked)
    {
        Init(
            rowPacked,
            selectedPacked,
            localToUnion,
            selectedCount,
            requestBlockTable,
            targetSlots,
            2 * requestCount,
            rowWidth,
            blockTableWidth,
            selectedCountStride,
            blockSize);
        scratchCapacity_ = scratchCapacity;
        boundedRows_ = true;
        needPacked_ = needPacked;
        rowPacked_.SetGlobalBuffer(
            rowPacked,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        selectedPacked_.SetGlobalBuffer(
            selectedPacked,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        localToUnion_.SetGlobalBuffer(
            localToUnion,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        targetSlots_.SetGlobalBuffer(
            targetSlots,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        rowCounts_.SetGlobalBuffer(
            rowCounts,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
    }

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
        uint32_t selectedCountStride,
        uint32_t blockSize)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        requestCount_ = rowCount / 2;
        requestWidth_ = 2 * rowWidth;
        scratchCapacity_ = requestWidth_;
        blockTableWidth_ = blockTableWidth;
        selectedCountStride_ = selectedCountStride;
        blockSize_ = blockSize;
        uint32_t shiftedBlockSize = blockSize_;
        while (shiftedBlockSize > 1) {
            shiftedBlockSize >>= 1;
            ++blockSizeShift_;
        }
        rowPacked_.SetGlobalBuffer(rowPacked, rowCount * rowWidth);
        selectedPacked_.SetGlobalBuffer(
            selectedPacked, requestCount_ * requestWidth_);
        localToUnion_.SetGlobalBuffer(localToUnion, rowCount * rowWidth);
        selectedCount_.SetGlobalBuffer(
            selectedCount, requestCount_ * selectedCountStride_);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable, requestCount_ * blockTableWidth);
        targetSlots_.SetGlobalBuffer(
            targetSlots, requestCount_ * requestWidth_);
        pipe_.InitBuffer(
            sortSrcBuf_, requestWidth_ * kPairWidth * sizeof(float));
        pipe_.InitBuffer(
            sortTmpBuf_, requestWidth_ * kPairWidth * sizeof(float));
        pipe_.InitBuffer(
            sortInputBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            unionBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            mappingBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            targetBuf_, requestWidth_ * sizeof(int64_t));
        pipe_.InitBuffer(
            blockTableBuf_, blockTableWidth_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        const uint64_t rowOffset =
            static_cast<uint64_t>(request) * scratchCapacity_;
        const uint64_t outputOffset =
            static_cast<uint64_t>(request) * scratchCapacity_;
        auto src = sortSrcBuf_.Get<float>();
        auto tmp = sortTmpBuf_.Get<float>();
        auto input = sortInputBuf_.Get<int32_t>();
        auto unionLocal = unionBuf_.Get<int32_t>();
        auto mapping = mappingBuf_.Get<int32_t>();
        auto targets = targetBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();
        auto srcInt = src.ReinterpretCast<int32_t>();
        uint32_t validElements = requestWidth_;
        if (boundedRows_) {
            validElements = static_cast<uint32_t>(
                rowCounts_.GetValue(rowOffset))
                + static_cast<uint32_t>(
                    rowCounts_.GetValue(rowOffset + rowWidth_));
            if (validElements == 0) {
                selectedCount_.SetValue(
                    request * selectedCountStride_, 0);
                return;
            }
            AscendC::DataCopy(
                input, rowPacked_[rowOffset], rowWidth_);
            AscendC::DataCopy(
                input[rowWidth_],
                rowPacked_[rowOffset + rowWidth_],
                rowWidth_);
        } else {
            AscendC::DataCopy(
                input, rowPacked_[rowOffset], requestWidth_);
        }
        if (needPacked_) {
            CopyGlobalToLocalExact(
                blockTable,
                requestBlockTable_[
                    static_cast<uint64_t>(request) * blockTableWidth_],
                blockTableWidth_);
        }
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(
            src, input, AscendC::RoundMode::CAST_NONE, requestWidth_);
        AscendC::Muls(src, src, -1.0F, requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CreateVecIndex(
            srcInt[requestWidth_], static_cast<int32_t>(0), requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        SortAll(src, tmp);
        Sync<AscendC::HardEvent::V_S>();

        auto sortedInt = src.ReinterpretCast<int32_t>();
        int32_t previous = -1;
        uint32_t rank = 0;
        for (uint32_t i = 0; i < validElements; ++i) {
            const int32_t token =
                -static_cast<int32_t>(src.GetValue(kPairWidth * i));
            const uint32_t original = static_cast<uint32_t>(
                sortedInt.GetValue(kPairWidth * i + 1));
            if (i == 0 || token != previous) {
                unionLocal.SetValue(rank, token);
                previous = token;
                ++rank;
            }
            mapping.SetValue(original, static_cast<int32_t>(rank - 1));
        }

        if (needPacked_ && rank != 0) {
            Sync<AscendC::HardEvent::S_V>();
            auto ranks = src.ReinterpretCast<int32_t>();
            auto logicalBlocks = ranks[requestWidth_];
            auto physicalBlocks = tmp.ReinterpretCast<int32_t>();
            auto blockTableOffsets = physicalBlocks[requestWidth_];
            AscendC::CreateVecIndex(
                ranks, static_cast<int32_t>(0), rank);
            AscendC::ShiftRight(
                logicalBlocks,
                ranks,
                static_cast<int32_t>(blockSizeShift_),
                rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                blockTableOffsets,
                logicalBlocks,
                static_cast<int32_t>(sizeof(int32_t)),
                rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(
                physicalBlocks,
                blockTable,
                blockTableOffsets.ReinterpretCast<uint32_t>(),
                static_cast<uint32_t>(0),
                rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                physicalBlocks,
                physicalBlocks,
                static_cast<int32_t>(blockSize_),
                rank);
            AscendC::Muls(
                logicalBlocks,
                logicalBlocks,
                static_cast<int32_t>(blockSize_),
                rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(ranks, ranks, logicalBlocks, rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(
                physicalBlocks, physicalBlocks, ranks, rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                targets,
                physicalBlocks,
                AscendC::RoundMode::CAST_NONE,
                rank);
            AscendC::PipeBarrier<PIPE_V>();
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            localToUnion_[rowOffset], mapping, requestWidth_);
        if (needPacked_) {
            CopyLocalToGlobalExact(
                selectedPacked_[outputOffset], unionLocal, rank);
            CopyLocalToGlobalExact(
                targetSlots_[outputOffset], targets, rank);
        }
        selectedCount_.SetValue(
            request * selectedCountStride_,
            needPacked_ ? static_cast<int32_t>(rank) : 0);
    }

private:
    __aicore__ inline void SortAll(
        AscendC::LocalTensor<float>& src,
        AscendC::LocalTensor<float>& tmp)
    {
        const uint32_t repeats = requestWidth_ / kSortGroup;
        AscendC::Sort32(
            tmp,
            src,
            src[requestWidth_].ReinterpretCast<uint32_t>(),
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
                src, tmp, requestWidth_ * kPairWidth);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::GlobalTensor<int32_t> rowPacked_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<int32_t> rowCounts_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortSrcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortInputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> unionBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mappingBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockTableBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestCount_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t blockSizeShift_ = 0;
    bool boundedRows_ = false;
    bool needPacked_ = true;
};
class DSAStagedSingleRowFinalizeKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* rowCounts,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int64_t* targetSlots,
        uint32_t requestCount,
        uint32_t rowWidth,
        uint32_t scratchCapacity,
        uint32_t blockTableWidth,
        uint32_t selectedCountStride,
        uint32_t blockSize,
        bool needPacked)
    {
        requestCount_ = requestCount;
        rowWidth_ = rowWidth;
        scratchCapacity_ = scratchCapacity;
        blockTableWidth_ = blockTableWidth;
        selectedCountStride_ = selectedCountStride;
        blockSize_ = blockSize;
        needPacked_ = needPacked;
        uint32_t shifted = blockSize_;
        while (shifted > 1) {
            shifted >>= 1;
            ++blockSizeShift_;
        }
        rowCounts_.SetGlobalBuffer(
            rowCounts,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        selectedCount_.SetGlobalBuffer(
            selectedCount,
            static_cast<uint64_t>(requestCount_) * selectedCountStride_);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable,
            static_cast<uint64_t>(requestCount_) * blockTableWidth_);
        targetSlots_.SetGlobalBuffer(
            targetSlots,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        pipe_.InitBuffer(rankBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(logicalBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(physicalBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(offsetBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(targetBuf_, rowWidth_ * sizeof(int64_t));
        pipe_.InitBuffer(
            blockTableBuf_, blockTableWidth_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        if (!needPacked_) {
            selectedCount_.SetValue(
                static_cast<uint64_t>(request) * selectedCountStride_, 0);
            return;
        }
        const uint64_t outputOffset =
            static_cast<uint64_t>(request) * scratchCapacity_;
        const uint32_t count = static_cast<uint32_t>(
            rowCounts_.GetValue(outputOffset));
        if (count == 0) {
            selectedCount_.SetValue(
                static_cast<uint64_t>(request) * selectedCountStride_, 0);
            return;
        }

        auto ranks = rankBuf_.Get<int32_t>();
        auto logical = logicalBuf_.Get<int32_t>();
        auto physical = physicalBuf_.Get<int32_t>();
        auto offsets = offsetBuf_.Get<int32_t>();
        auto targets = targetBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();
        if (needPacked_) {
            CopyGlobalToLocalExact(
                blockTable,
                requestBlockTable_[
                    static_cast<uint64_t>(request) * blockTableWidth_],
                blockTableWidth_);
        }
        Sync<AscendC::HardEvent::MTE2_V>();

        AscendC::CreateVecIndex(
            ranks, static_cast<int32_t>(0), count);
        AscendC::ShiftRight(
            logical,
            ranks,
            static_cast<int32_t>(blockSizeShift_),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            offsets,
            logical,
            static_cast<int32_t>(sizeof(int32_t)),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(
            physical,
            blockTable,
            offsets.ReinterpretCast<uint32_t>(),
            static_cast<uint32_t>(0),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            physical,
            physical,
            static_cast<int32_t>(blockSize_),
            count);
        AscendC::Muls(
            logical,
            logical,
            static_cast<int32_t>(blockSize_),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(ranks, ranks, logical, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(physical, physical, ranks, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            targets,
            physical,
            AscendC::RoundMode::CAST_NONE,
            count);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            targetSlots_[outputOffset], targets, count);
        selectedCount_.SetValue(
            static_cast<uint64_t>(request) * selectedCountStride_,
            static_cast<int32_t>(count));
    }

private:
    AscendC::GlobalTensor<int32_t> rowCounts_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rankBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> logicalBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> physicalBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockTableBuf_;
    uint32_t requestCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t blockSizeShift_ = 0;
    bool needPacked_ = true;
};
class DSAStagedBoundaryRemapKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* localToUnion,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t rowsPerRequest,
        uint32_t scratchCapacity,
        uint32_t coreCount)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        rowsPerRequest_ = rowsPerRequest;
        scratchCapacity_ = scratchCapacity;
        coreCount_ = coreCount;
        topkIndices_.SetGlobalBuffer(
            topkIndices,
            static_cast<uint64_t>(rowCount_) * rowWidth_);
        splitBoundary_.SetGlobalBuffer(splitBoundary, rowCount_);
        rowReqIndices_.SetGlobalBuffer(rowReqIndices, rowCount_);
        const uint32_t requestCount = rowCount_ / rowsPerRequest_;
        localToUnion_.SetGlobalBuffer(
            localToUnion,
            static_cast<uint64_t>(requestCount) * scratchCapacity_);
        const uint32_t rowBytes = rowWidth_ * sizeof(int32_t);
        const uint32_t maskBytes = rowWidth_ / 8;
        pipe_.InitBuffer(inputBuf_, rowBytes);
        pipe_.InitBuffer(clampedBuf_, rowBytes);
        pipe_.InitBuffer(mappingBuf_, rowBytes);
        pipe_.InitBuffer(outputBuf_, rowBytes);
        pipe_.InitBuffer(offsetBuf_, rowBytes);
        pipe_.InitBuffer(nonNegativeMaskBuf_, maskBytes);
        pipe_.InitBuffer(beforeBoundaryMaskBuf_, maskBytes);
        pipe_.InitBuffer(selectedMaskBuf_, maskBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        for (uint32_t row = core; row < rowCount_; row += coreCount_) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        if (rowReqIndices_.GetValue(row) < 0) {
            // The compact stage already zeroed graph padding rows.
            return;
        }
        const uint32_t request = row / rowsPerRequest_;
        const uint32_t requestRow = row % rowsPerRequest_;
        const uint64_t inputOffset =
            static_cast<uint64_t>(row) * rowWidth_;
        const uint64_t mappingOffset =
            static_cast<uint64_t>(request) * scratchCapacity_
            + static_cast<uint64_t>(requestRow) * rowWidth_;
        const int32_t boundary = splitBoundary_.GetValue(row);
        if (boundary <= 0) {
            return;
        }

        auto input = inputBuf_.Get<int32_t>();
        auto clamped = clampedBuf_.Get<int32_t>();
        auto mapping = mappingBuf_.Get<int32_t>();
        auto output = outputBuf_.Get<int32_t>();
        auto offsets = offsetBuf_.Get<uint32_t>();
        auto nonNegativeMask = nonNegativeMaskBuf_.Get<uint8_t>();
        auto beforeBoundaryMask = beforeBoundaryMaskBuf_.Get<uint8_t>();
        auto selectedMask = selectedMaskBuf_.Get<uint8_t>();
        AscendC::DataCopy(
            input, topkIndices_[inputOffset], rowWidth_);
        AscendC::DataCopy(
            mapping, localToUnion_[mappingOffset], rowWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();

        // After compaction selected positions contain a local rank. Because a
        // row is unique, every local rank is below its split boundary, while
        // ignored absolute positions are at or above that boundary.
        AscendC::Maxs(
            clamped, input, static_cast<int32_t>(0), rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            nonNegativeMask,
            clamped,
            input,
            AscendC::CMPMODE::EQ,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(
            clamped, input, boundary - 1, rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            beforeBoundaryMask,
            clamped,
            input,
            AscendC::CMPMODE::EQ,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(
            selectedMask.ReinterpretCast<uint16_t>(),
            nonNegativeMask.ReinterpretCast<uint16_t>(),
            beforeBoundaryMask.ReinterpretCast<uint16_t>(),
            rowWidth_ / 16);
        AscendC::PipeBarrier<PIPE_V>();

        // Clamp every lane before Gather; ignored absolute positions may be
        // much larger than the row-local map, even though Select discards
        // their gathered values.
        AscendC::Maxs(
            clamped, input, static_cast<int32_t>(0), rowWidth_);
        AscendC::Mins(
            clamped,
            clamped,
            static_cast<int32_t>(rowWidth_ - 1),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            offsets.ReinterpretCast<int32_t>(),
            clamped,
            static_cast<int32_t>(sizeof(int32_t)),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(
            output,
            mapping,
            offsets,
            static_cast<uint32_t>(0),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            output.ReinterpretCast<float>(),
            selectedMask,
            output.ReinterpretCast<float>(),
            input.ReinterpretCast<float>(),
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            topkIndices_[inputOffset], output, rowWidth_);
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> clampedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mappingBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> nonNegativeMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> beforeBoundaryMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t rowsPerRequest_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t coreCount_ = 0;
};
extern "C" __global__ __aicore__ void dsa_staged_compact_rows_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* rowPacked,
    __gm__ int32_t* rowCounts,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t scratchCapacity,
    uint32_t coreCount,
    bool clearInvalidRows)
{
    if ASCEND_IS_AIV {
        DSAStagedCompactRowsKernel op;
        op.Init(
            topkIndices, splitBoundary, rowReqIndices, rowPacked, rowCounts,
            rowCount, rowWidth, requestCount, rowsPerRequest,
            scratchCapacity, coreCount, clearInvalidRows);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_production_sort_union_kernel(
    __gm__ int32_t* rowPacked,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* localToUnion,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    __gm__ int32_t* rowCounts,
    uint32_t requestCount,
    uint32_t rowWidth,
    uint32_t scratchCapacity,
    uint32_t blockTableWidth,
    uint32_t selectedCountStride,
    uint32_t blockSize,
    bool needPacked)
{
    if ASCEND_IS_AIV {
        DSAStagedSortUnionKernel op;
        op.InitProduction(
            rowPacked, selectedPacked, localToUnion, selectedCount,
            requestBlockTable, targetSlots, rowCounts, requestCount,
            rowWidth, scratchCapacity, blockTableWidth,
            selectedCountStride, blockSize, needPacked);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_single_row_finalize_kernel(
    __gm__ int32_t* rowCounts,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    uint32_t requestCount,
    uint32_t rowWidth,
    uint32_t scratchCapacity,
    uint32_t blockTableWidth,
    uint32_t selectedCountStride,
    uint32_t blockSize,
    bool needPacked)
{
    if ASCEND_IS_AIV {
        DSAStagedSingleRowFinalizeKernel op;
        op.Init(
            rowCounts, selectedCount, requestBlockTable, targetSlots,
            requestCount, rowWidth, scratchCapacity, blockTableWidth,
            selectedCountStride, blockSize, needPacked);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_boundary_remap_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* localToUnion,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t rowsPerRequest,
    uint32_t scratchCapacity,
    uint32_t coreCount)
{
    if ASCEND_IS_AIV {
        DSAStagedBoundaryRemapKernel op;
        op.Init(
            topkIndices, splitBoundary, rowReqIndices, localToUnion,
            rowCount, rowWidth, rowsPerRequest, scratchCapacity,
            coreCount);
        op.Process();
    }
}

}  // namespace

namespace vllm_ascend {

void dsa_prepare_sparse_indices_staged_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* rowReqIndices, void* requestBlockTable,
    void* selectedPacked, void* selectedCount, void* targetSlots,
    void* localToUnion, uint32_t rowCount, uint32_t rowWidth,
    uint32_t requestCount, uint32_t rowsPerRequest,
    uint32_t scratchCapacity, uint32_t blockTableWidth,
    uint32_t selectedCountStride, uint32_t blockSize,
    uint32_t coreCount, bool needPacked, bool clearInvalidRows)
{
    dsa_staged_compact_rows_kernel<<<coreCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        rowCount, rowWidth, requestCount, rowsPerRequest,
        scratchCapacity, coreCount, clearInvalidRows);

    if (rowsPerRequest == 1) {
        dsa_staged_single_row_finalize_kernel<<<
            requestCount, nullptr, stream>>>(
            static_cast<int32_t*>(localToUnion),
            static_cast<int32_t*>(selectedCount),
            static_cast<int32_t*>(requestBlockTable),
            static_cast<int64_t*>(targetSlots),
            requestCount, rowWidth, scratchCapacity,
            blockTableWidth, selectedCountStride, blockSize,
            needPacked);
        return;
    }

    dsa_staged_production_sort_union_kernel<<<
        requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(localToUnion),
        requestCount, rowWidth, scratchCapacity,
        blockTableWidth, selectedCountStride, blockSize,
        needPacked);
    dsa_staged_boundary_remap_kernel<<<coreCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(localToUnion),
        rowCount, rowWidth, rowsPerRequest, scratchCapacity,
        coreCount);
}

}  // namespace vllm_ascend

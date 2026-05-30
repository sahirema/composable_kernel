// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_gemm_common.hpp"

using F8   = ck::f8_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

namespace {

template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<std::tuple<Xs...>, std::tuple<Ys...>>
{
    using type = std::tuple<Xs..., Ys...>;
};

} // namespace

template <typename Tuple>
class TestGemmMultiplyMultiplyWP_FP8_MK_NK
    : public ck::test::TestGemmMultiplyMultiplyWPCommon<
          typename tuple_concat<std::tuple<Row, Col, Row, Col>, Tuple>::type>
{
};

// clang-format off
using KernelTypes_MK_NK = ::testing::Types<
#if defined(CK_ENABLE_FP8)
    std::tuple< F8, F8, F8, F32, F32, F16>,
    std::tuple< F8, F8, F8, F32, F32, BF16>
#endif
    >;
// clang-format on

TYPED_TEST_SUITE(TestGemmMultiplyMultiplyWP_FP8_MK_NK, KernelTypes_MK_NK);

TYPED_TEST(TestGemmMultiplyMultiplyWP_FP8_MK_NK, Regular0)
{
    std::vector<int> Ms{128, 224, 256, 448, 512};
    constexpr int N = 512;
    constexpr int K = 2048;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestGemmMultiplyMultiplyWP_FP8_MK_NK, Regular1)
{
    std::vector<int> Ms{128, 224, 256, 448, 512};
    constexpr int N = 1024;
    constexpr int K = 4096;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestGemmMultiplyMultiplyWP_FP8_MK_NK, Regular2)
{
    std::vector<int> Ms{128, 256, 512};
    constexpr int N = 448;
    constexpr int K = 2048;

    for(int M : Ms)
        this->Run(M, N, K);
}

// The following cases cover the GemmSpec-aware N/K padding paths in
// DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle. They use problem sizes whose N (resp. K) is
// not a multiple of any instance's NPerBlock (resp. KPerBlock), so only instances built with
// an N/K-padding GemmSpec are selected and verified; instances without the matching padding
// spec report IsSupportedArgument() == false and are skipped. N is kept a multiple of 32 and K
// a multiple of 64 so the host-side weight preshuffle stays exact -- the partial trailing block
// tile is exactly the region the right-pad B/C descriptors zero-fill on the device.

TYPED_TEST(TestGemmMultiplyMultiplyWP_FP8_MK_NK, NPadding)
{
    std::vector<int> Ms{128, 256};
    constexpr int N = 320; // not a multiple of NPerBlock (128/256/512)
    constexpr int K = 2048;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestGemmMultiplyMultiplyWP_FP8_MK_NK, KPadding)
{
    std::vector<int> Ms{128, 256};
    constexpr int N = 512;
    constexpr int K = 2112; // multiple of 64 but not of KPerBlock (128/256/512)

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestGemmMultiplyMultiplyWP_FP8_MK_NK, NKPadding)
{
    std::vector<int> Ms{128, 256};
    constexpr int N = 320; // N padding
    constexpr int K = 2112; // K padding

    for(int M : Ms)
        this->Run(M, N, K);
}

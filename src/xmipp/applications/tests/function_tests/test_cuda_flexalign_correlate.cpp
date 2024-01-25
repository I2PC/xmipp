#include <gtest/gtest.h>
#include <random>
#include <limits>

#include "reconstruction_cuda/cuda_flexalign_correlate.h"
#include "reconstruction_adapt_cuda/basic_mem_manager.h"
#include "reconstruction_cuda/cuda_fft.h"
#include "core/xmipp_image.h"


size_t correlations(const Dimensions &d) {
    return d.n() * (d.n()-1) / 2;
}

template<typename T>
class FlexAlignCorrelateTest : public ::testing::Test {
public:
    T* createData(const Dimensions &dim) {
        auto settings = FFTSettings<T>(dim);
        auto *sd = reinterpret_cast<T*>(BasicMemManager::instance().get(settings.sBytes(), MemType::CUDA_HOST));
        memset(sd, 0, settings.sBytes());

        for (auto n = 0; n < dim.n(); ++n) {
            auto offset = n * dim.sizeSingle();
            // draw point in the center offsetted by n
            sd[offset + (n + dim.y() / 2) * dim.x() + dim.x() / 2 + n] = 1;
        }

        // debug
        // auto mda = MultidimArray<T>(dim.n(), dim.z(), dim.y(), dim.x(), sd);
        // auto img = Image<T>(mda);
        // img.write("data.mrc");

        return sd;
    }

    float* createPos(const Dimensions &dim) {
        const auto size = 2 * posEdge + correlations(dim);
        posReal = reinterpret_cast<T*>(BasicMemManager::instance().get(size * 2 * sizeof(float), MemType::CUDA_HOST));
        for (auto i = 0; i < size * 2; ++i) {
            posReal[i] = std::numeric_limits<float>::infinity();
        }
        return posReal + 2 * posEdge;
    }

    void testPos(const Dimensions &dim) {
        const auto delta = 0.0001f;

        auto *b = posReal;
        for (auto i = 0; i < posEdge * 2; ++i) {
            ASSERT_EQ(std::numeric_limits<float>::infinity(), b[i]) << " at " << i;
        }

        auto *pos = posReal + 2 * posEdge;
        auto index = 0;
        for (auto i = 0; i < dim.n(); ++i) {
            for (auto j = i + 1; j < dim.n(); ++j, ++index) {
                auto normalize = [i, j, &dim](auto &x, auto &y) {
                    x -= static_cast<float>(dim.x()) / 2.f;
                    y -= static_cast<float>(dim.y()) / 2.f;
                };
                auto x = pos[index * 2];
                auto y = pos[index * 2 + 1];
                normalize(x, y);
                // printf("%lu - %lu: [%f, %f] \n", i, j, x, y);
                ASSERT_NEAR(x, i - j, delta) << " at " << index;
                ASSERT_NEAR(y, i - j, delta) << " at " << index;
            }
        }

        const auto size = posEdge + dim.n() * (dim.n() -1) / 2;
        auto *e = posReal + size * 2;
        for (auto i = 0; i < posEdge * 2; ++i) {
            ASSERT_EQ(std::numeric_limits<float>::infinity(), e[i]) << " at " << i;
        }
    }


    void run(const Dimensions &dim, size_t batch, size_t bufferSize) {
        assert(dim.x() > dim.n() * 2);
        assert(dim.y() > dim.n() * 2); // because of the shift
        GPU gpu;
        gpu.set();

        const auto params = typename CUDAFlexAlignCorrelate<T>::Params {
            .dim = dim,
            .bufferSize = bufferSize,
            .batch = batch,
        };

        auto settings = FFTSettings<T>(dim, dim.n());
        auto *sd = createData(dim);

        auto *fd = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(settings.fBytes(), MemType::CUDA_HOST));
        auto transformer = CudaFFT<T>();
        transformer.init(gpu, settings);
        transformer.fft(sd, fd);

        auto *pos = createPos(dim);

        auto correlator = CUDAFlexAlignCorrelate<T>(params, gpu);
        correlator.init();
        correlator.run(fd, pos, std::sqrt(2.f * std::pow(static_cast<float>(dim.n()), 2)));
        correlator.synch();

        testPos(dim);

        BasicMemManager::instance().give(sd);
        BasicMemManager::instance().give(fd);
        BasicMemManager::instance().give(posReal);
    }

private:
    const size_t posEdge = 100;
    float *posReal;

};

TYPED_TEST_SUITE_P(FlexAlignCorrelateTest);

TYPED_TEST_P(FlexAlignCorrelateTest, Batch) {
    const auto dim = Dimensions (42, 24, 1, 10);
    for (auto batch = 1; batch <= correlations(dim); ++batch) {
        this->run(dim, batch, 1);
    }
}

TYPED_TEST_P(FlexAlignCorrelateTest, BufferSize) {
    const auto dim = Dimensions (24, 42, 1, 10);
    for (auto buffer = 1; buffer <= dim.n(); ++buffer) {
        this->run(dim, 1, buffer);
    }
}

TYPED_TEST_P(FlexAlignCorrelateTest, BatchBufferSize) {
    const auto dim = Dimensions (36, 86, 1, 17);
    for (auto batch = 1; batch <= correlations(dim); ++batch) {
        for (auto buffer = 1; buffer <= dim.n(); ++buffer) {
            this->run(dim, batch, buffer);
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(FlexAlignCorrelateTest,
    Batch,
    BufferSize,
    BatchBufferSize
);

using ScalarTypes = ::testing::Types< float >;
INSTANTIATE_TYPED_TEST_SUITE_P(CUDAFlexAlignCorrelate, FlexAlignCorrelateTest, ScalarTypes);

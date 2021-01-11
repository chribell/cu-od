#include <iostream>
#include <cxxopts.hpp>
#include <thrust/reduce.h>
#include "io.hpp"
#include "device_timer.cuh"
#include "helpers.cuh"

struct filter {
    unsigned int k;

    filter(unsigned int k) : k(k) {}

    __host__ __device__ unsigned int operator()(const unsigned int a) const {
        return a < k;
    }
};

template <typename T, bool weighted>
__global__ void computeEuclideanParallel(const T* pointsRow, const T* pointsCol, const T* weights,
                                         unsigned int cardinality, unsigned int dimensions, unsigned int chunkSize,
                                         unsigned int offset, unsigned int size, double threshold, unsigned int* neighbors);

int main(int argc, char** argv) {
    try {
        fmt::print("{}\n", "GPU Naive outlier detection");

        int multiprocessorCount;
        int maxThreadsPerBlock;

        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        // arguments
        std::string input;
        std::string output = "out.txt";
        std::string weightsFile;
        bool hasWeights = false;
        unsigned int cardinality;
        unsigned int dimensions;
        unsigned int blocks = multiprocessorCount * 16;
        unsigned int blockSize = maxThreadsPerBlock / 2;
        unsigned int chunkSize = 10;
        unsigned int windowSize = 10000;
        unsigned int slideSize = 500;
        unsigned int k = 50;
        double threshold = 2.0;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("i,input", "Input dataset path", cxxopts::value<std::string>(input))
                ("o,output", "Output result path", cxxopts::value<std::string>(output))
                // Outlier detection related arguments
                ("w,window", "Window size (must be multiple of slide, default: 10000)", cxxopts::value<unsigned int>(windowSize))
                ("s,slide", "Slide size (default: 500)", cxxopts::value<unsigned int>(slideSize))
                ("c,cardinality", "Dataset cardinality", cxxopts::value<unsigned int>(cardinality))
                ("d,dimensions", "Point dimensionality", cxxopts::value<unsigned int>(dimensions))
                ("k", "Minimum number of neighbors to consider a point inlier", cxxopts::value<unsigned int>(k))
                ("t,threshold", "Threshold value (default: 2.0)", cxxopts::value<double>(threshold))
                ("weights", "Weight file (the number of weights must be equal to the number of dimensions)", cxxopts::value<std::string>(weightsFile))
                // GPU related arguments
                ("chunk", "Chunk size (points assigned per blocked)", cxxopts::value<unsigned int>(chunkSize))
                ("blocks", "Number of blocks (default: " + std::to_string(blocks) + ")", cxxopts::value<unsigned int>(blocks))
                ("threads", "Threads per block (default: " + std::to_string(blockSize) + ")", cxxopts::value<unsigned int>(blockSize))
                ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            fmt::print("{}\n", options.help());
            return 0;
        }

        if (!result.count("input")) {
            fmt::print("{}\n", "No input dataset given! Exiting...");
            return 1;
        }

        if (!result.count("cardinality")) {
            fmt::print("{}\n", "No cardinality given! Exiting...");
            return 1;
        }

        if (!result.count("dimensions")) {
            fmt::print("{}\n", "No dimensions given! Exiting...");
            return 1;
        }

        if (windowSize % slideSize != 0) {
            fmt::print("{}\n", "Error, window size must be multiple of slide size! Exiting...");
            return 1;
        }

        float* weights = new float[dimensions]{1.00};

        if (result.count("weights")) {
            hasWeights = true;
            readWeights(weightsFile, weights);
        }

        Dataset* d = readDataset(input, cardinality, dimensions);

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "│{9: ^{2}}|{10: ^{2}}│\n"
                "└{11:─^{1}}┘\n", "Query", 51, 25,
                "Window size (w)", windowSize,
                "Slide size (s)", slideSize,
                "Min. neighbors (k)", k,
                "Threshold", threshold, ""
        );

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "└{9:─^{1}}┘\n", "Dataset characteristics", 51, 25,
                "Cardinality", d->cardinality,
                "Dimensions", d->dimensions,
                "Total elements", d->cardinality * d->dimensions, ""
        );

        DeviceTimer deviceTimer;

        float* devicePointsRow;
        float* devicePointsCol;
        float* deviceWeights;
        unsigned int* deviceNeighbors;

        EventPair* devMemAlloc = deviceTimer.add("Device memory allocation");
        errorCheck(cudaMalloc((void**) &devicePointsRow, sizeof(float) * d->cardinality * d->dimensions))
        errorCheck(cudaMalloc((void**) &devicePointsCol, sizeof(float) * d->cardinality * d->dimensions))
        errorCheck(cudaMalloc((void**) &deviceWeights, sizeof(float) * d->dimensions))
        errorCheck(cudaMalloc((void**) &deviceNeighbors, sizeof(unsigned int) * windowSize))
        DeviceTimer::finish(devMemAlloc);

        EventPair* dataTransfer = deviceTimer.add("Transfer to device");
        errorCheck(cudaMemcpy(devicePointsRow, d->pointsRow, sizeof(float) * d->cardinality * d->dimensions,
                              cudaMemcpyHostToDevice))
        errorCheck(cudaMemcpy(devicePointsCol, d->pointsCol, sizeof(float) * d->cardinality * d->dimensions,
                              cudaMemcpyHostToDevice))
        errorCheck(cudaMemcpy(deviceWeights, weights, sizeof(float) * d->dimensions,
                              cudaMemcpyHostToDevice))
        DeviceTimer::finish(dataTransfer);

        unsigned int sharedMem = 2 * chunkSize * d->dimensions * sizeof(float);
        cudaFuncSetCacheConfig(computeEuclideanParallel<float, true>, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(computeEuclideanParallel<float, false>, cudaFuncCachePreferL1);

        unsigned int currentSize = 0;
        unsigned int currentStart = 0;

        std::vector<unsigned int> counts;

        for (unsigned int i = 0; i < (cardinality / slideSize) + (cardinality % slideSize != 0 ? 1 : 0); ++i) {

            if (currentSize < windowSize) {
                currentSize += slideSize;
            } else {
                currentStart++;
            }

            EventPair* clearMem = deviceTimer.add("Clear neighbors");
            errorCheck(cudaMemset(deviceNeighbors, 0, sizeof(unsigned int) * currentSize))
            DeviceTimer::finish(clearMem);


            EventPair* calc = deviceTimer.add("Kernel");

            if (hasWeights) {
                computeEuclideanParallel<float, true><<<currentSize / chunkSize, dimensions, sharedMem>>>(
                        devicePointsRow,
                        devicePointsCol,
                        deviceWeights,
                        d->cardinality,
                        d->dimensions,
                        chunkSize,
                        (currentStart * slideSize),
                        currentSize,
                        threshold,
                        deviceNeighbors);
            } else {
                computeEuclideanParallel<float, false><<<currentSize / chunkSize, dimensions, sharedMem>>>(
                        devicePointsRow,
                        devicePointsCol,
                        deviceWeights,
                        d->cardinality,
                        d->dimensions,
                        chunkSize,
                        (currentStart * slideSize),
                        currentSize,
                        threshold,
                        deviceNeighbors);
            }

            unsigned int deviceRes = thrust::transform_reduce(thrust::device, deviceNeighbors,
                                                              deviceNeighbors + currentSize, filter(k), 0,
                                                              thrust::plus<unsigned int>());
            DeviceTimer::finish(calc);
            counts.push_back(deviceRes);
        }

        EventPair* freeDevMem = deviceTimer.add("Free device memory");
        errorCheck(cudaFree(devicePointsRow))
        errorCheck(cudaFree(devicePointsCol))
        errorCheck(cudaFree(deviceNeighbors))
        DeviceTimer::finish(freeDevMem);

        cudaDeviceSynchronize();
        deviceTimer.print();

        fmt::print("Writing results to {}\n", output);
        writeResult(counts, output);
        fmt::print("Finished!\n");

    } catch (const cxxopts::OptionException& e) {
        fmt::print("{}\n", e.what());
        return 1;
    }
    return 0;
}

template <typename T, bool weighted>
__global__ void computeEuclideanParallel(const T* pointsRow, const T* pointsCol, const T* weights,
                                         unsigned int cardinality, unsigned int dimensions, unsigned int chunkSize,
                                         unsigned int offset, unsigned int size, double threshold, unsigned int* neighbors) {
    extern T __shared__ s[];
    T* sharedPoints = s;
    T* result = (T*) &sharedPoints[chunkSize * dimensions] + threadIdx.x * chunkSize;

    unsigned int bx = blockIdx.x;

    while ((bx * chunkSize) < size) {
        unsigned int tx = threadIdx.x;
        // load points to shared memory
        for (int i = 0; i < chunkSize; i++)
            sharedPoints[(i * dimensions) + tx] = pointsRow[(dimensions * offset) + (((bx * chunkSize) + i) * dimensions) + tx];
        __syncthreads();

        while (tx < size) {
            for (int i = 0; i < chunkSize; i++)
                result[i] = 0.0f;
            for (int i = 0; i < dimensions; i++) {
                float tmp = pointsCol[(cardinality * i + offset) + tx];
                for (int j = 0; j < chunkSize; j++) {
                    float res = tmp - sharedPoints[i + (j * dimensions)];
                    if(weighted) {
                        result[j] += (res * res) * weights[i];
                    } else {
                        result[j] += res * res;
                    }
                }
            }
            for (int i = 0; i < chunkSize; i++) {
//                distances[((i + (bx * chunkSize)) * cardinality) + tx] = result[i];
                if (sqrtf(result[i]) <= threshold) {
                    unsigned int tmp = ((i + (bx * chunkSize)) * size) + tx;
                    unsigned int a = (tmp / size);
                    unsigned int b = (tmp % size);
                    if (a < b) {
                        atomicAdd(neighbors + a, 1);
                        atomicAdd(neighbors + b, 1);
                    }
                }
            }
            tx += blockDim.x;
        }
        __syncthreads();
        bx += gridDim.x;
    }
}
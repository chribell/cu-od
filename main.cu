#include <iostream>
#include <cxxopts.hpp>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include "io.hpp"
#include "device_timer.cuh"
#include "helpers.cuh"

struct Args {
    int multiprocessorCount{};
    int maxThreadsPerBlock{};
    std::string input;
    std::string output = "out.txt";
    std::string weightsFile;
    std::string precision = "float";
    bool hasWeights = false;
    unsigned int cardinality{};
    unsigned int dimensions{};
    unsigned int chunkSize = 10;
    unsigned int windowSize = 10000;
    unsigned int slideSize = 500;
    unsigned int k = 50;
    double threshold = 2.0;
    unsigned int blocks{};
    unsigned int blockSize{};
};

struct filter {
    unsigned int k;

    filter(unsigned int k) : k(k) {}

    __host__ __device__ unsigned int operator()(const unsigned int a) const {
        return a < k;
    }
};

template<typename T, bool weighted>
__global__ void computeEuclideanParallel(const T* pointsRow, const T* pointsCol, const T* weights,
                                         unsigned int cardinality, unsigned int dimensions, unsigned int chunkSize,
                                         unsigned int offset, unsigned int size, double threshold,
                                         unsigned int* neighbors);

template<typename T>
void processDataset(Dataset<T>* dataset, const Args& args);

int main(int argc, char** argv) {
    try {
        fmt::print("{}\n", "GPU Naive outlier detection");

        Args args;

        cudaDeviceGetAttribute(&args.multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&args.maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        args.blocks = args.multiprocessorCount * 16;
        args.blockSize = args.maxThreadsPerBlock / 2;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("i,input", "Input dataset path", cxxopts::value<std::string>(args.input))
                ("o,output", "Output result path", cxxopts::value<std::string>(args.output))
                ("p,precision", "Precision type", cxxopts::value<std::string>(args.precision))
                // Outlier detection related arguments
                ("w,window", "Window size (must be multiple of slide, default: 10000)",
                 cxxopts::value<unsigned int>(args.windowSize))
                ("s,slide", "Slide size (default: 500)", cxxopts::value<unsigned int>(args.slideSize))
                ("c,cardinality", "Dataset cardinality", cxxopts::value<unsigned int>(args.cardinality))
                ("d,dimensions", "Point dimensionality", cxxopts::value<unsigned int>(args.dimensions))
                ("k", "Minimum number of neighbors to consider a point inlier", cxxopts::value<unsigned int>(args.k))
                ("t,threshold", "Threshold value (default: 2.0)", cxxopts::value<double>(args.threshold))
                ("weights", "Weight file (the number of weights must be equal to the number of dimensions)",
                 cxxopts::value<std::string>(args.weightsFile))
                // GPU related arguments
                ("chunk", "Chunk size (points assigned per blocked)", cxxopts::value<unsigned int>(args.chunkSize))
                ("blocks", "Number of blocks (default: " + std::to_string(args.blocks) + ")",
                 cxxopts::value<unsigned int>(args.blocks))
                ("threads", "Threads per block (default: " + std::to_string(args.blockSize) + ")",
                 cxxopts::value<unsigned int>(args.blockSize))
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

        if (args.windowSize % args.slideSize != 0) {
            fmt::print("{}\n", "Error, window size must be multiple of slide size! Exiting...");
            return 1;
        }

        if (result.count("weights")) {
            args.hasWeights = true;
        }

        if (args.precision == "float") {
            processDataset<float>(new Dataset<float>(args.input, args.cardinality, args.dimensions), args);
        } else {
            processDataset<double>(new Dataset<double>(args.input, args.cardinality, args.dimensions), args);
        }

        fmt::print("Finished!\n");

    } catch (const cxxopts::OptionException& e) {
        fmt::print("{}\n", e.what());
        return 1;
    }
    return 0;
}

// taken from https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
template<typename T>
__device__ T* shared_memory_proxy() {
    // do we need an __align__() here? I don't think so...
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}


template<typename T, bool weighted>
__global__ void computeEuclideanParallel(const T* pointsRow, const T* pointsCol, const T* weights,
                                         unsigned int cardinality, unsigned int dimensions, unsigned int chunkSize,
                                         unsigned int offset, unsigned int size, double threshold,
                                         unsigned int* neighbors) {
    auto smem = shared_memory_proxy<T>();
    T* sharedPoints = smem;
    T* result = (T*) &sharedPoints[chunkSize * dimensions] + threadIdx.x * chunkSize;

    unsigned int bx = blockIdx.x;

    while ((bx * chunkSize) < size) {
        unsigned int tx = threadIdx.x;
        // load points to shared memory
        for (int i = 0; i < chunkSize; i++)
            sharedPoints[(i * dimensions) + tx] = pointsRow[(dimensions * offset) +
                                                            (((bx * chunkSize) + i) * dimensions) + tx];
        __syncthreads();

        while (tx < size) {
            for (int i = 0; i < chunkSize; i++)
                result[i] = 0.0f;
            for (int i = 0; i < dimensions; i++) {
                T tmp = pointsCol[(cardinality * i + offset) + tx];
                for (int j = 0; j < chunkSize; j++) {
                    T res = tmp - sharedPoints[i + (j * dimensions)];
                    if (weighted) {
                        result[j] += (res * res) * weights[i];
                    } else {
                        result[j] += res * res;
                    }
                }
            }
            for (int i = 0; i < chunkSize; i++) {
//                distances[((i + (bx * chunkSize)) * cardinality) + tx] = result[i];
                if (sqrt(result[i]) <= threshold) {
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

template<typename T>
void processDataset(Dataset<T>* dataset, const Args& args) {

    fmt::print(
            "┌{0:─^{1}}┐\n"
            "│{3: ^{2}}|{4: ^{2}}│\n"
            "│{5: ^{2}}|{6: ^{2}}│\n"
            "│{7: ^{2}}|{8: ^{2}}│\n"
            "│{9: ^{2}}|{10: ^{2}}│\n"
            "└{11:─^{1}}┘\n", "Query", 51, 25,
            "Window size (w)", args.windowSize,
            "Slide size (s)", args.slideSize,
            "Min. neighbors (k)", args.k,
            "Threshold", args.threshold, ""
    );

    fmt::print(
            "┌{0:─^{1}}┐\n"
            "│{3: ^{2}}|{4: ^{2}}│\n"
            "│{5: ^{2}}|{6: ^{2}}│\n"
            "│{7: ^{2}}|{8: ^{2}}│\n"
            "└{9:─^{1}}┘\n", "Dataset characteristics", 51, 25,
            "Cardinality", dataset->cardinality,
            "Dimensions", dataset->dimensions,
            "Total elements", dataset->cardinality * dataset->dimensions, ""
    );

    DeviceTimer deviceTimer;

    T* devicePointsRow;
    T* devicePointsCol;
    T* deviceWeights;
    unsigned int* deviceNeighbors;
    unsigned int* deviceOutliers;

    EventPair* devMemAlloc = deviceTimer.add("Device memory allocation");
    errorCheck(cudaMalloc((void**) &devicePointsRow, sizeof(T) * dataset->cardinality * dataset->dimensions))
    errorCheck(cudaMalloc((void**) &devicePointsCol, sizeof(T) * dataset->cardinality * dataset->dimensions))
    if (args.hasWeights) {
        errorCheck(cudaMalloc((void**) &deviceWeights, sizeof(T) * dataset->dimensions))
    }
    errorCheck(cudaMalloc((void**) &deviceNeighbors, sizeof(unsigned int) * args.windowSize))
    errorCheck(cudaMalloc((void**) &deviceOutliers, sizeof(unsigned int) * args.windowSize))
    DeviceTimer::finish(devMemAlloc);

    EventPair* dataTransfer = deviceTimer.add("Transfer to device");
    errorCheck(
            cudaMemcpy(devicePointsRow, dataset->pointsRow, sizeof(T) * dataset->cardinality * dataset->dimensions,
                       cudaMemcpyHostToDevice))
    errorCheck(
            cudaMemcpy(devicePointsCol, dataset->pointsCol, sizeof(T) * dataset->cardinality * dataset->dimensions,
                       cudaMemcpyHostToDevice))
    if (args.hasWeights) {
        errorCheck(cudaMemcpy(deviceWeights, dataset->weights, sizeof(T) * dataset->dimensions,
                              cudaMemcpyHostToDevice))
    }
    DeviceTimer::finish(dataTransfer);

    cudaFuncSetCacheConfig(computeEuclideanParallel<float, true>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(computeEuclideanParallel<float, false>, cudaFuncCachePreferL1);

    cudaFuncSetCacheConfig(computeEuclideanParallel<double, true>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(computeEuclideanParallel<double, false>, cudaFuncCachePreferL1);

    unsigned int currentSize = 0;
    unsigned int currentStart = 0;

    std::vector<unsigned int> counts;
    std::vector<unsigned int> outliers;

    for (unsigned int i = 0;
         i < (args.cardinality / args.slideSize) + (args.cardinality % args.slideSize != 0 ? 1 : 0); ++i) {

        if (currentSize < args.windowSize) {
            currentSize += args.slideSize;
        } else {
            currentStart++;
        }

        EventPair* clearMem = deviceTimer.add("Clear neighbors");
        errorCheck(cudaMemset(deviceNeighbors, 0, sizeof(unsigned int) * currentSize))
        DeviceTimer::finish(clearMem);

        EventPair* calc = deviceTimer.add("Kernel");

        unsigned int offset = currentStart * args.slideSize;
        unsigned int currentEnd = offset + currentSize;
        unsigned int sharedMem = 2 * args.chunkSize * dataset->dimensions * sizeof(T);

        if (currentEnd > args.cardinality) { // last iteration
            currentSize = currentSize - (currentEnd - args.cardinality);
        }

        if (args.hasWeights) {
            computeEuclideanParallel<T, true><<<currentSize / args.chunkSize, args.dimensions, sharedMem>>>(
                    devicePointsRow,
                    devicePointsCol,
                    deviceWeights,
                    dataset->cardinality,
                    dataset->dimensions,
                    args.chunkSize,
                    offset,
                    currentSize,
                    args.threshold,
                    deviceNeighbors);
        } else {
            computeEuclideanParallel<T, false><<<currentSize / args.chunkSize, args.dimensions, sharedMem>>>(
                    devicePointsRow,
                    devicePointsCol,
                    deviceWeights,
                    dataset->cardinality,
                    dataset->dimensions,
                    args.chunkSize,
                    offset,
                    currentSize,
                    args.threshold,
                    deviceNeighbors);
        }

        DeviceTimer::finish(calc);

        EventPair* extractResults = deviceTimer.add("Extract results");
        unsigned int deviceRes = thrust::transform_reduce(thrust::device, deviceNeighbors,
                                                          deviceNeighbors + currentSize, filter(args.k), 0,
                                                          thrust::plus<unsigned int>());

        thrust::copy_if(thrust::make_counting_iterator<unsigned int>((currentStart * args.slideSize) + 1),
                        thrust::make_counting_iterator<unsigned int>((currentStart * args.slideSize) + currentSize + 1),
                        thrust::device_ptr<unsigned int>(deviceNeighbors),
                        thrust::device_ptr<unsigned int>(deviceOutliers),
                        filter(args.k));

        std::vector<unsigned int> tmp(deviceRes);
        errorCheck(cudaMemcpy(&tmp[0], deviceOutliers, sizeof(unsigned int) * deviceRes, cudaMemcpyDeviceToHost))
        DeviceTimer::finish(extractResults);

        outliers.insert(outliers.end(), tmp.begin(), tmp.end());
        counts.push_back(deviceRes);
    }

    EventPair* freeDevMem = deviceTimer.add("Free device memory");
    errorCheck(cudaFree(devicePointsRow))
    errorCheck(cudaFree(devicePointsCol))
    errorCheck(cudaFree(deviceNeighbors))
    DeviceTimer::finish(freeDevMem);

    cudaDeviceSynchronize();
    deviceTimer.print();

    fmt::print("┌{0:─^{1}}┐\n"
               "│{3: ^{2}}|{4: ^{2}}│\n"
               "└{5:─^{1}}┘\n", "Result", 51, 25,
               "Total outliers", outliers.size(), ""
    );

    fmt::print("Writing outliers to {}\n", std::string("outliers_").append(args.output));
    writeOutliersResult(outliers, std::string("outliers_").append(args.output));

    fmt::print("Writing counts to {}\n", std::string("counts_").append(args.output));
    writeCountsResult(counts, std::string("counts_").append(args.output));
}

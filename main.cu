#include <iostream>
#include <cxxopts.hpp>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include "io.hpp"
#include "device_timer.cuh"
#include "helpers.cuh"
#include <fmt/ranges.h>

struct Args {
    int multiprocessorCount{};
    int maxThreadsPerBlock{};
    std::string input;
    std::string output;
    std::string weightsFile;
    std::string groundTruthFile;
    std::string precision = "float";
    bool skipHeader = false;
    bool hasWeights = false;
    unsigned int cardinality{};
    unsigned int dimensions{};
    unsigned int chunkSize = 10;
    unsigned int windowSize = 10000;
    unsigned int slideSize = 500;
    unsigned int k = 50;
    bool isFixed = false;
    unsigned int fixedPoints = 0;
    std::vector<double> radius;
    unsigned int blocks{};
    unsigned int blockSize{};
};

template<typename T>
struct DeviceData {
    // store dataset in device memory twice, both in row-major and col-major format
    T* pointsRow;
    T* pointsCol;
    T* weights;
    unsigned int* neighbors;
    unsigned int* outliers;
};

struct Result {
    std::vector<unsigned int> counts;
    std::vector<unsigned int> outliers;
};

struct filter {
    unsigned int k;

    filter(unsigned int k) : k(k) {}

    __host__ __device__ unsigned int operator()(const unsigned int a) const {
        return a < k;
    }
};

template<typename T, bool weighted>
__global__ void computeEuclideanSlidingWindow(DeviceData<T> data,
                                         unsigned int cardinality, unsigned int dimensions, unsigned int chunkSize,
                                         unsigned int offset, unsigned int size, double radius);
template<typename T, bool weighted>
__global__ void computeEuclideanFixed(DeviceData<T> data,
                                         unsigned int cardinality, unsigned int dimensions, unsigned int chunkSize,
                                         unsigned int fixedPoints, unsigned int offset, unsigned int size, double radius);

template<typename T>
void processDataset(Dataset<T>* dataset, const Args& args, std::set<unsigned int>& groundTruth);

int main(int argc, char** argv) {
    try {
        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{2: ^{1}}│\n"
                "└{0:─^{1}}┘\n", "", 51, "GPU Naive outlier detection");

        Args args;

        cudaDeviceGetAttribute(&args.multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&args.maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("i,input", "Input dataset path", cxxopts::value<std::string>(args.input))
                ("o,output", "Output result path", cxxopts::value<std::string>(args.output))
                ("p,precision", "Precision type", cxxopts::value<std::string>(args.precision))
                ("skip-header", "Skip input dataset header", cxxopts::value<bool>(args.skipHeader))
                // Outlier detection related arguments
                ("w,window", "Window size (must be multiple of slide, default: 10000)",
                 cxxopts::value<unsigned int>(args.windowSize))
                ("f,fixed", "First N points are used as distance baseline", cxxopts::value<unsigned int>(args.fixedPoints))
                ("s,slide", "Slide size (default: 500)", cxxopts::value<unsigned int>(args.slideSize))
                ("c,cardinality", "Dataset cardinality", cxxopts::value<unsigned int>(args.cardinality))
                ("d,dimensions", "Point dimensionality", cxxopts::value<unsigned int>(args.dimensions))
                ("k", "Minimum number of neighbors to consider a point inlier", cxxopts::value<unsigned int>(args.k))
                ("r,radius", "Radius value (default: 2.0)", cxxopts::value<std::vector<double>>(args.radius))
                ("weights", "Weight file (the number of weights must be equal to the number of dimensions)",
                 cxxopts::value<std::string>(args.weightsFile))
                ("ground-truth", "Ground truth file (the outlier ids)",
                 cxxopts::value<std::string>(args.groundTruthFile))
                // GPU related arguments
                ("chunk", "Chunk size (points assigned per block)", cxxopts::value<unsigned int>(args.chunkSize))
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

        if (args.radius.empty()) {
            fmt::print("{}\n", "No radius given! Exiting...");
            return 1;
        }

        if (args.fixedPoints > 0) {
            args.isFixed = true;
        }

        if (args.windowSize % args.slideSize != 0 && !args.isFixed) {
            fmt::print("{}\n", "Error, window size must be multiple of slide size! Exiting...");
            return 1;
        }

        if (args.fixedPoints % args.slideSize != 0 && args.isFixed) {
            fmt::print("{}\n", "Error, fixed points must be multiple of slide size! Exiting...");
            return 1;
        }

        if (result.count("weights")) {
            args.hasWeights = true;
        }

        std::set<unsigned int> groundTruth;

        if (result.count("ground-truth")) {
            groundTruth = readGroundTruth(args.groundTruthFile);
        }

        if (args.precision == "float") {
            processDataset<float>(new Dataset<float>(args.input, args.cardinality, args.dimensions, args.skipHeader),
                                  args,
                                  groundTruth);
        } else {
            processDataset<double>(new Dataset<double>(args.input, args.cardinality, args.dimensions, args.skipHeader),
                                   args,
                                   groundTruth);
        }

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
__global__ void computeEuclideanSlidingWindow(DeviceData<T> data,
                                         unsigned int cardinality, unsigned int dimensions, unsigned int chunkSize,
                                         unsigned int offset, unsigned int size, double radius) {
    auto smem = shared_memory_proxy<T>();
    T* sharedPoints = smem;
    T* result = (T*) &sharedPoints[chunkSize * dimensions] + threadIdx.x * chunkSize;

    unsigned int bx = blockIdx.x;

    while ((bx * chunkSize) < size) {
        unsigned int tx = threadIdx.x;
        // load points to shared memory
        // each thread is responsible to store a specific dimension to shared memory
        for (int i = 0; i < chunkSize; i++){
            sharedPoints[(i * dimensions) + tx] = data.pointsRow[(dimensions * offset) +
                                                            (((bx * chunkSize) + i) * dimensions) + tx];
        }

        __syncthreads();

        while (tx < size) {
            for (int i = 0; i < chunkSize; i++)
                result[i] = 0.0f;
            for (int i = 0; i < dimensions; i++) {
                T tmp = data.pointsCol[(cardinality * i + offset) + tx];
                for (int j = 0; j < chunkSize; j++) {
                    T res = tmp - sharedPoints[i + (j * dimensions)];
                    if (weighted) {
                        result[j] += (res * res) * data.weights[i];
                    } else {
                        result[j] += res * res;
                    }
                }
            }
            for (int i = 0; i < chunkSize; i++) {
                if (sqrt(result[i]) <= radius) {
                    unsigned int tmp = ((i + (bx * chunkSize)) * size) + tx;
                    unsigned int a = (tmp / size);
                    unsigned int b = (tmp % size);
                    if (a < b) {
                        atomicAdd(data.neighbors + a, 1);
                        atomicAdd(data.neighbors + b, 1);
                    }
                }
            }
            tx += blockDim.x;
        }
        __syncthreads();
        bx += gridDim.x;
    }
}

template<typename T, bool weighted>
__global__ void computeEuclideanFixed(DeviceData<T> data,
                                         unsigned int cardinality, unsigned int dimensions, unsigned int chunkSize,
                                         unsigned int fixedPoints, unsigned int offset, unsigned int size, double radius) {
    auto smem = shared_memory_proxy<T>();
    T* sharedPoints = smem;
    T* result = (T*) &sharedPoints[chunkSize * dimensions] + threadIdx.x * chunkSize;

    unsigned int bx = blockIdx.x;

    while ((bx * chunkSize) < size) {
        unsigned int tx = threadIdx.x;
        // load points to shared memory
        for (int i = 0; i < chunkSize; i++){
            if ((bx * chunkSize) + i < fixedPoints) { // then load fixed points to shared memory
                sharedPoints[(i * dimensions) + tx] = data.pointsRow[(((bx * chunkSize) + i) * dimensions) + tx];
            } else { // load slide points to shared memory
                sharedPoints[(i * dimensions) + tx] = data.pointsRow[(dimensions * offset) +
                                                                     (((bx * chunkSize) + i - fixedPoints) * dimensions) + tx];
            }
        }

        __syncthreads();

        while (tx < size) {
            for (int i = 0; i < chunkSize; i++)
                result[i] = 0.0f;
            for (int i = 0; i < dimensions; i++) {
                T tmp;
                if (tx < fixedPoints) { // fetch fixed point dimension
                    tmp = data.pointsCol[(cardinality * i) + tx];
                } else { // fetch slide point dimension
                    tmp = data.pointsCol[(cardinality * i + offset) + tx - fixedPoints];
                }
                for (int j = 0; j < chunkSize; j++) {
                    T res = tmp - sharedPoints[i + (j * dimensions)];
                    if (weighted) {
                        result[j] += (res * res) * data.weights[i];
                    } else {
                        result[j] += res * res;
                    }
                }
            }
            for (int i = 0; i < chunkSize; i++) {
                if (sqrt(result[i]) <= radius) {
                    unsigned int tmp = ((i + (bx * chunkSize)) * size) + tx;
                    unsigned int a = (tmp / size);
                    unsigned int b = (tmp % size);
                    if (a < b && (a < fixedPoints || b < fixedPoints)) {
                        atomicAdd(data.neighbors + a, 1);
                        atomicAdd(data.neighbors + b, 1);
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
void allocateDeviceMemory(DeviceData<T>& deviceData, Dataset<T>* dataset, const Args& args)
{
    errorCheck(cudaMalloc((void**) &(deviceData.pointsRow), sizeof(T) * dataset->cardinality * dataset->dimensions))
    errorCheck(cudaMalloc((void**) &(deviceData.pointsCol), sizeof(T) * dataset->cardinality * dataset->dimensions))
    if (args.hasWeights) {
        errorCheck(cudaMalloc((void**) &(deviceData.weights), sizeof(T) * dataset->dimensions))
    }

    if (args.isFixed) {
        errorCheck(cudaMalloc((void**) &(deviceData.neighbors), sizeof(unsigned int) * (args.fixedPoints + args.slideSize)))
        errorCheck(cudaMalloc((void**) &(deviceData.outliers), sizeof(unsigned int) * (args.fixedPoints + args.slideSize)))
    } else { // sliding window
        errorCheck(cudaMalloc((void**) &(deviceData.neighbors), sizeof(unsigned int) * args.windowSize))
        errorCheck(cudaMalloc((void**) &(deviceData.outliers), sizeof(unsigned int) * args.windowSize))
    }
}
template<typename T>

void transferToDeviceMemory(DeviceData<T>& deviceData, Dataset<T>* dataset, const Args& args)
{
    errorCheck(
            cudaMemcpy(deviceData.pointsRow, dataset->pointsRow, sizeof(T) * dataset->cardinality * dataset->dimensions,
                       cudaMemcpyHostToDevice))
    errorCheck(
            cudaMemcpy(deviceData.pointsCol, dataset->pointsCol, sizeof(T) * dataset->cardinality * dataset->dimensions,
                       cudaMemcpyHostToDevice))
    if (args.hasWeights) {
        errorCheck(cudaMemcpy(deviceData.weights, dataset->weights, sizeof(T) * dataset->dimensions,
                              cudaMemcpyHostToDevice))
    }
}

void clearNeighbors(unsigned int* deviceNeighbors, unsigned int size)
{
    errorCheck(cudaMemset(deviceNeighbors, 0, sizeof(unsigned int) * size))
}

template<typename T>
void freeDeviceMemory(DeviceData<T>& deviceData)
{
    errorCheck(cudaFree(deviceData.pointsRow))
    errorCheck(cudaFree(deviceData.pointsCol))
    errorCheck(cudaFree(deviceData.neighbors))
    errorCheck(cudaFree(deviceData.outliers))
}

template<typename T>
std::vector<unsigned int> extractResults(DeviceData<T>& deviceData, unsigned int startID, unsigned int endID,
                                         unsigned int offset, unsigned int size, unsigned int k)
{
    unsigned int deviceRes = thrust::transform_reduce(thrust::device,
                                                      thrust::device_ptr<unsigned int>(deviceData.neighbors + offset),
                                                      thrust::device_ptr<unsigned int>(deviceData.neighbors + size),
                                                      filter(k),
                                                      0,
                                                      thrust::plus<unsigned int>());

    thrust::copy_if(thrust::device,
                    thrust::make_counting_iterator<unsigned int>(startID),
                    thrust::make_counting_iterator<unsigned int>(endID),
                    thrust::device_ptr<unsigned int>(deviceData.neighbors + offset),
                    thrust::device_ptr<unsigned int>(deviceData.outliers),
                    filter(k));

    std::vector<unsigned int> tmp(deviceRes);
    errorCheck(cudaMemcpy(&tmp[0], deviceData.outliers, sizeof(unsigned int) * deviceRes, cudaMemcpyDeviceToHost))

    return tmp;
}

template<typename T>
Result executeFixed(Dataset<T>* dataset, DeviceData<T>& deviceData, const Args& args, DeviceTimer& deviceTimer, double radius)
{
    unsigned int currentStart = args.fixedPoints;
    unsigned int size = args.fixedPoints + args.slideSize;

    Result result;
    unsigned int remainingPoints = args.cardinality - args.fixedPoints;
    unsigned int numOfIterations = (remainingPoints / args.slideSize) + (remainingPoints % args.slideSize != 0 ? 1 : 0);

    for (unsigned int i = 0; i < numOfIterations; ++i) {
        EventPair* clearMem = deviceTimer.add("Clear neighbors");
        clearNeighbors(deviceData.neighbors, size);
        DeviceTimer::finish(clearMem);

        EventPair* calc = deviceTimer.add("Kernel");

        unsigned int currentEnd = currentStart + args.slideSize;
        unsigned int sharedMem = 2 * std::min(args.chunkSize, args.slideSize) * dataset->dimensions * sizeof(T);

        if (currentEnd > args.cardinality) { // last iteration
            size = size - (currentEnd - args.cardinality);
        }

        if (args.hasWeights) {
            computeEuclideanFixed<T, true><<<std::max(args.chunkSize, size) / args.chunkSize, args.dimensions, sharedMem>>>(
                    deviceData,
                    dataset->cardinality,
                    dataset->dimensions,
                    std::min(args.chunkSize, size),
                    args.fixedPoints,
                    currentStart,
                    size,
                    radius);
        } else {
            computeEuclideanFixed<T, false><<<std::max(args.chunkSize, size) / args.chunkSize, args.dimensions, sharedMem>>>(
                    deviceData,
                    dataset->cardinality,
                    dataset->dimensions,
                    std::min(args.chunkSize, size),
                    args.fixedPoints,
                    currentStart,
                    size,
                    radius);
            cudaDeviceSynchronize();
        }
        DeviceTimer::finish(calc);

        EventPair* extractRes = deviceTimer.add("Extract results");
        std::vector<unsigned int> tmp = extractResults<T>(deviceData, currentStart, currentStart + args.slideSize,
                                                          args.fixedPoints, size, args.k);
        DeviceTimer::finish(extractRes);

        result.outliers.insert(result.outliers.end(), tmp.begin(), tmp.end());
        result.counts.push_back(tmp.size());

        currentStart += args.slideSize;
    }

    std::set<unsigned int> s;
    for (unsigned int& outlier : result.outliers) s.insert(outlier);
    result.outliers.assign(s.begin(), s.end());

    return result;
}


template<typename T>
Result executeSlidingWindow(Dataset<T>* dataset, DeviceData<T>& deviceData, const Args& args, DeviceTimer& deviceTimer, double radius)
{
    unsigned int currentSize = 0;
    unsigned int currentStart = 0;

    Result result;
    unsigned int numOfIterations = (args.cardinality / args.slideSize) + (args.cardinality % args.slideSize != 0 ? 1 : 0);

    for (unsigned int i = 0; i < numOfIterations; ++i) {

        if (currentSize < args.windowSize) {
            currentSize += args.slideSize;
        } else {
            currentStart++;
        }

        EventPair* clearMem = deviceTimer.add("Clear neighbors");
        clearNeighbors(deviceData.neighbors, currentSize);
        DeviceTimer::finish(clearMem);

        EventPair* calc = deviceTimer.add("Kernel");

        unsigned int offset = currentStart * args.slideSize;
        unsigned int currentEnd = offset + currentSize;
        unsigned int sharedMem = 2 * std::min(args.chunkSize, args.slideSize) * dataset->dimensions * sizeof(T);

        if (currentEnd > args.cardinality) { // last iteration
            currentSize = currentSize - (currentEnd - args.cardinality);
        }

        if (args.hasWeights) {
            computeEuclideanSlidingWindow<T, true><<< std::max(currentSize, args.chunkSize) / args.chunkSize, args.dimensions, sharedMem>>>(
                    deviceData,
                    dataset->cardinality,
                    dataset->dimensions,
                    std::min(args.chunkSize, currentSize),
                    offset,
                    currentSize,
                    radius);
        } else {
            computeEuclideanSlidingWindow<T, false><<<std::max(currentSize, args.chunkSize) / args.chunkSize, args.dimensions, sharedMem>>>(
                    deviceData,
                    dataset->cardinality,
                    dataset->dimensions,
                    std::min(args.chunkSize, currentSize),
                    offset,
                    currentSize,
                    radius);
        }
        DeviceTimer::finish(calc);

        EventPair* extractRes = deviceTimer.add("Extract results");
        std::vector<unsigned int> tmp = extractResults<T>(deviceData,
                                                          currentStart * args.slideSize,
                                                          (currentStart * args.slideSize) + currentSize,
                                                          0,
                                                          currentSize,
                                                          args.k);
        DeviceTimer::finish(extractRes);

        result.outliers.insert(result.outliers.end(), tmp.begin(), tmp.end());
        result.counts.push_back(tmp.size());
    }

    std::set<unsigned int> s;
    for (unsigned int& outlier : result.outliers) s.insert(outlier);
    result.outliers.assign(s.begin(), s.end());

    return result;
}

void calculateScore(Result& result, std::set<unsigned int>& groundTruth)
{
    std::vector<unsigned int>::iterator it;
    std::vector<unsigned int> v(std::min(groundTruth.size(), result.outliers.size()));
    it = std::set_intersection(groundTruth.begin(), groundTruth.end(), result.outliers.begin(), result.outliers.end(),
                               v.begin());
    v.resize(it - v.begin());
    unsigned int TP = v.size();
    unsigned int FP = result.outliers.size() - TP;
    unsigned int FN = groundTruth.size() - TP;
    double precision = ((double) TP / (double) (TP + FP)) * 100.0;
    double recall = ((double) TP / (double) (TP + FN)) * 100.0;

    fmt::print(
            "┌{0:─^{1}}┐\n"
            "│{3: ^{2}}|{4: ^{2}}│\n"
            "│{5: ^{2}}|{6: ^{2}}│\n"
            "│{7: ^{2}}|{8: ^{2}}│\n"
            "│{9: ^{2}}|{10: ^{2}.2f}│\n"
            "│{11: ^{2}}|{12: ^{2}.2f}│\n"
            "└{13:─^{1}}┘\n", "Evaluation", 51, 25,
            "TP", TP,
            "FP", FP,
            "FN", FN,
            "Precision", precision,
            "Recall", recall,
            ""
    );
}

void writeOutput(const std::string& output, double radius, unsigned int k, Result& result)
{
    fmt::print("Writing outliers to {}.outliers\n", output);
    writeOutliersResult(result.outliers, radius, k, std::string(output).append(".outliers"));

    fmt::print("Writing counts to {}.counts\n", output);
    writeCountsResult(result.counts, radius, k, std::string(output).append(".counts"));
}

template<typename T>
void processDataset(Dataset<T>* dataset, const Args& args, std::set<unsigned int>& groundTruth) {

    // print dataset information
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
    DeviceData<T> deviceData{};

    EventPair* devMemAlloc = deviceTimer.add("Device memory allocation");
    allocateDeviceMemory<T>(deviceData, dataset, args);
    DeviceTimer::finish(devMemAlloc);

    EventPair* dataTransfer = deviceTimer.add("Transfer to device");
    transferToDeviceMemory<T>(deviceData, dataset, args);
    DeviceTimer::finish(dataTransfer);

    cudaFuncSetCacheConfig(computeEuclideanSlidingWindow<float, true>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(computeEuclideanSlidingWindow<float, false>, cudaFuncCachePreferL1);

    cudaFuncSetCacheConfig(computeEuclideanSlidingWindow<double, true>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(computeEuclideanSlidingWindow<double, false>, cudaFuncCachePreferL1);

    cudaFuncSetCacheConfig(computeEuclideanFixed<float, true>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(computeEuclideanFixed<float, false>, cudaFuncCachePreferL1);

    cudaFuncSetCacheConfig(computeEuclideanFixed<double, true>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(computeEuclideanFixed<double, false>, cudaFuncCachePreferL1);

    for (auto& r: args.radius) {
        Result result;
        if (args.isFixed) {
            fmt::print(
                    "┌{0:─^{1}}┐\n"
                    "│{3: ^{2}}|{4: ^{2}}│\n"
                    "│{5: ^{2}}|{6: ^{2}}│\n"
                    "│{7: ^{2}}|{8: ^{2}}│\n"
                    "│{9: ^{2}}|{10: ^{2}}│\n"
                    "│{11: ^{2}}|{12: ^{2}}│\n"
                    "└{13:─^{1}}┘\n", "Query", 51, 25,
                    "Type", "Fixed",
                    "Fixed points (f)", args.fixedPoints,
                    "Slide size (s)", args.slideSize,
                    "Radius (r)", r,
                    "Min. neighbors (k)", args.k,
                    ""
            );
            result = executeFixed<T>(dataset, deviceData, args, deviceTimer, r);
        } else { // sliding window
            fmt::print(
                    "┌{0:─^{1}}┐\n"
                    "│{3: ^{2}}|{4: ^{2}}│\n"
                    "│{5: ^{2}}|{6: ^{2}}│\n"
                    "│{7: ^{2}}|{8: ^{2}}│\n"
                    "│{9: ^{2}}|{10: ^{2}}│\n"
                    "│{11: ^{2}}|{12: ^{2}}│\n"
                    "└{13:─^{1}}┘\n", "Query", 51, 25,
                    "Type", "Sliding window",
                    "Window size (w)", args.windowSize,
                    "Slide size (s)", args.slideSize,
                    "Radius (r)", r,
                    "Min. neighbors (k)", args.k,
                    ""
            );
            result = executeSlidingWindow<T>(dataset, deviceData, args, deviceTimer, r);
        }

        if (!groundTruth.empty()) {
            calculateScore(result, groundTruth);
        }


        if (!args.output.empty()) {
            writeOutput(args.output, r, args.k, result);
        }


        fmt::print("┌{0:─^{1}}┐\n"
                   "│{3: ^{2}}|{4: ^{2}}│\n"
                   "└{5:─^{1}}┘\n", "Result", 51, 25,
                   "Total outliers", result.outliers.size(), ""
        );
    }

    errorCheck(cudaDeviceSynchronize())


    EventPair* freeDevMem = deviceTimer.add("Free device memory");
    freeDeviceMemory(deviceData);
    DeviceTimer::finish(freeDevMem);


    deviceTimer.print();
}

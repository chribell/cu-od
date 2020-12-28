cmake_minimum_required(VERSION 3.19)
project(cu_distance CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(SM_ARCH 61 CACHE STRING "CUDA SM architecture e.g. 50, 52, 61")

include(FetchContent)

# CXX options library
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        v2.2.1
)
FetchContent_MakeAvailable(cxxopts)
# Adds cxxopts

# Formatting library
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        7.1.2
)
FetchContent_MakeAvailable(fmt)
# Adds fmt::fmt

add_executable(cu_od main.cu)
target_link_libraries(cu_od PRIVATE cxxopts fmt::fmt)
target_include_directories(cu_od PRIVATE include)
set_target_properties(
        cu_od
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(cu_od PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

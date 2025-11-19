cmake_minimum_required(VERSION 3.18)
project(GeneSlime CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcudadevrt")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3")

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

include_directories(${CMAKE_SOURCE_DIR})

add_executable(geneslime
    main.cu
)

set_target_properties(geneslime PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(geneslime
    cudadevrt
)

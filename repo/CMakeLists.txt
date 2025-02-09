cmake_minimum_required(VERSION 3.18)
project(KMeans LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compilation options
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# If there is support for OpenMP Offloading, add the flag
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fopenmp-targets=nvptx64" OPENMP_OFFLOAD_SUPPORTED)
if(OPENMP_OFFLOAD_SUPPORTED)
    set(OFFLOAD_FLAGS "-fopenmp-targets=nvptx64")
endif()

# Sources and Headlines
set(SRC_UTILS kmeans_utils.cpp kmeans_utils.h)

# Create three executable files
add_executable(kmeans kmeans.cpp ${SRC_UTILS})
add_executable(kmeans_openmp kmeans_openmp.cpp ${SRC_UTILS})
add_executable(kmeans_omp_target kmeans_omp_target.cpp ${SRC_UTILS})

# Linking with OpenMP
if(OpenMP_CXX_FOUND)
	target_link_libraries(kmeans_openmp OpenMP::OpenMP_CXX)
	target_link_libraries(kmeans_omp_target OpenMP::OpenMP_CXX)
endif()

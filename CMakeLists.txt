cmake_minimum_required(VERSION 3.10)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_60,code=sm_60
    )


# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    hellocuda
    featureGenerator.cu
    CovarFeature.cpp
    main.cpp
    )

if(MSVC)

include_directories(
    "C:\\opencv\\include"
    .
    )

link_directories(
    "C:\\opencv\\lib"
    )

target_link_libraries(
    hellocuda
    opencv_world344.lib
    )
endif (MSVC)


if(UNIX)

include_directories(
    /usr/local/include
    )

link_directories(
    /usr/local/lib
    )
target_link_libraries(
    hellocuda
    -lopencv_core
    -lopencv_imgproc
    -lopencv_highgui
    -lopencv_videoio
    -lopencv_imgcodecs
    -lopencv_video
    )
endif (UNIX)

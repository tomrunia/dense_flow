cmake_minimum_required(VERSION 2.8)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

project( denseFlow )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( denseImage denseImage.cpp )
target_link_libraries( denseImage ${OpenCV_LIBS} )

add_executable( denseFlow_gpu denseFlow_gpu.cpp )
target_link_libraries( denseFlow_gpu ${OpenCV_LIBS} )

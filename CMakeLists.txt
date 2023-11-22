cmake_minimum_required(VERSION 3.12)

project(PeopleCounter)
set(CMAKE_BUILD_TYPE Debug)

#find_package(CUDA 12.1 REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

find_package(OpenCV REQUIRED)

set(TENSORRT_ROOT /usr/src/tensorrt/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER} ${TENSORRT_LIBRARY_PLUGIN})

set(YAML_INCLUDE ./peopleReId/depends/yaml-cpp/include)
set(YAML_LIB_DIR ./peopleReId/depends/yaml-cpp/libs)
link_directories(${YAML_LIB_DIR})

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${COMMON_INCLUDE}
        ${YAML_INCLUDE} )
add_executable(PeopleCounter main.cpp peopleTracker/MultipleObjectTracker.cpp peopleTracker/ObjectTracker.cpp
        peopleDetector/PeopleDetector.cpp common/Constants.cpp common/Constants.hpp peopleReId/common.cpp peopleReId/fast-reid.cpp
        peopleReId/feature.cpp peopleReId/model.cpp)
include_directories(common)
target_link_libraries(PeopleCounter ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} yaml-cpp)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(common)

add_executable(modelComparison modelComparison.cpp peopleTracker/MultipleObjectTracker.cpp peopleTracker/ObjectTracker.cpp
        peopleDetector/PeopleDetector.cpp common/Constants.cpp common/Constants.hpp)
target_link_libraries(modelComparison ${OpenCV_LIBS})

add_executable(featuresComparison fearuresComparison.cpp common/Constants.cpp common/Constants.hpp peopleReId/common.cpp peopleReId/fast-reid.cpp
        peopleReId/feature.cpp peopleReId/model.cpp)
target_link_libraries(featuresComparison ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} yaml-cpp)
# cmake needs this line
cmake_minimum_required(VERSION 3.1)
# cmake_minimum_required(VERSION 2.8)

# Enable C++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(edge_tpu_deeplabv3_benchmark)

# SET (CMAKE_VERBOSE_MAKEFILE 1)

# Compile options
set(CMAKE_C_FLAGS "-Wall -pthread ")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11  -lstdc++")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

set(CMAKE_BUILD_TYPE release)
# set(CMAKE_BUILD_TYPE debug)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(edge_tpu_deeplabv3_benchmark deeplab_benchmark.cpp)

# Link your application with OpenCV libraries
target_link_libraries(edge_tpu_deeplabv3_benchmark LINK_PRIVATE ${OpenCV_LIBS})

EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Architecture: ${ARCHITECTURE}")

# Link your application with Tensorflow Lite
if( ${ARCHITECTURE} STREQUAL "armv7l" )
    target_link_libraries(edge_tpu_deeplabv3_benchmark LINK_PRIVATE ${CMAKE_SOURCE_DIR}/../lib/libtensorflow-lite_arm32.a)
    target_link_libraries(edge_tpu_deeplabv3_benchmark LINK_PRIVATE ${CMAKE_SOURCE_DIR}/../edgetpu-native/libedgetpu/libedgetpu_arm32.so)
else()
    target_link_libraries(edge_tpu_deeplabv3_benchmark LINK_PRIVATE ${CMAKE_SOURCE_DIR}/../lib/libtensorflow-lite_arm64.a)
    target_link_libraries(edge_tpu_deeplabv3_benchmark LINK_PRIVATE ${CMAKE_SOURCE_DIR}/../edgetpu-native/libedgetpu/libedgetpu_arm64.so)
endif()


# target_link_libraries(edge_tpu_deeplabv3_benchmark Threads::Threads)
find_library(LIBRT rt)
if(LIBRT)
  target_link_libraries(edge_tpu_deeplabv3_benchmark LINK_PRIVATE ${LIBRT})
endif()

find_library(LIBDL dl)
if(LIBDL)
  target_link_libraries(edge_tpu_deeplabv3_benchmark LINK_PRIVATE ${LIBDL})
endif()

target_include_directories(edge_tpu_deeplabv3_benchmark PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(edge_tpu_deeplabv3_benchmark PUBLIC ${CMAKE_SOURCE_DIR}/../edgetpu-native/libedgetpu/)
target_include_directories(edge_tpu_deeplabv3_benchmark PUBLIC ${CMAKE_SOURCE_DIR}/../edgetpu-native/tensorflow/)
target_include_directories(edge_tpu_deeplabv3_benchmark PUBLIC ${CMAKE_SOURCE_DIR}/../edgetpu-native/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)

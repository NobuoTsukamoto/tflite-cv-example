# cmake needs this line
cmake_minimum_required(VERSION 3.1)
# cmake_minimum_required(VERSION 2.8)

# Enable C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(ffnet_tflite_gpu)

SET (CMAKE_VERBOSE_MAKEFILE 1)

# Compile options
set(CMAKE_C_FLAGS "-Wall -pthread ")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

set(CMAKE_BUILD_TYPE release)
# set(CMAKE_BUILD_TYPE debug)

set(TFLITE_ENABLE_GPU ON)

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
add_executable(ffnet_tflite_gpu ffnet_tflite_gpu.cpp utils.cpp )

# Link your application with OpenCV libraries
target_link_libraries(ffnet_tflite_gpu LINK_PRIVATE ${OpenCV_LIBS})

EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Architecture: ${ARCHITECTURE}")

find_library(LIBRT rt)
if(LIBRT)
  target_link_libraries(ffnet_tflite_gpu LINK_PRIVATE ${LIBRT})
endif()

find_library(LIBDL dl)
if(LIBDL)
  target_link_libraries(ffnet_tflite_gpu LINK_PRIVATE ${LIBDL})
endif()

add_subdirectory(
  "${CMAKE_SOURCE_DIR}/../tensorflow/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)


target_link_libraries(ffnet_tflite_gpu LINK_PRIVATE tensorflow-lite)

target_include_directories(ffnet_tflite_gpu PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(ffnet_tflite_gpu PUBLIC ${CMAKE_SOURCE_DIR}/../tensorflow/)
target_include_directories(ffnet_tflite_gpu PUBLIC ${CMAKE_SOURCE_DIR}/../tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)

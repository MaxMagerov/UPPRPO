cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 20)
project(CircularBuffer)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(CircularBuffer source/main.cpp include/CircularBuffer.h)

add_library(CircularBuffer_Lib STATIC include/CircularBuffer.h)
set_target_properties(CircularBuffer_Lib PROPERTIES LINKER_LANGUAGE CXX)
add_subdirectory(test)

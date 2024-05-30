cmake_minimum_required(VERSION 3.27)
set(CMAKE_CXX_STANDARD 20)
project(Tests)

add_subdirectory(googletest)

add_executable(Tests
        ChangeTest.cpp
        ConstructorTest.cpp
        EquityTest.cpp
        GetterTest.cpp
        LinearizeTest.cpp
        RotateTest.cpp
        TestUtils.h)

target_link_libraries(Tests CircularBuffer_Lib)

target_link_libraries(Tests gtest gtest_main)
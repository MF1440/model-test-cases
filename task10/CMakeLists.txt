cmake_minimum_required(VERSION 3.14)
project(task10 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_executable(task10 main.cpp)
target_link_libraries(task10 PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_include_directories(task10 PRIVATE include)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(task10 PRIVATE
        -O3 -march=native -flto=auto
        -ffunction-sections -fdata-sections
        -fno-threadsafe-statics
        -fdelete-null-pointer-checks
    )
    target_link_options(task10 PRIVATE -flto=auto -Wl,--gc-sections)
endif()

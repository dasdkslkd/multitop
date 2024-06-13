cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(multitop)
set(CMAKE_BUILD_TYPE RELEASE)

# compile options
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

# package
find_package(Torch REQUIRED PATHS D:\\Environment\\libtorch)

# complile files
file(glob recurse sourcefile *.cpp)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mma)

add_executable(${PROJECT_NAME} ${sourcefile})

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_ENVIRONMENT "PATH=D:\\Environment\\libtorch\\lib;D:\\IDE\\VS2022\\VC\\Redist\\MSVC\\14.36.32532\\debug_nonredist\\x64\\Microsoft.VC143.DebugCRT;D:\\IDE\\VS2022\\VC\\Redist\\MSVC\\14.36.32532\\debug_nonredist\\x64\\Microsoft.VC143.DebugOpenMP")

include_directories("${CMAKE_SOURCE_DIR}")

# libtorch
target_link_libraries(test-torch ${TORCH_LIBRARIES})
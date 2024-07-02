cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(multitop)
set(CMAKE_BUILD_TYPE RELEASE)

# compile options
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

# include paths
# set(Eigen3_DIR D:\\Environment\\eigen-3.4.0\\cmake)
include_directories(D:\\Environment\\eigen-3.4.0\\build\\include)
# package
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Torch REQUIRED PATHS D:\\Environment\\libtorch-release)
find_package(OpenMP REQUIRED)


# complile files
file(GLOB_RECURSE sourcefile *.cpp)
file(GLOB_RECURSE buildCopy build/*.cpp)
list(REMOVE_ITEM sourcefile ${buildCopy})
# message(STATUS "${buildCopy}")
# message(STATUS "${sourcefile}")
file(GLOB_RECURSE headerfile *.h)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mma)
include_directories(${CMAKE_SOURCE_DIR}/fem)
include_directories(${CMAKE_SOURCE_DIR}/element)
include_directories(${CMAKE_SOURCE_DIR}/IO)

add_executable(${PROJECT_NAME} ${sourcefile} ${headerfile})

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_ENVIRONMENT "PATH=D:\\Environment\\libtorch\\lib;D:\\IDE\\VS2022\\VC\\Redist\\MSVC\\14.36.32532\\debug_nonredist\\x64\\Microsoft.VC143.DebugCRT;D:\\IDE\\VS2022\\VC\\Redist\\MSVC\\14.36.32532\\debug_nonredist\\x64\\Microsoft.VC143.DebugOpenMP")

# libtorch	
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
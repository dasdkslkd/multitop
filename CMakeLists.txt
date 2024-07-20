cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(multitop)
#set(CMAKE_BUILD_TYPE RELEASE)

# compile options
set(CMAKE_CXX_FLAGS_DEBUG "-O2 -Ob1 -MDd")
#add_compile_options(-O2)
#add_compile_options(-Ob1)
set(CMAKE_CXX_STANDARD 17)

# include paths
# set(Eigen3_DIR D:\\Environment\\eigen-3.4.0\\cmake)
include_directories(D:\\Environment\\eigen-3.4.0\\build\\include)
# package
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
# find_package(CUDA REQUIRED)
# find_library(CUDA_DRIVER_LIBRARY
#              NAMES cuda_driver cuda
#              HINTS ${CUDA_TOOLKIT_ROOT_DIR}
#                    ENV CUDA_PATH
#              PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
# if (NOT CUDA_DRIVER_LIBRARY)
#     # Don't try any stub directories until we have exhausted all other search locations.
#     find_library(CUDA_DRIVER_LIBRARY
#                  NAMES cuda_driver cuda
#                  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
#                        ENV CUDA_PATH
#                  PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs)
# endif ()
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
add_subdirectory(mma-g)
find_package(Torch REQUIRED PATHS D:\\Environment\\libtorch)
find_package(OpenMP REQUIRED)


# compile lib


# complile files
# file(GLOB_RECURSE sourcefile *.cpp)
# file(GLOB_RECURSE buildCopy build/*.cpp)
# list(REMOVE_ITEM sourcefile ${buildCopy})
# # message(STATUS "${buildCopy}")
# # message(STATUS "${sourcefile}")	
# file(GLOB_RECURSE headerfile *.h)
file(GLOB sourcefile *.cpp mma/*.cpp fem/*.cpp element/*.cpp IO/*.cpp)
file(GLOB cudafile culib/*.cu culib/*.cuh fem/*.cu element/*.cu mma/*.cu)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mma)
include_directories(${CMAKE_SOURCE_DIR}/fem)
include_directories(${CMAKE_SOURCE_DIR}/element)
include_directories(${CMAKE_SOURCE_DIR}/IO)
include_directories(${CMAKE_SOURCE_DIR}/culib)

add_library(culib STATIC ${cudafile})
set_target_properties(culib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(culib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(culib PROPERTIES CUDA_ARCHITECTURES "61;75;86")
target_compile_options(culib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--relocatable-device-code=true
      -dc
	--extended-lambda
	--std=c++17
	--expt-relaxed-constexpr
	--compile
	>)
target_include_directories(culib PRIVATE ${CMAKE_SOURCE_DIR}/culib)

# target_link_libraries(culib PRIVATE ${CUDA_LIBRARIES})
# target_link_libraries(culib PRIVATE ${CUDA_DRIVER_LIBRARY})
# target_link_libraries(culib PRIVATE ${CUDA_cusolver_LIBRARY})
target_link_libraries(culib PRIVATE CUDA::cudart)
target_link_libraries(culib PRIVATE CUDA::cuda_driver)
target_link_libraries(culib PRIVATE CUDA::cusolver)

target_link_libraries(culib PRIVATE Eigen3::Eigen)
target_link_libraries(culib PRIVATE ${TORCH_LIBRARIES})

add_executable(${PROJECT_NAME} ${sourcefile})

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_ENVIRONMENT "PATH=D:\\Environment\\Cuda\\bin;D:\\Environment\\libtorch\\lib;D:\\IDE\\VS2022\\VC\\Redist\\MSVC\\14.36.32532\\debug_nonredist\\x64\\Microsoft.VC143.DebugCRT;D:\\IDE\\VS2022\\VC\\Redist\\MSVC\\14.36.32532\\debug_nonredist\\x64\\Microsoft.VC143.DebugOpenMP")

# link	
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} PRIVATE mma)
target_link_libraries(${PROJECT_NAME} PRIVATE culib)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DMMA_WITH_OPENMP)

file(GLOB DLLS lib/*.dll)
add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${DLLS}
                   $<TARGET_FILE_DIR:${PROJECT_NAME}>)
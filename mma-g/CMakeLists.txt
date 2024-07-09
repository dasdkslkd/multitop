cmake_minimum_required(VERSION 3.8)

project(mma)

file(GLOB_RECURSE sourcefile *.cpp)
file(GLOB_RECURSE buildcopy build/*.cpp)
file(GLOB_RECURSE cudafiles *.cu)
file(GLOB_RECURSE cudabuildcopy build/*.cu)
list(REMOVE_ITEM sourcefile ${buildcopy} ${CMAKE_SOURCE_DIR}/mma/mmaOpt.cpp)
list(REMOVE_ITEM cudafiles ${cudabuildcopy})
file(GLOB_RECURSE headerfile *.h)
file(GLOB outerheader cuda_helper/*.h)
list(REMOVE_ITEM headerfile ${outerheader})
file(GLOB_RECURSE cudaheaders *.cuh)

# list (APPEND libfiles mma/mmaOpt.cu culib/lib.cu culib/lib.cuh culib/gpuVector.cuh culib/cudaCommon.cuh mma/mmaOpt.cpp )
#list (APPEND libheaders cudaCommon.h gpuVector.cuh lib.cuh mmaOpt.h)

# add_library(mma STATIC ${libfiles})
add_library(mma STATIC ${cudafiles} ${cudaheaders} ${CMAKE_SOURCE_DIR}/mma/mmaOpt.cpp)

target_compile_options(mma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --std=c++17
>)

set_target_properties(mma PROPERTIES CUDA_ARCHITECTURES "61;75;86")
target_include_directories(mma PRIVATE "${CMAKE_SOURCE_DIR}/cuda_helper")
target_include_directories(mma PRIVATE ${CMAKE_SOURCE_DIR}/culib)
# target_compile_definitions(mma PRIVATE -D_EXPORT_MMAOPT)
target_link_libraries (mma PRIVATE Eigen3::Eigen)
target_link_libraries(mma PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(mma PRIVATE ${CUDA_DRIVER_LIBRARY})
target_link_libraries(mma PRIVATE ${CUDA_cusolver_LIBRARY})
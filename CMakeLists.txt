################################################################################

cmake_minimum_required(VERSION 3.10)
project(mma)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

################################################################################

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

# Use folder in Visual Studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Color output
include(UseColors)

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Generate position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################

# Include Eigen.
include_directories("external/eigen")

add_subdirectory(src/mma)
add_subdirectory(src/sparse_mma)
add_subdirectory(src/gcmma)
add_subdirectory(tests)

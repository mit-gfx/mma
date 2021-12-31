# Main code
add_library(mma_mma MMASolver.cpp)
target_include_directories(mma_mma PUBLIC ..)

# Let's get a little bit paranoid
include(SetWarnings)
target_compile_options(mma_mma PRIVATE ${ALL_WARNINGS})

# Alias target
add_library(mma::mma ALIAS mma_mma)

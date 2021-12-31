# Main code
add_library(mma_sparse_mma SparseMMASolver.cpp)
target_include_directories(mma_sparse_mma PUBLIC ..)

# Let's get a little bit paranoid
include(SetWarnings)
target_compile_options(mma_sparse_mma PRIVATE ${ALL_WARNINGS})

# Alias target
add_library(mma::sparse_mma ALIAS mma_sparse_mma)

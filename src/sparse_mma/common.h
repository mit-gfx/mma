#pragma once

#include "Eigen/Sparse"
#include "Eigen/Dense"

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/time.h>

using real = double;
using SparseMatrix = Eigen::SparseMatrix<real>;
using MatrixXr = Eigen::Matrix<real, -1, -1>;
using VectorXr = Eigen::Matrix<real, -1, 1>;
using RowVectorXr = Eigen::Matrix<real, 1, -1>;
using SparseMatrixElements = std::vector<Eigen::Triplet<real>>;

const std::string GreenHead();
const std::string RedHead();
const std::string YellowHead();
const std::string CyanHead();
const std::string GreenTail();
const std::string RedTail();
const std::string YellowTail();
const std::string CyanTail();

// Timing.
void Tic();
void Toc(const std::string& message);

const SparseMatrixElements FromSparseMatrix(const SparseMatrix& A);
const SparseMatrix ToSparseMatrix(const int row, const int col, const SparseMatrixElements& nonzeros);
const SparseMatrix SparseDiagonalMatrix(const VectorXr& diagonal);
const real SparseMatrixTrace(const SparseMatrix& A);
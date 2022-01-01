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
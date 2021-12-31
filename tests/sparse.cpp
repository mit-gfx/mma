#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "mma/MMASolver.h"
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

// We test mma and sparse_mma on a sparse problem:
// min_x 0.5 * x * A * x + b * x
// s.t.  B * x <= c.
//       lb <= x <= ub.
// Here, both A and B are sparse matrices and A is symmetric positive definite.

using real = double;
using SparseMatrix = Eigen::SparseMatrix<real>;
using MatrixXr = Eigen::Matrix<real, -1, -1>;
using VectorXr = Eigen::Matrix<real, -1, 1>;

// sparsity is between 0 and 1; larger sparsity = more sparse matrix.
const SparseMatrix GenerateSparseMatrix(const int row_num, const int col_num, const real sparsity) {
	const SparseMatrix A = MatrixXr::Random(row_num, col_num).sparseView(1, sparsity);
	return A;
}

struct Problem {
public:
	int n, m;

	SparseMatrix A;
	VectorXr b;
	SparseMatrix B;
	VectorXr c;
	VectorXr lb;
	VectorXr ub;
};

const real ComputeObjective(const Problem& p, const VectorXr& x, VectorXr& loss_grad) {
	const real loss = 0.5 * x.dot(p.A * x) + p.b.dot(x);
	loss_grad = p.A * x + p.b;
	return loss;
}

void ComputeConstraints(const Problem& p, const VectorXr& x, VectorXr& constraints,
	SparseMatrix& constraints_grad) {
	constraints = p.B * x - p.c;
	constraints_grad = p.B;
}

const std::vector<real> VectorXrToStdVector(const VectorXr& v) {
	const int n = static_cast<int>(v.size());
	std::vector<real> vv(n, 0);
	for (int i = 0; i < n; ++i) vv[i] = v(i);
	return vv;
}

const std::string GreenHead() {
    return "\x1b[6;30;92m";
}

const std::string RedHead() {
    return "\x1b[6;30;91m";
}

const std::string YellowHead() {
    return "\x1b[6;30;93m";
}

const std::string CyanHead() {
    return "\x1b[6;30;96m";
}

const std::string GreenTail() {
    return "\x1b[0m";
}

const std::string RedTail() {
    return "\x1b[0m";
}

const std::string YellowTail() {
    return "\x1b[0m";
}

const std::string CyanTail() {
    return "\x1b[0m";
}

// Timing.
static std::stack<timeval> t_begins;

void Tic() {
    timeval t_begin;
    gettimeofday(&t_begin, nullptr);
    t_begins.push(t_begin);
}

void Toc(const std::string& message) {
    timeval t_end;
    gettimeofday(&t_end, nullptr);
    timeval t_begin = t_begins.top();
    const real t_interval = (t_end.tv_sec - t_begin.tv_sec) + (t_end.tv_usec - t_begin.tv_usec) / 1e6;
    std::cout << CyanHead() << "[Timing] " << message << ": " << t_interval << "s"
              << CyanTail() << std::endl;
    t_begins.pop();
}

int main(int argc, char *argv[]) {
	const int seed = std::stoi(std::string(argv[1]));
	srand(seed);
	const int n = std::stoi(std::string(argv[2]));
	const int m = std::stoi(std::string(argv[3]));
	const real sparsity = std::stod(std::string(argv[4]));
	std::cout << "Initializing a problem with random seed " << seed << std::endl;
	std::cout << "The problem has " << n << " variables and " << m << " constraints." << std::endl;
	std::cout << "The set sparsity is " << sparsity << std::endl;

	Problem prob;
	prob.n = n;
	prob.m = m;

	std::cout << "Generating A..." << std::endl;
	const SparseMatrix A_half = GenerateSparseMatrix(n, n, sparsity);
	SparseMatrix I(n, n); I.setIdentity();
	prob.A = A_half.transpose() * A_half + I;
	std::cout << "Sparsity of A: " << 1 - prob.A.nonZeros() * 1.0 / (prob.A.rows() * prob.A.cols()) << std::endl;

	std::cout << "Generating B..." << std::endl;
	prob.B = GenerateSparseMatrix(m, n, sparsity);
	std::cout << "Sparsity of B: " << 1 - prob.B.nonZeros() * 1.0 / (prob.B.rows() * prob.B.cols()) << std::endl;

	// Now generate a feasible solution.
	const VectorXr x_feasible = VectorXr::Random(n);
	prob.lb = VectorXr::Constant(n, x_feasible.minCoeff() - 1.0);
	prob.ub = VectorXr::Constant(n, x_feasible.maxCoeff() + 1.0);

	// Generate c.
	prob.c = VectorXr::Constant(m, (prob.B * x_feasible).maxCoeff() + 1.0);

	// Generate b.
	prob.b = VectorXr::Random(n);

	// Initial guess.
	const VectorXr rand_zero_one = (VectorXr::Random(n).array() + 1) / 2;
	const VectorXr x0 = prob.lb + rand_zero_one.cwiseProduct(prob.ub - prob.lb);

	// MMA.
	std::shared_ptr<MMASolver> mma = std::make_shared<MMASolver>(n, m);
	real movlim = 0.2;
	VectorXr x = x0;
	VectorXr xold = x0;
	VectorXr xmin = prob.lb;
	VectorXr xmax = prob.ub;

	real ch = 1.0;
	int itr = 0;
	while (ch > 0.002 && itr < 100) {
		itr++;

		VectorXr df;
		const real f = ComputeObjective(prob, x, df);
		VectorXr g;
		SparseMatrix dg;
		ComputeConstraints(prob, x, g, dg);

		// Set outer move limits.
		xmax = prob.ub.cwiseMin(VectorXr(x.array() + movlim));
		xmin = prob.lb.cwiseMax(VectorXr(x.array() - movlim));

		// Call the update method.
		Tic();
		mma->Update(x.data(), df.data(), g.data(), dg.toDense().data(), xmin.data(), xmax.data());
		Toc("mma->Update time");

		// Compute infnorm on design change.
		ch = (x - xold).cwiseAbs().maxCoeff();
		xold = x;

		// Print to screen
		printf("it.: %d, obj.: %f, ch.: %f \n",itr, f, ch);
	}

	return 0;
}
#include "common.h"

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

const SparseMatrixElements FromSparseMatrix(const SparseMatrix& A) {
    SparseMatrixElements nonzeros;
    for (int k = 0; k < A.outerSize(); ++k)
        for (SparseMatrix::InnerIterator it(A, k); it; ++it)
            nonzeros.push_back(Eigen::Triplet<real>(it.row(), it.col(), it.value()));
    return nonzeros;
}

const SparseMatrix ToSparseMatrix(const int row, const int col, const SparseMatrixElements& nonzeros) {
    SparseMatrix A(row, col);
    A.setFromTriplets(nonzeros.begin(), nonzeros.end());
    return A;
}

const SparseMatrix SparseDiagonalMatrix(const VectorXr& diagonal) {
    const int n = static_cast<int>(diagonal.size());
    SparseMatrixElements nonzeros;
    for (int i = 0; i < n; ++i) nonzeros.push_back(Eigen::Triplet<real>(i, i, diagonal(i)));
    return ToSparseMatrix(n, n, nonzeros);
}

const real SparseMatrixTrace(const SparseMatrix& A) {
    const int n = static_cast<int>(A.rows());
    real trace = 0;
    for (int i = 0; i < n; ++i) trace += A.coeff(i, i);
    return trace;
}
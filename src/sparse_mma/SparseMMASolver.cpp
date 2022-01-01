////////////////////////////////////////////////////////////////////////////////
// Copyright © 2018 Jérémie Dumas
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////

#include "SparseMMASolver.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// PUBLIC
////////////////////////////////////////////////////////////////////////////////

SparseMMASolver::SparseMMASolver(int nn, int mm, real ai, real ci, real di)
	: n(nn)
	, m(mm)
	, iter(0)
	, xmamieps(1.0e-5)
	//, epsimin(1e-7)
	, epsimin(std::sqrt(n + m) * 1e-9)
	, raa0(0.00001)
	, move(0.5)
	, albefa(0.1)
	, asyminit(0.5) // 0.2;
	, asymdec(0.7) // 0.65;
	, asyminc(1.2) // 1.08;
	, a(VectorXr::Constant(m, ai))
	, c(VectorXr::Constant(m, ci))
	, d(VectorXr::Constant(m, di))
	, y(m)
	, lam(m)
	, mu(m), s(2 * m)
	, low(n)
	, upp(n)
	, alpha(n)
	, beta(n)
	, p0(n)
	, q0(n)
	, b(m)
	, grad(m)
	, pij(n, m)
	, qij(n, m) 
	, hess(m, m)
	, xold1(n)
	, xold2(n)
{ }

void SparseMMASolver::SetAsymptotes(real init, real decrease, real increase) {

	// asymptotes initialization and increase/decrease
	asyminit = init;
	asymdec = decrease;
	asyminc = increase;
}

void SparseMMASolver::Update(VectorXr& xval, const VectorXr& dfdx, const VectorXr& gx,
	const MatrixXr& dgdx, const VectorXr& xmin, const VectorXr& xmax)
{
	// Generate the subproblem
	GenSub(xval, dfdx, gx, dgdx, xmin, xmax);

	// Update xolds
	xold2 = xold1;
	xold1 = xval;
	// std::copy_n(xval, n, xold1.data());

	// Solve the dual with an interior point method
	SolveDIP(xval);

	// Solve the dual with a steepest ascent method
	// SolveDSA(xval);
}

////////////////////////////////////////////////////////////////////////////////
// PRIVATE
////////////////////////////////////////////////////////////////////////////////

void SparseMMASolver::SolveDIP(VectorXr& x) {

	lam = c / 2.0;
	mu.setConstant(1);

	const real tol = epsimin; // 1.0e-9*sqrt(m+n);
	real epsi = 1.0;
	real err = 1.0;
	int loop;

	while (epsi > tol) {

		loop = 0;
		while (err > 0.9 * epsi && loop < 100) {
			loop++;

			// Set up Newton system
			XYZofLAMBDA(x);
			DualGrad(x);
			grad = -grad - lam.cwiseInverse() * epsi;
			DualHess(x);

			// Solve Newton system
			if (m > 1) {
				// s.head(m) = H^-1 * g.
				/*
				const MatrixXr H = hess;
				const VectorXr g = grad;
				Factorize(hess, m);
				Solve(hess, grad, m);
				std::cout << "rel err of H * grad - g" << (H * grad - g).cwiseAbs().maxCoeff() / g.cwiseAbs().maxCoeff() << std::endl;
				// The outputs are very small.
				*/
				grad = hess.colPivHouseholderQr().solve(grad);
				s.head(m) = grad;
			} else if (m > 0) {
				s[0] = grad[0] / hess(0, 0);
			}

			// Get the full search direction
			s.tail(m) = -mu + lam.cwiseInverse() * epsi - s.head(m).cwiseProduct(mu).cwiseQuotient(lam);

			// Perform linesearch and update lam and mu
			DualLineSearch();

			XYZofLAMBDA(x);

			// Compute KKT res
			err = DualResidual(x, epsi);
		}
		epsi = epsi * 0.1;
	}
}

void SparseMMASolver::SolveDSA(VectorXr& x) {
	lam.setConstant(1);

	const real tol = epsimin; // 1.0e-9*sqrt(m+n);
	real err = 1.0;
	int loop = 0;

	while (err > tol && loop < 500) {
		loop++;
		XYZofLAMBDA(x);
		DualGrad(x);
		real theta = 1.0;
		lam = (lam + theta * grad).cwiseMax(0);
		err = grad.norm();
	}
}

real SparseMMASolver::DualResidual(VectorXr& x, real epsi) {

	VectorXr res(2 * m);
	res.head(m) = -b - a * z - y + mu;
	const RowVectorXr inv_upp_x = (upp - x).cwiseInverse();
	const RowVectorXr inv_x_low = (x - low).cwiseInverse();
	res.head(m) += inv_upp_x * pij + inv_x_low * qij;
	res.tail(m) = mu.cwiseProduct(lam).array() - epsi;

	return res.cwiseAbs().maxCoeff();
}

void SparseMMASolver::DualLineSearch() {

	real theta = 1.005;
	theta = std::max(std::max(theta, (-1.01 * s.head(m).cwiseQuotient(lam)).maxCoeff()),
		(-1.01 * s.tail(m).cwiseQuotient(mu)).maxCoeff());
	theta = 1.0 / theta;

	lam += theta * s.head(m);
	mu += theta * s.tail(m);
}

void SparseMMASolver::DualHess(VectorXr& x) {
	VectorXr df2(n);

	const VectorXr pjlam = p0 + pij * lam;
	const VectorXr qjlam = q0 + qij * lam;
	// PQ = diag(1 / (upp - x)^2) * pij - diag(1 / (x - low)^2) * qij.
	const MatrixXr PQ = pij.cwiseProduct((upp - x).cwiseAbs2().cwiseInverse() * RowVectorXr::Ones(m))
		- qij.cwiseProduct((x - low).cwiseAbs2().cwiseInverse() * RowVectorXr::Ones(m));
	df2 = -1.0 / (2.0 * pjlam.array() / (upp - x).array().pow(3) + 2.0 * qjlam.array() / (x - low).array().pow(3));
	const VectorXr sqrt_pjlam = pjlam.cwiseSqrt();
	const VectorXr sqrt_qjlam = qjlam.cwiseSqrt();
	const VectorXr xp = (sqrt_pjlam.cwiseProduct(low) + sqrt_qjlam.cwiseProduct(upp)).cwiseQuotient(sqrt_pjlam + sqrt_qjlam);
	df2 = (xp.array() < alpha.array()).select(0, df2);
	df2 = (xp.array() > beta.array()).select(0, df2);

	// Create the matrix/matrix/matrix product: PQ^T * diag(df2) * PQ
	hess = PQ.transpose() * df2.asDiagonal() * PQ;

	const real lamai = lam.dot(a);
	lam = lam.cwiseMax(0);
	hess.diagonal() += (lam.array() > c.array()).select(-1, VectorXr::Zero(m));
	hess.diagonal() += -mu.cwiseQuotient(lam);

	if (lamai > 0.0) {
		hess += -10.0 * a * a.transpose();
	}

	// pos def check
	const real HessTrace = hess.trace();
	real HessCorr = 1e-4 * HessTrace / m;

	if (-1.0 * HessCorr < 1.0e-7) {
		HessCorr = -1.0e-7;
	}

	hess.diagonal() += VectorXr::Constant(m, HessCorr);
}

void SparseMMASolver::DualGrad(VectorXr& x) {
	grad = -b - a * z - y;
	const RowVectorXr inv_upp_x = (upp - x).cwiseInverse();
	const RowVectorXr inv_x_low = (x - low).cwiseInverse();
	grad += inv_upp_x * pij + inv_x_low * qij;
}

void SparseMMASolver::XYZofLAMBDA(VectorXr& x) {
	lam = lam.cwiseMax(0);
	y = (lam - c).cwiseMax(0);
	const real lamai = lam.dot(a);
	z = std::max(0.0, 10.0 * (lamai - 1.0)); // SINCE a0 = 1.0

	const VectorXr pjlam = p0 + pij * lam;
	const VectorXr qjlam = q0 + qij * lam;
	const VectorXr sqrt_pjlam = pjlam.cwiseSqrt();
	const VectorXr sqrt_qjlam = qjlam.cwiseSqrt();
	x = (sqrt_pjlam.cwiseProduct(low) + sqrt_qjlam.cwiseProduct(upp)).cwiseQuotient(sqrt_pjlam + sqrt_qjlam);
	x = x.cwiseMax(alpha);
	x = x.cwiseMin(beta);
}

void SparseMMASolver::GenSub(const VectorXr& xval, const VectorXr& dfdx, const VectorXr& gx, const MatrixXr& dgdx,
	const VectorXr& xmin, const VectorXr& xmax)
{
	// Forward the iterator
	iter++;

	// Set asymptotes
	if (iter < 3) {
		low = xval - asyminit * (xmax - xmin);
		upp = xval + asyminit * (xmax - xmin);
	} else {
		const VectorXr zzz = (xval - xold1).cwiseProduct(xold1 - xold2);
		const VectorXr gamma = (zzz.array() < 0).select(asymdec,
			(zzz.array() > 0).select(asyminc, VectorXr::Ones(n)));
		low = xval - gamma.cwiseProduct(xold1 - low);
		upp = xval + gamma.cwiseProduct(upp - xold1);
		const VectorXr xmami = (xmax - xmin).cwiseMax(xmamieps);
		low = low.cwiseMax(xval - 100.0 * xmami);
		low = low.cwiseMin(xval - 1.0e-5 * xmami);
		upp = upp.cwiseMax(xval + 1.0e-5 * xmami);
		upp = upp.cwiseMin(xval + 100.0 * xmami);
		const VectorXr xmi = xmin.array() - 1.0e-6;
		const VectorXr xma = xmax.array() + 1.0e-6;
		low = (xval.array() < xmi.array()).select(xval - (xma - xval) / 0.9, low);
		upp = (xval.array() < xmi.array()).select(xval + (xma - xval) / 0.9, upp);
		low = (xval.array() > xma.array()).select(xval - (xval - xmi) / 0.9, low);
		upp = (xval.array() > xma.array()).select(xval + (xval - xmi) / 0.9, upp);
	}

	// Set bounds and the coefficients for the approximation
	// real raa0 = 0.5*1e-6;
	alpha = xmin.cwiseMax(low + albefa * (xval - low));
	alpha = alpha.cwiseMax(xval - move * (xmax - xmin));
	alpha = alpha.cwiseMin(xmax);
	beta = xmax.cwiseMin(upp - albefa * (upp - xval));
	beta = beta.cwiseMin(xval + move * (xmax - xmin));
	beta = beta.cwiseMax(xmin);
	// Objective function.
	const VectorXr dfdxp = dfdx.cwiseMax(0);
	const VectorXr dfdxm = (-1.0 * dfdx).cwiseMax(0);
	const VectorXr xmamiinv = (xmax - xmin).cwiseMax(xmamieps).cwiseInverse();
	const VectorXr pq = 0.001 * dfdx.cwiseAbs() + raa0 * xmamiinv;
	p0 = (upp - xval).cwiseAbs2().cwiseProduct(dfdxp + pq);
	q0 = (xval - low).cwiseAbs2().cwiseProduct(dfdxm + pq);
	// Constraints.
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; j++) {
			real dgdxp = std::max(0.0, dgdx(j, i));
			real dgdxm = std::max(0.0, -1.0 * dgdx(j, i));
			real pq_ij = 0.001 * std::abs(dgdx(j, i)) + raa0 * xmamiinv(i);
			pij(i, j) = std::pow(upp(i) - xval(i), 2.0) * (dgdxp + pq_ij);
			qij(i, j) = std::pow(xval(i) - low(i), 2.0) * (dgdxm + pq_ij);
		}
	}

	// The constant for the constraints
	b = -gx;
	const RowVectorXr inv_upp_xval = (upp - xval).cwiseInverse();
	const RowVectorXr inv_xval_low = (xval - low).cwiseInverse();
	b += inv_upp_xval * pij + inv_xval_low * qij;
}
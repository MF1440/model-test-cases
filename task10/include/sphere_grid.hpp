#ifndef TASK10_SPHERE_GRID_HPP
#define TASK10_SPHERE_GRID_HPP

#include "typedefs.hpp"

#include <cmath>

template <typename Scalar>
class SphereGrid {
public:
    explicit SphereGrid(int n, Scalar radius = static_cast<Scalar>(6378135.0))
        : _n(n), _radius(radius), _pts(3, n)
    {
        constexpr Scalar golden = static_cast<Scalar>(1.6180339887498949);
        constexpr Scalar pi = static_cast<Scalar>(M_PI);

        for (int i = 0; i < _n; ++i) {
            Scalar theta = std::acos(Scalar(1) - Scalar(2) * (i + Scalar(0.5)) / _n);
            Scalar phi = Scalar(2) * pi * i / golden;

            _pts(0, i) = _radius * std::sin(theta) * std::cos(phi);
            _pts(1, i) = _radius * std::sin(theta) * std::sin(phi);
            _pts(2, i) = _radius * std::cos(theta);
        }
    }

    int N() const { return _n; }
    Scalar R() const { return _radius; }
    const DenseMatrix<Scalar, 3, Eigen::Dynamic>& Pts() const { return _pts; }
    auto Point(int i) const { return _pts.col(i); }

    inline Scalar SqDist(int i, int j) const {
        return (_pts.col(i) - _pts.col(j)).squaredNorm();
    }

private:
    int _n;
    Scalar _radius;
    DenseMatrix<Scalar, 3, Eigen::Dynamic> _pts;
};

#endif // TASK10_SPHERE_GRID_HPP

#ifndef TASK10_TYPEDEFS_HPP
#define TASK10_TYPEDEFS_HPP

#include <cstdint>
#include <Eigen/Dense>

using Index = Eigen::Index;
using StorageIndex = std::int32_t;

template <typename T, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
using DenseMatrix = Eigen::Matrix<T, Rows, Cols>;

template <typename T, int Rows = Eigen::Dynamic>
using Vector = Eigen::Matrix<T, Rows, 1>;

template <typename T>
using RowMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>;

#endif // TASK10_TYPEDEFS_HPP

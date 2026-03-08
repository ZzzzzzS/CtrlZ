/**
 * @file MathFunction.hpp
 * @author zishun zhou
 * @brief Math functions for Vector and Tensor types
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <cmath>
#include <algorithm>
#include "VectorType.hpp"
#include "TensorType.hpp"

namespace z
{
    namespace math
    {
        /************************** Vector utility functions **************************/

        /**
         * @brief clamp val to min and max (vector-vector)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val value to clamp
         * @param min min value
         * @param max max value
         * @return Vector<T, N> clamp result
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> clamp(const Vector<T, N>& val, const Vector<T, N>& min, const Vector<T, N>& max)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::clamp(val[i], min[i], max[i]);
            }
            return result;
        }

        /**
         * @brief clamp val to min and max (vector-scalar)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val value to clamp
         * @param min min value
         * @param max max value
         * @return Vector<T, N> clamp result
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> clamp(const Vector<T, N>& val, T min, T max)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::clamp(val[i], min, max);
            }
            return result;
        }

        /**
         * @brief abs val
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val value to abs
         * @return Vector<T, N> abs result
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> abs(const Vector<T, N>& val)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::abs(val[i]);
            }
            return result;
        }

        /**
         * @brief min val1 and val2 (vector-vector)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val1 value 1
         * @param val2 value 2
         * @return Vector<T, N> min result
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> min(const Vector<T, N>& val1, const Vector<T, N>& val2)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::min(val1[i], val2[i]);
            }
            return result;
        }

        /**
         * @brief max val1 and val2 (vector-vector)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val1 value 1
         * @param val2 value 2
         * @return Vector<T, N> max result
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> max(const Vector<T, N>& val1, const Vector<T, N>& val2)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::max(val1[i], val2[i]);
            }
            return result;
        }

        /**
         * @brief min val and scalar
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val value to min
         * @param vmin min value
         * @return Vector<T, N> min result
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> min(const Vector<T, N>& val, T vmin)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::min(val[i], vmin);
            }
            return result;
        }

        /**
         * @brief max val and scalar
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val value to max
         * @param vmax max value
         * @return Vector<T, N> max result
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> max(const Vector<T, N>& val, T vmax)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::max(val[i], vmax);
            }
            return result;
        }

        /************************** Vector math functions **************************/

        /**
         * @brief Element-wise exponential e^x
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with exponential values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> exp(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::exp(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise natural logarithm ln(x)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with log values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> log(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::log(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise base-10 logarithm log10(x)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with log10 values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> log10(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::log10(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise base-2 logarithm log2(x)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with log2 values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> log2(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::log2(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise power with vector exponent
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param base base vector
         * @param exp exponent vector
         * @return Vector<T, N> vector with power values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> pow(const Vector<T, N>& base, const Vector<T, N>& exp)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::pow(base[i], exp[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise power with scalar exponent
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param base base vector
         * @param exp exponent scalar
         * @return Vector<T, N> vector with power values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> pow(const Vector<T, N>& base, T exp)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::pow(base[i], exp);
            }
            return result;
        }

        /**
         * @brief Element-wise square root
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with square root values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> sqrt(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::sqrt(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise cubic root
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with cubic root values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> cbrt(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::cbrt(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise sine
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with sine values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> sin(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::sin(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise cosine
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with cosine values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> cos(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::cos(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise tangent
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with tangent values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> tan(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::tan(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise arcsine
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with arcsine values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> asin(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::asin(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise arccosine
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with arccosine values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> acos(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::acos(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise arctangent
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with arctangent values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> atan(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::atan(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise hyperbolic sine
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with hyperbolic sine values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> sinh(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::sinh(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise hyperbolic cosine
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with hyperbolic cosine values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> cosh(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::cosh(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise hyperbolic tangent
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with hyperbolic tangent values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> tanh(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::tanh(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise inverse hyperbolic sine
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with inverse hyperbolic sine values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> asinh(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::asinh(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise inverse hyperbolic cosine
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with inverse hyperbolic cosine values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> acosh(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::acosh(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise inverse hyperbolic tangent
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with inverse hyperbolic tangent values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> atanh(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::atanh(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise 2^x
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with 2^x values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> exp2(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::exp2(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise exp(x) - 1
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with exp(x) - 1 values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> expm1(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::expm1(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise log(1 + x)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with log1p values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> log1p(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::log1p(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise floor
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with floor values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> floor(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::floor(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise ceiling
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with ceil values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> ceil(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::ceil(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise round to nearest
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with round values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> round(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::round(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise truncation toward zero
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with trunc values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> trunc(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::trunc(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise floating-point remainder (vector-vector)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param x dividend vector
         * @param y divisor vector
         * @return Vector<T, N> vector with fmod values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> fmod(const Vector<T, N>& x, const Vector<T, N>& y)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::fmod(x[i], y[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise floating-point remainder with scalar
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param x dividend vector
         * @param y divisor scalar
         * @return Vector<T, N> vector with fmod values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> fmod(const Vector<T, N>& x, T y)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::fmod(x[i], y);
            }
            return result;
        }

        /**
         * @brief Element-wise hypot(x, y) = sqrt(x^2 + y^2)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param x first vector
         * @param y second vector
         * @return Vector<T, N> vector with hypot values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> hypot(const Vector<T, N>& x, const Vector<T, N>& y)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::hypot(x[i], y[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise arctangent of y/x (considering quadrant)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param y first vector
         * @param x second vector
         * @return Vector<T, N> vector with arctangent values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> atan2(const Vector<T, N>& y, const Vector<T, N>& x)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::atan2(y[i], x[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise error function erf(x)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with erf values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> erf(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::erf(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise complementary error function erfc(x)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with erfc values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> erfc(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::erfc(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise gamma function
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with gamma values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> tgamma(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::tgamma(vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise log-gamma function
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with lgamma values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> lgamma(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::lgamma(vec[i]);
            }
            return result;
        }

        /************************** Vector activation functions **************************/

        /**
         * @brief Element-wise ReLU activation max(0, x)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with ReLU values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> relu(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::max(vec[i], static_cast<T>(0));
            }
            return result;
        }

        /**
         * @brief Element-wise leaky ReLU activation
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @param negative_slope slope for x < 0
         * @return Vector<T, N> vector with leaky ReLU values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> leaky_relu(const Vector<T, N>& vec, T negative_slope = static_cast<T>(0.01))
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = (vec[i] >= static_cast<T>(0)) ? vec[i] : (negative_slope * vec[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise sigmoid activation 1 / (1 + exp(-x))
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with sigmoid values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> sigmoid(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-vec[i]));
            }
            return result;
        }

        /**
         * @brief Element-wise softplus activation ln(1 + exp(x))
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with softplus values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> softplus(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::log(static_cast<T>(1) + std::exp(vec[i]));
            }
            return result;
        }

        /**
         * @brief Element-wise ELU activation
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @param alpha scale for x < 0
         * @return Vector<T, N> vector with ELU values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> elu(const Vector<T, N>& vec, T alpha = static_cast<T>(1))
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = (vec[i] >= static_cast<T>(0)) ? vec[i] : (alpha * (std::exp(vec[i]) - static_cast<T>(1)));
            }
            return result;
        }

        /**
         * @brief Element-wise SELU activation
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @param lambda scale for all x
         * @param alpha scale for x < 0
         * @return Vector<T, N> vector with SELU values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> selu(const Vector<T, N>& vec, T lambda = static_cast<T>(1.0507009873554805), T alpha = static_cast<T>(1.6732632423543772))
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                T v = (vec[i] >= static_cast<T>(0)) ? vec[i] : (alpha * (std::exp(vec[i]) - static_cast<T>(1)));
                result[i] = lambda * v;
            }
            return result;
        }

        /**
         * @brief Element-wise GELU activation
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with GELU values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> gelu(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            constexpr T k0 = static_cast<T>(0.5);
            constexpr T k1 = static_cast<T>(0.7978845608028654);
            constexpr T k2 = static_cast<T>(0.044715);
            for (size_t i = 0; i < N; i++)
            {
                T x = vec[i];
                T inner = k1 * (x + k2 * x * x * x);
                result[i] = k0 * x * (static_cast<T>(1) + std::tanh(inner));
            }
            return result;
        }

        /**
         * @brief Element-wise Swish activation x * sigmoid(x)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with Swish values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> swish(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                T x = vec[i];
                result[i] = x / (static_cast<T>(1) + std::exp(-x));
            }
            return result;
        }

        /**
         * @brief Element-wise Mish activation x * tanh(softplus(x))
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with Mish values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> mish(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                T x = vec[i];
                result[i] = x * std::tanh(std::log(static_cast<T>(1) + std::exp(x)));
            }
            return result;
        }

        /**
         * @brief Element-wise softsign activation x / (1 + |x|)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with softsign values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> softsign(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                T x = vec[i];
                result[i] = x / (static_cast<T>(1) + std::abs(x));
            }
            return result;
        }

        /**
         * @brief Element-wise hard sigmoid activation
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @param slope slope for linear region
         * @param offset offset for linear region
         * @return Vector<T, N> vector with hard sigmoid values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> hard_sigmoid(const Vector<T, N>& vec, T slope = static_cast<T>(0.2), T offset = static_cast<T>(0.5))
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = std::clamp(slope * vec[i] + offset, static_cast<T>(0), static_cast<T>(1));
            }
            return result;
        }

        /**
         * @brief Element-wise hard swish activation x * relu6(x + 3) / 6
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with hard swish values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> hard_swish(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                T x = vec[i];
                T relu6 = std::clamp(x + static_cast<T>(3), static_cast<T>(0), static_cast<T>(6));
                result[i] = x * relu6 / static_cast<T>(6);
            }
            return result;
        }

        /************************** Vector utility functions **************************/

        /**
         * @brief Return a vector of elements selected from either val1 or val2, depending on condition.
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param cond When True (nonzero), yield val1, otherwise yield val2
         * @param val1 value vector 1
         * @param val2 value vector 2
         * @return Vector<T, N> result vector
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> where(const Vector<bool, N>& cond, const Vector<T, N>& val1, const Vector<T, N>& val2)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = cond[i] ? val1[i] : val2[i];
            }
            return result;
        }

        /**
         * @brief Computes element-wise equality (vector-vector)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val1 the vector to compare
         * @param val2 the vector to compare with
         * @return Vector<bool, N> the output vector
         */
        template<typename T, size_t N>
        constexpr Vector<bool, N> eq(const Vector<T, N>& val1, const Vector<T, N>& val2)
        {
            Vector<bool, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = (val1[i] == val2[i]);
            }
            return result;
        }

        /**
         * @brief Computes element-wise equality (vector-scalar)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val1 the vector to compare
         * @param val2 the value to compare with
         * @return Vector<bool, N> the output vector
         */
        template<typename T, size_t N>
        constexpr Vector<bool, N> eq(const Vector<T, N>& val1, T val2)
        {
            Vector<bool, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = (val1[i] == val2);
            }
            return result;
        }

        /**
         * @brief Computes element-wise not equal to (vector-vector)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val1 the vector to compare
         * @param val2 the vector to compare with
         * @return Vector<bool, N> the output vector
         */
        template<typename T, size_t N>
        constexpr Vector<bool, N> ne(const Vector<T, N>& val1, const Vector<T, N>& val2)
        {
            Vector<bool, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = (val1[i] != val2[i]);
            }
            return result;
        }

        /**
         * @brief Computes element-wise not equal to (vector-scalar)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param val1 the vector to compare
         * @param val2 the value to compare with
         * @return Vector<bool, N> the output vector
         */
        template<typename T, size_t N>
        constexpr Vector<bool, N> ne(const Vector<T, N>& val1, T val2)
        {
            Vector<bool, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = (val1[i] != val2);
            }
            return result;
        }

        /**
         * @brief Tests if all elements in input evaluate to True.
         * @tparam N length of vector
         * @param val input vector
         * @return true if all elements are true
         * @return false otherwise
         */
        template<size_t N>
        constexpr bool all(const Vector<bool, N>& val)
        {
            for (size_t i = 0; i < N; i++)
            {
                if (!val[i])
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Tests if any elements in input evaluate to True.
         * @tparam N length of vector
         * @param val input vector
         * @return true if any element is true
         * @return false otherwise
         */
        template<size_t N>
        constexpr bool any(const Vector<bool, N>& val)
        {
            for (size_t i = 0; i < N; i++)
            {
                if (val[i])
                {
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief concatenate multiple z Vector
         * @tparam T type
         * @tparam Ns length of vectors
         * @param vectors vectors
         * @return concatenated vector
         */
        template<typename T, size_t ...Ns>
        constexpr auto cat(const Vector<T, Ns>&... vectors) {
            constexpr size_t total_size = (Ns + ...);
            Vector<T, total_size> result;
            size_t offset = 0;
            ((std::copy(vectors.begin(), vectors.end(), result.begin() + offset), offset += Ns), ...);
            return result;
        }

        /**
         * @brief Sum all elements of vector
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return T sum of all elements
         */
        template<typename T, size_t N>
        constexpr T sum(const Vector<T, N>& vec)
        {
            T result = T(0);
            for (size_t i = 0; i < N; i++)
            {
                result += vec[i];
            }
            return result;
        }

        /**
         * @brief Compute mean of all elements
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return T mean of all elements
         */
        template<typename T, size_t N>
        constexpr T mean(const Vector<T, N>& vec)
        {
            return sum(vec) / static_cast<T>(N);
        }

        /**
         * @brief Compute product of all elements
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return T product of all elements
         */
        template<typename T, size_t N>
        constexpr T prod(const Vector<T, N>& vec)
        {
            T result = T(1);
            for (size_t i = 0; i < N; i++)
            {
                result *= vec[i];
            }
            return result;
        }

        /**
         * @brief Compute variance of all elements
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return T variance of all elements
         */
        template<typename T, size_t N>
        constexpr T var(const Vector<T, N>& vec)
        {
            T m = mean(vec);
            T result = T(0);
            for (size_t i = 0; i < N; i++)
            {
                T diff = vec[i] - m;
                result += diff * diff;
            }
            return result / static_cast<T>(N);
        }

        /**
         * @brief Compute standard deviation of all elements
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return T standard deviation of all elements
         */
        template<typename T, size_t N>
        constexpr T stddev(const Vector<T, N>& vec)
        {
            return std::sqrt(var(vec));
        }

        /**
         * @brief Element-wise square
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with squared values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> square(const Vector<T, N>& vec)
        {
            return vec * vec;
        }

        /**
         * @brief Element-wise reciprocal (1/x)
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with reciprocal values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> reciprocal(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = T(1) / vec[i];
            }
            return result;
        }

        /**
         * @brief Element-wise sign function
         * @tparam T type of vector element
         * @tparam N length of vector
         * @param vec input vector
         * @return Vector<T, N> vector with sign values (-1, 0, 1)
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> sign(const Vector<T, N>& vec)
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = (vec[i] > T(0)) ? T(1) : ((vec[i] < T(0)) ? T(-1) : T(0));
            }
            return result;
        }

        /**
         * @brief Create a 1D vector with values from start to stop with step
         * @tparam T type of vector element
         * @tparam N size of vector
         * @param start start value
         * @param step step size
         * @return Vector<T, N> 1D vector with arange values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> arange(T start = T(0), T step = T(1))
        {
            Vector<T, N> result;
            for (size_t i = 0; i < N; ++i)
            {
                result[i] = start + static_cast<T>(i) * step;
            }
            return result;
        }

        /**
         * @brief Create a 1D vector with evenly spaced values from start to end
         * @tparam T type of vector element
         * @tparam N size of vector
         * @param start start value
         * @param end end value
         * @return Vector<T, N> 1D vector with linspace values
         */
        template<typename T, size_t N>
        constexpr Vector<T, N> linspace(T start, T end)
        {
            Vector<T, N> result;
            if (N == 1)
            {
                result[0] = start;
                return result;
            }
            T step = (end - start) / static_cast<T>(N - 1);
            for (size_t i = 0; i < N; ++i)
            {
                result[i] = start + static_cast<T>(i) * step;
            }
            return result;
        }

        /************************** Tensor utility functions **************************/
        /**
         * @brief Create a 2D identity matrix
         * @tparam T type of tensor element
         * @tparam N size of square matrix
         * @return Tensor<T, N, N> identity matrix
         */
        template<typename T, int64_t N>
        constexpr Tensor<T, N, N> eye()
        {
            Tensor<T, N, N> result(T(0));
            for (int64_t i = 0; i < N; ++i)
            {
                result(i, i) = T(1);
            }
            return result;
        }

        /**
         * @brief Alias for eye function
         * @tparam T type of tensor element
         * @tparam N size of square matrix
         * @return Tensor<T, N, N> identity matrix
         */
        template<typename T, int64_t N>
        constexpr Tensor<T, N, N> Identity()
        {
            return eye<T, N>();
        }

        /**
         * @brief Create a diagonal matrix from a 1D tensor
         * @tparam T type of tensor element
         * @tparam N size of diagonal
         * @param vec input 1D tensor
         * @return Tensor<T, N, N> diagonal matrix
         */
        template<typename T, int64_t N>
        constexpr Tensor<T, N, N> diag(const Tensor<T, N>& vec)
        {
            Tensor<T, N, N> result(T(0));
            for (int64_t i = 0; i < N; ++i)
            {
                result(i, i) = vec[i];
            }
            return result;
        }

        /**
         * @brief Compute the transpose of a 2D tensor
         * @tparam T type of tensor element
         * @tparam M number of rows
         * @tparam N number of columns
         * @param mat input matrix
         * @return Tensor<T, N, M> transposed matrix
         */
        template<typename T, int64_t M, int64_t N>
        constexpr Tensor<T, N, M> transpose(const Tensor<T, M, N>& mat)
        {
            Tensor<T, N, M> result;
            for (int64_t i = 0; i < M; ++i)
            {
                for (int64_t j = 0; j < N; ++j)
                {
                    result(j, i) = mat(i, j);
                }
            }
            return result;
        }

        /************************** Tensor math functions **************************/

        /**
         * @brief Element-wise absolute value
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with absolute values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> abs(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::abs(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise square root
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with square root values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> sqrt(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::sqrt(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise cubic root
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with cubic root values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> cbrt(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::cbrt(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise exponential
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with exponential values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> exp(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::exp(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise 2^x
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with 2^x values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> exp2(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::exp2(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise exp(x) - 1
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with exp(x) - 1 values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> expm1(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::expm1(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise natural logarithm
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with log values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> log(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::log(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise base-10 logarithm
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with log10 values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> log10(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::log10(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise base-2 logarithm
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with log2 values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> log2(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::log2(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise log(1 + x)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with log1p values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> log1p(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::log1p(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise sine
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with sine values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> sin(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::sin(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise cosine
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with cosine values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> cos(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::cos(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise tangent
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with tangent values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> tan(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::tan(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise arcsine
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with arcsine values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> asin(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::asin(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise arccosine
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with arccosine values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> acos(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::acos(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise arctangent
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with arctangent values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> atan(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::atan(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise arctangent of y/x (considering quadrant)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param y first tensor
         * @param x second tensor
         * @return Tensor<T, Dims...> tensor with arctangent values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> atan2(const Tensor<T, Dims...>& y, const Tensor<T, Dims...>& x)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < y.size(); ++i)
            {
                result[i] = std::atan2(y[i], x[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise hyperbolic sine
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with hyperbolic sine values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> sinh(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::sinh(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise hyperbolic cosine
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with hyperbolic cosine values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> cosh(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::cosh(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise hyperbolic tangent
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with hyperbolic tangent values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> tanh(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::tanh(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise inverse hyperbolic sine
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with inverse hyperbolic sine values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> asinh(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::asinh(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise inverse hyperbolic cosine
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with inverse hyperbolic cosine values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> acosh(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::acosh(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise inverse hyperbolic tangent
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with inverse hyperbolic tangent values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> atanh(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::atanh(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise floor
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with floor values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> floor(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::floor(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise ceiling
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with ceil values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> ceil(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::ceil(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise round to nearest
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with round values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> round(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::round(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise truncation toward zero
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with trunc values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> trunc(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::trunc(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise floating-point remainder
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param x dividend tensor
         * @param y divisor tensor
         * @return Tensor<T, Dims...> tensor with fmod values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> fmod(const Tensor<T, Dims...>& x, const Tensor<T, Dims...>& y)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < x.size(); ++i)
            {
                result[i] = std::fmod(x[i], y[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise floating-point remainder with scalar
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param x dividend tensor
         * @param y divisor scalar
         * @return Tensor<T, Dims...> tensor with fmod values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> fmod(const Tensor<T, Dims...>& x, T y)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < x.size(); ++i)
            {
                result[i] = std::fmod(x[i], y);
            }
            return result;
        }

        /**
         * @brief Element-wise hypot(x, y) = sqrt(x^2 + y^2)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param x first tensor
         * @param y second tensor
         * @return Tensor<T, Dims...> tensor with hypot values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> hypot(const Tensor<T, Dims...>& x, const Tensor<T, Dims...>& y)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < x.size(); ++i)
            {
                result[i] = std::hypot(x[i], y[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise error function erf(x)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with erf values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> erf(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::erf(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise complementary error function erfc(x)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with erfc values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> erfc(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::erfc(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise gamma function
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with tgamma values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> tgamma(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::tgamma(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise log-gamma function
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with lgamma values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> lgamma(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::lgamma(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise power
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param exponent exponent value
         * @return Tensor<T, Dims...> tensor with power values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> pow(const Tensor<T, Dims...>& t, T exponent)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::pow(t[i], exponent);
            }
            return result;
        }

        /**
         * @brief Element-wise power with tensor-tensor
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param base base tensor
         * @param exp exponent tensor
         * @return Tensor<T, Dims...> tensor with power values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> pow(const Tensor<T, Dims...>& base, const Tensor<T, Dims...>& exp)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < base.size(); ++i)
            {
                result[i] = std::pow(base[i], exp[i]);
            }
            return result;
        }

        /**
         * @brief Sum all elements of tensor
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T sum of all elements
         */
        template<typename T, int64_t... Dims>
        constexpr T sum(const Tensor<T, Dims...>& t)
        {
            T result = T(0);
            for (size_t i = 0; i < t.size(); ++i)
            {
                result += t[i];
            }
            return result;
        }

        /**
         * @brief Compute mean of all elements
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T mean of all elements
         */
        template<typename T, int64_t... Dims>
        constexpr T mean(const Tensor<T, Dims...>& t)
        {
            return sum(t) / static_cast<T>(t.size());
        }

        /**
         * @brief Compute minimum element
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T minimum element
         */
        template<typename T, int64_t... Dims>
        constexpr T min(const Tensor<T, Dims...>& t)
        {
            T result = t[0];
            for (size_t i = 1; i < t.size(); ++i)
            {
                if (t[i] < result)
                    result = t[i];
            }
            return result;
        }

        /**
         * @brief Compute maximum element
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T maximum element
         */
        template<typename T, int64_t... Dims>
        constexpr T max(const Tensor<T, Dims...>& t)
        {
            T result = t[0];
            for (size_t i = 1; i < t.size(); ++i)
            {
                if (t[i] > result)
                    result = t[i];
            }
            return result;
        }

        /**
         * @brief Clamp tensor values to a range [min_val, max_val]
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param min_val minimum value
         * @param max_val maximum value
         * @return Tensor<T, Dims...> clamped tensor
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> clamp(const Tensor<T, Dims...>& t, T min_val, T max_val)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::max(min_val, std::min(t[i], max_val));
            }
            return result;
        }

        /**
         * @brief Clamp tensor values to a range [min_val, max_val]
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param min_val minimum value tensor
         * @param max_val maximum value tensor
         * @return Tensor<T, Dims...> clamped tensor
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> clamp(const Tensor<T, Dims...>& t, const Tensor<T, Dims...>& min_val, const Tensor<T, Dims...>& max_val)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::max(min_val[i], std::min(t[i], max_val[i]));
            }
            return result;
        }

        /**
         * @brief Element-wise minimum of two tensors
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param a first tensor
         * @param b second tensor
         * @return Tensor<T, Dims...> tensor with minimum values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> min(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < a.size(); ++i)
            {
                result[i] = std::min(a[i], b[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise minimum of tensor and scalar
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param val scalar value
         * @return Tensor<T, Dims...> tensor with minimum values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> min(const Tensor<T, Dims...>& t, T val)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::min(t[i], val);
            }
            return result;
        }

        /**
         * @brief Element-wise maximum of two tensors
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param a first tensor
         * @param b second tensor
         * @return Tensor<T, Dims...> tensor with maximum values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> max(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < a.size(); ++i)
            {
                result[i] = std::max(a[i], b[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise maximum of tensor and scalar
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param val scalar value
         * @return Tensor<T, Dims...> tensor with maximum values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> max(const Tensor<T, Dims...>& t, T val)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::max(t[i], val);
            }
            return result;
        }

        /************************** Tensor activation functions **************************/

        /**
         * @brief Element-wise ReLU activation max(0, x)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with ReLU values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> relu(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::max(t[i], T(0));
            }
            return result;
        }

        /**
         * @brief Element-wise leaky ReLU activation
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param negative_slope slope for x < 0
         * @return Tensor<T, Dims...> tensor with leaky ReLU values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> leaky_relu(const Tensor<T, Dims...>& t, T negative_slope = T(0.01))
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = (t[i] >= T(0)) ? t[i] : (negative_slope * t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise sigmoid activation 1 / (1 + exp(-x))
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with sigmoid values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> sigmoid(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = T(1) / (T(1) + std::exp(-t[i]));
            }
            return result;
        }

        /**
         * @brief Element-wise softplus activation ln(1 + exp(x))
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with softplus values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> softplus(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::log(T(1) + std::exp(t[i]));
            }
            return result;
        }

        /**
         * @brief Element-wise ELU activation
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param alpha scale for x < 0
         * @return Tensor<T, Dims...> tensor with ELU values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> elu(const Tensor<T, Dims...>& t, T alpha = T(1))
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = (t[i] >= T(0)) ? t[i] : (alpha * (std::exp(t[i]) - T(1)));
            }
            return result;
        }

        /**
         * @brief Element-wise SELU activation
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param lambda scale for all x
         * @param alpha scale for x < 0
         * @return Tensor<T, Dims...> tensor with SELU values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> selu(const Tensor<T, Dims...>& t,
            T lambda = T(1.0507009873554805),
            T alpha = T(1.6732632423543772))
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                T v = (t[i] >= T(0)) ? t[i] : (alpha * (std::exp(t[i]) - T(1)));
                result[i] = lambda * v;
            }
            return result;
        }

        /**
         * @brief Element-wise GELU activation
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with GELU values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> gelu(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            constexpr T k0 = T(0.5);
            constexpr T k1 = T(0.7978845608028654); // sqrt(2/pi)
            constexpr T k2 = T(0.044715);
            for (size_t i = 0; i < t.size(); ++i)
            {
                T x = t[i];
                T inner = k1 * (x + k2 * x * x * x);
                result[i] = k0 * x * (T(1) + std::tanh(inner));
            }
            return result;
        }

        /**
         * @brief Element-wise Swish activation x * sigmoid(x)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with Swish values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> swish(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                T x = t[i];
                result[i] = x / (T(1) + std::exp(-x));
            }
            return result;
        }

        /**
         * @brief Element-wise Mish activation x * tanh(softplus(x))
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with Mish values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> mish(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                T x = t[i];
                result[i] = x * std::tanh(std::log(T(1) + std::exp(x)));
            }
            return result;
        }

        /**
         * @brief Element-wise softsign activation x / (1 + |x|)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with softsign values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> softsign(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                T x = t[i];
                result[i] = x / (T(1) + std::abs(x));
            }
            return result;
        }

        /**
         * @brief Element-wise hard sigmoid activation
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param slope slope for linear region
         * @param offset offset for linear region
         * @return Tensor<T, Dims...> tensor with hard sigmoid values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> hard_sigmoid(const Tensor<T, Dims...>& t,
            T slope = T(0.2),
            T offset = T(0.5))
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::clamp(slope * t[i] + offset, T(0), T(1));
            }
            return result;
        }

        /**
         * @brief Element-wise hard swish activation x * relu6(x + 3) / 6
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with hard swish values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> hard_swish(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                T x = t[i];
                T relu6 = std::clamp(x + T(3), T(0), T(6));
                result[i] = x * relu6 / T(6);
            }
            return result;
        }

        /************************** Tensor utility functions **************************/

        /**
         * @brief Check if all elements are true
         * @tparam Dims dimensions of tensor
         * @param t input bool tensor
         * @return true if all elements are true
         * @return false otherwise
         */
        template<int64_t... Dims>
        constexpr bool all(const Tensor<bool, Dims...>& t)
        {
            for (size_t i = 0; i < t.size(); ++i)
            {
                if (!t[i])
                    return false;
            }
            return true;
        }

        /**
         * @brief Check if any element is true
         * @tparam Dims dimensions of tensor
         * @param t input bool tensor
         * @return true if any element is true
         * @return false otherwise
         */
        template<int64_t... Dims>
        constexpr bool any(const Tensor<bool, Dims...>& t)
        {
            for (size_t i = 0; i < t.size(); ++i)
            {
                if (t[i])
                    return true;
            }
            return false;
        }

        /**
         * @brief Element-wise where condition selection
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param cond condition tensor
         * @param x values where condition is true
         * @param y values where condition is false
         * @return Tensor<T, Dims...> selected values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> where(const Tensor<bool, Dims...>& cond,
            const Tensor<T, Dims...>& x,
            const Tensor<T, Dims...>& y)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < cond.size(); ++i)
            {
                result[i] = cond[i] ? x[i] : y[i];
            }
            return result;
        }

        /**
         * @brief Compute product of all elements
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T product of all elements
         */
        template<typename T, int64_t... Dims>
        constexpr T prod(const Tensor<T, Dims...>& t)
        {
            T result = T(1);
            for (size_t i = 0; i < t.size(); ++i)
            {
                result *= t[i];
            }
            return result;
        }

        /**
         * @brief Compute variance of all elements
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T variance of all elements
         */
        template<typename T, int64_t... Dims>
        constexpr T var(const Tensor<T, Dims...>& t)
        {
            T m = mean(t);
            T result = T(0);
            for (size_t i = 0; i < t.size(); ++i)
            {
                T diff = t[i] - m;
                result += diff * diff;
            }
            return result / static_cast<T>(t.size());
        }

        /**
         * @brief Compute standard deviation of all elements
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T standard deviation of all elements
         */
        template<typename T, int64_t... Dims>
        constexpr T stddev(const Tensor<T, Dims...>& t)
        {
            return std::sqrt(var(t));
        }

        /**
         * @brief Element-wise square
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with squared values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> square(const Tensor<T, Dims...>& t)
        {
            return t * t;
        }

        /**
         * @brief Element-wise reciprocal (1/x)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with reciprocal values
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> reciprocal(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = T(1) / t[i];
            }
            return result;
        }

        /**
         * @brief Element-wise sign function
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with sign values (-1, 0, 1)
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> sign(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = (t[i] > T(0)) ? T(1) : ((t[i] < T(0)) ? T(-1) : T(0));
            }
            return result;
        }

        /**
         * @brief Computes element-wise equality (tensor-tensor)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param a the tensor to compare
         * @param b the tensor to compare with
         * @return Tensor<bool, Dims...> the output tensor
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<bool, Dims...> eq(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b)
        {
            Tensor<bool, Dims...> result;
            for (size_t i = 0; i < a.size(); ++i)
            {
                result[i] = (a[i] == b[i]);
            }
            return result;
        }

        /**
         * @brief Computes element-wise equality (tensor-scalar)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t the tensor to compare
         * @param val the value to compare with
         * @return Tensor<bool, Dims...> the output tensor
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<bool, Dims...> eq(const Tensor<T, Dims...>& t, T val)
        {
            Tensor<bool, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = (t[i] == val);
            }
            return result;
        }

        /**
         * @brief Computes element-wise not equal to (tensor-tensor)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param a the tensor to compare
         * @param b the tensor to compare with
         * @return Tensor<bool, Dims...> the output tensor
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<bool, Dims...> ne(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b)
        {
            Tensor<bool, Dims...> result;
            for (size_t i = 0; i < a.size(); ++i)
            {
                result[i] = (a[i] != b[i]);
            }
            return result;
        }

        /**
         * @brief Computes element-wise not equal to (tensor-scalar)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t the tensor to compare
         * @param val the value to compare with
         * @return Tensor<bool, Dims...> the output tensor
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<bool, Dims...> ne(const Tensor<T, Dims...>& t, T val)
        {
            Tensor<bool, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = (t[i] != val);
            }
            return result;
        }

        /**
         * @brief Create a tensor filled with a specific value (alias for full)
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param value the value to fill
         * @return Tensor<T, Dims...> tensor filled with value
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> fill(const T& value)
        {
            return Tensor<T, Dims...>::full(value);
        }

        /**
         * @brief Create an empty tensor (uninitialized, for performance)
         * Note: Elements are default-initialized, use with caution
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @return Tensor<T, Dims...> empty tensor
         */
        template<typename T, int64_t... Dims>
        constexpr Tensor<T, Dims...> empty()
        {
            return Tensor<T, Dims...>();
        }

        /************************** Tensor matrix multiplication functions **************************/

        /**
         * @brief batch matrix multiplication (bmm) - free function version
         * @details Perform batch matrix multiplication on two 3D tensors.
         *          a: (B, M, K), b: (B, K, N), result: (B, M, N)
         *          Usage: auto result = bmm(a, b); // automatic type deduction
         * @tparam T type of tensor element
         * @tparam B batch size
         * @tparam M first matrix rows
         * @tparam K inner dimension
         * @tparam N second matrix columns
         * @param a first batch matrix with shape (B, M, K)
         * @param b second batch matrix with shape (B, K, N)
         * @return Tensor<T, B, M, N> result batch matrix
         */
        template<typename T, int64_t B, int64_t M, int64_t K, int64_t N>
        constexpr Tensor<T, B, M, N> bmm(const Tensor<T, B, M, K>& a, const Tensor<T, B, K, N>& b)
        {
            Tensor<T, B, M, N> result;

            for (int64_t batch = 0; batch < B; ++batch)
            {
                for (int64_t i = 0; i < M; ++i)
                {
                    for (int64_t j = 0; j < N; ++j)
                    {
                        T sum = T(0);
                        for (int64_t k = 0; k < K; ++k)
                        {
                            sum += a(batch, i, k) * b(batch, k, j);
                        }
                        result(batch, i, j) = sum;
                    }
                }
            }
            return result;
        }

        /**
         * @brief matrix multiplication (mm) - free function version for 2D tensors
         * @details Perform matrix multiplication on two 2D tensors.
         *          a: (M, K), b: (K, N), result: (M, N)
         *          Usage: auto result = mm(a, b); // automatic type deduction
         * @tparam T type of tensor element
         * @tparam M first matrix rows
         * @tparam K inner dimension
         * @tparam N second matrix columns
         * @param a first matrix with shape (M, K)
         * @param b second matrix with shape (K, N)
         * @return Tensor<T, M, N> result matrix
         */
        template<typename T, int64_t M, int64_t K, int64_t N>
        constexpr Tensor<T, M, N> mm(const Tensor<T, M, K>& a, const Tensor<T, K, N>& b)
        {
            Tensor<T, M, N> result;

            for (int64_t i = 0; i < M; ++i)
            {
                for (int64_t j = 0; j < N; ++j)
                {
                    T sum = T(0);
                    for (int64_t k = 0; k < K; ++k)
                    {
                        sum += a(i, k) * b(k, j);
                    }
                    result(i, j) = sum;
                }
            }
            return result;
        }

    };
};

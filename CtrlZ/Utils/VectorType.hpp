/**
 * @file VectorType.hpp
 * @author zishun zhou
 * @brief 定义了一些向量类型
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <iostream>
#include <array>
#include <cmath>
#include <algorithm>
#include <functional>
#include <format>

 /**
  * @brief overload operator << for std::array to print array
  *
  * @tparam T array type
  * @tparam N array length
  * @param os std::ostream
  * @param arr std::array<T, N>
  * @return std::ostream&
  */
template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr) {
    os << "[";
    for (std::size_t i = 0; i < N; ++i) {
        os << arr[i];
        if (i < N - 1) {
            os << ", ";
        }
    }
    os << "]\n";
    return os;
}

/**
 * @brief overload operator << for bool type std::array to print array
 *
 * @tparam N array length
 * @param os std::ostream
 * @param arr std::array<T, N>
 * @return std::ostream&
 */
template<std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<bool, N>& arr) {
    os << "[";
    for (std::size_t i = 0; i < N; ++i) {
        os << (arr[i] ? "true" : "false");
        if (i < N - 1) {
            os << ", ";
        }
    }
    os << "]\n";
    return os;
}

namespace z
{
    /**
     * @brief math namespace, contains some math functions
     *
     */
    namespace math
    {
        /**
         * @brief Vector class, support some vector operations, like dot, cross, normalize, etc.
         *
         * @tparam T type of vector element
         * @tparam N length of vector
         */
        template<typename T, size_t N>
        class Vector : public std::array<T, N>
        {
            static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type");
        public:
            /**
             * @brief clamp val to min and max
             * @param val value to clamp
             * @param min min value
             * @param max max value
             * @return Vector<T, N> clamp result
             */
            static constexpr Vector<T, N> clamp(const Vector<T, N>& val, const Vector<T, N>& min, const Vector<T, N>& max)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {

                    result[i] = std::clamp(val[i], min[i], max[i]);
                }
                return result;
            }

            /**
             * @brief abs val
             * @param val value to abs
             * @return Vector<T, N> abs result
             */
            static constexpr Vector<T, N> abs(const Vector<T, N>& val)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::abs(val[i]);
                }
                return result;
            }

            /**
             * @brief min val1 and val2
             * @param val1 value 1
             * @param val2 value 2
             * @return Vector<T, N> min result
             */
            static constexpr Vector<T, N> min(const Vector<T, N>& val1, const Vector<T, N>& val2)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::min(val1[i], val2[i]);
                }
                return result;
            }

            /**
             * @brief max val1 and val2
             * @param val1 value 1
             * @param val2 value 2
             * @return Vector<T, N> max result
             */
            static constexpr Vector<T, N> max(const Vector<T, N>& val1, const Vector<T, N>& val2)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::max(val1[i], val2[i]);
                }
                return result;
            }

            /**
             * @brief clamp val to min and max
             * @param val value to clamp
             * @param min min value
             * @param max max value
             * @return Vector<T, N> clamp result
             */
            static constexpr Vector<T, N> clamp(const Vector<T, N>& val, T min, T max)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::clamp(val[i], min, max);
                }
                return result;
            }

            /**
             * @brief max val to max
             * @param val value to max
             * @param max max value
             * @return Vector<T, N> max result
             */
            static constexpr Vector<T, N> max(const Vector<T, N>& val, T max)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::max(val[i], max);
                }
                return result;
            }

            /**
             * @brief min val to min
             * @param val value to min
             * @param min min value
             * @return Vector<T, N> min result
             */
            static constexpr Vector<T, N> min(const Vector<T, N>& val, T min)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::min(val[i], min);
                }
                return result;
            }

            /**
             * @brief create a vector with all elements set to 0
             *
             * @return Vector<T, N>
             */
            static constexpr Vector<T, N> zeros()
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = 0;
                }
                return result;
            }

            /**
             * @brief create a vector with all elements set to 1
             *
             * @return Vector<T, N>
             */
            static constexpr Vector<T, N> ones()
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = 1;
                }
                return result;
            }

            /// @brief apply function to vector
            using ApplyFunc = std::function<T(const T&, size_t)>;

            /**
             * @brief apply function to vector element
             *
             * @param val vector
             * @param func apply function
             * @return Vector<T, N> result vector
             */
            static Vector<T, N> apply(const Vector<T, N>& val, ApplyFunc func)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = func(val[i], i);
                }
                return result;
            }

            /**
             * @brief create a vector with random elements between 0 and 1
             * @warning this function does not set seed, therefore the result is not completely random
             * user should set seed before call this function.
             *
             * @return Vector<T, N>
             */
            static Vector<T, N> rand()
            {
                Vector<T, N> result;
                //std::srand(std::time({}));//set seed
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = static_cast<T>(std::rand()) / RAND_MAX;
                }
                return result;
            }

            /**
             * @brief Return a vector of elements selected from either val1 or val2, depending on condition.
             *
             * @param cond When True (nonzero), yield val1, otherwise yield val2
             * @param val1 value vector 1
             * @param val2 value vector 2
             * @return constexpr Vector<T, N> result vector
             */
            static constexpr Vector<T, N> where(const Vector<bool, N>& cond, const Vector<T, N>& val1, const Vector<T, N>& val2)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = cond[i] ? val1[i] : val2[i];
                }
                return result;
            }

            /**
             * @brief Computes element-wise equality
             *
             * @param val1 the vector to compare
             * @param val2 the vector to compare with
             * @return constexpr Vector<bool, N>  the output vector
             */
            static constexpr Vector<bool, N> eq(const Vector<T, N>& val1, const Vector<T, N>& val2)
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = (val1[i] == val2[i]);
                }
                return result;
            }

            /**
             * @brief Computes element-wise equality
             *
             * @param val1 the vector to compare
             * @param val2 the value to compare with
             * @return constexpr Vector<bool, N> the output vector
             */
            static constexpr Vector<bool, N> eq(const Vector<T, N>& val1, T val2)
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = (val1[i] == val2);
                }
                return result;
            }

            /**
             * @brief Computes element-wise not equal to
             *
             * @param val1 the vector to compare
             * @param val2 the vector to compare with
             * @return constexpr Vector<bool, N> the output vector
             */
            static constexpr Vector<bool, N> ne(const Vector<T, N>& val1, const Vector<T, N>& val2)
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = (val1[i] != val2[i]);
                }
                return result;
            }

            /**
             * @brief Computes element-wise not equal to
             *
             * @param val1 the vector to compare
             * @param val2 the value to compare with
             * @return constexpr Vector<bool, N> the output vector
             */
            static constexpr Vector<bool, N> ne(const Vector<T, N>& val1, T val2)
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = (val1[i] != val2);
                }
                return result;
            }

            /*****math functions******/

            /**
             * @brief element-wise exponential
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> exp(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::exp(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise natural logarithm
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> log(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::log(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise base-10 logarithm
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> log10(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::log10(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise base-2 logarithm
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> log2(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::log2(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise power (base^exp), vector-vector
             * @param base base vector
             * @param exp exponent vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> pow(const Vector<T, N>& base, const Vector<T, N>& exp)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::pow(base[i], exp[i]);
                }
                return result;
            }

            /**
             * @brief element-wise power (base^exp), vector-scalar
             * @param base base vector
             * @param exp exponent scalar
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> pow(const Vector<T, N>& base, T exp)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::pow(base[i], exp);
                }
                return result;
            }

            /**
             * @brief element-wise square root
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> sqrt(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::sqrt(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise cubic root
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> cbrt(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::cbrt(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise sine (radians)
             * @param vec input vector (radians)
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> sin(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::sin(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise cosine (radians)
             * @param vec input vector (radians)
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> cos(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::cos(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise tangent (radians)
             * @param vec input vector (radians)
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> tan(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::tan(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise arcsine (radians)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> asin(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::asin(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise arccosine (radians)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> acos(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::acos(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise arctangent (radians)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> atan(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::atan(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise hyperbolic sine
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> sinh(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::sinh(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise hyperbolic cosine
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> cosh(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::cosh(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise hyperbolic tangent
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> tanh(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::tanh(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise inverse hyperbolic sine
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> asinh(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::asinh(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise inverse hyperbolic cosine
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> acosh(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::acosh(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise inverse hyperbolic tangent
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> atanh(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::atanh(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise 2^x
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> exp2(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::exp2(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise exp(x) - 1
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> expm1(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::expm1(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise log(1 + x)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> log1p(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::log1p(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise floor
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> floor(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::floor(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise ceil
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> ceil(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::ceil(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise round to nearest
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> round(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::round(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise truncation toward zero
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> trunc(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::trunc(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise floating-point remainder (x mod y), vector-vector
             * @param x dividend vector
             * @param y divisor vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> fmod(const Vector<T, N>& x, const Vector<T, N>& y)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::fmod(x[i], y[i]);
                }
                return result;
            }

            /**
             * @brief element-wise floating-point remainder (x mod y), vector-scalar
             * @param x dividend vector
             * @param y divisor scalar
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> fmod(const Vector<T, N>& x, T y)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::fmod(x[i], y);
                }
                return result;
            }

            /**
             * @brief element-wise hypot(x, y)
             * @param x x vector
             * @param y y vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> hypot(const Vector<T, N>& x, const Vector<T, N>& y)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::hypot(x[i], y[i]);
                }
                return result;
            }

            /**
             * @brief element-wise atan2(y, x)
             * @param y y vector
             * @param x x vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> atan2(const Vector<T, N>& y, const Vector<T, N>& x)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::atan2(y[i], x[i]);
                }
                return result;
            }

            /**
             * @brief element-wise error function erf(x)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> erf(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::erf(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise complementary error function erfc(x)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> erfc(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::erfc(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise gamma function Γ(x)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> tgamma(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::tgamma(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise log-gamma function ln|Γ(x)|
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> lgamma(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::lgamma(vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise ReLU activation max(0, x)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> relu(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::max(vec[i], static_cast<T>(0));
                }
                return result;
            }

            /**
             * @brief element-wise leaky ReLU activation
             * @param vec input vector
             * @param negative_slope slope for x < 0
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> leaky_relu(const Vector<T, N>& vec, T negative_slope = static_cast<T>(0.01))
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = (vec[i] >= static_cast<T>(0)) ? vec[i] : (negative_slope * vec[i]);
                }
                return result;
            }

            /**
             * @brief element-wise sigmoid activation 1 / (1 + exp(-x))
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> sigmoid(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-vec[i]));
                }
                return result;
            }

            /**
             * @brief element-wise softplus activation ln(1 + exp(x))
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> softplus(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::log(static_cast<T>(1) + std::exp(vec[i]));
                }
                return result;
            }

            /**
             * @brief element-wise ELU activation
             * @param vec input vector
             * @param alpha scale for x < 0
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> elu(const Vector<T, N>& vec, T alpha = static_cast<T>(1))
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = (vec[i] >= static_cast<T>(0)) ? vec[i] : (alpha * (std::exp(vec[i]) - static_cast<T>(1)));
                }
                return result;
            }

            /**
             * @brief element-wise SELU activation
             * @param vec input vector
             * @param lambda scale for all x
             * @param alpha scale for x < 0
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> selu(const Vector<T, N>& vec,
                T lambda = static_cast<T>(1.0507009873554805),
                T alpha = static_cast<T>(1.6732632423543772))
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
             * @brief element-wise GELU activation (tanh approximation)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> gelu(const Vector<T, N>& vec)
            {
                Vector<T, N> result;
                constexpr T k0 = static_cast<T>(0.5);
                constexpr T k1 = static_cast<T>(0.7978845608028654); // sqrt(2/pi)
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
             * @brief element-wise Swish activation x * sigmoid(x)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> swish(const Vector<T, N>& vec)
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
             * @brief element-wise Mish activation x * tanh(softplus(x))
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> mish(const Vector<T, N>& vec)
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
             * @brief element-wise softsign activation x / (1 + |x|)
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> softsign(const Vector<T, N>& vec)
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
             * @brief element-wise hard sigmoid activation
             * @param vec input vector
             * @param slope slope for linear region
             * @param offset offset for linear region
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> hard_sigmoid(const Vector<T, N>& vec,
                T slope = static_cast<T>(0.2),
                T offset = static_cast<T>(0.5))
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = std::clamp(slope * vec[i] + offset, static_cast<T>(0), static_cast<T>(1));
                }
                return result;
            }

            /**
             * @brief element-wise hard swish activation x * relu6(x + 3) / 6
             * @param vec input vector
             * @return Vector<T, N> result vector
             */
            static constexpr Vector<T, N> hard_swish(const Vector<T, N>& vec)
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



            /**
             * @brief get a bool type vector with the same size as this vector
             *
             */
            using BoolType = Vector<bool, N>;

            /**
             * @brief get the int type vector with the same size as this vector
             *
             */
            using IntType = Vector<int, N>;

        public:

            /**
             * @brief operator ()
             *
             * @param idx index
             * @return T& reference of element
             */
            constexpr T& operator()(int idx)
            {
                if (idx < -static_cast<int>(N) || idx >= static_cast<int>(N))
                    throw std::out_of_range("Index out of range in Vector<bool, N>::operator[]");
                // idx must be in range [-N, N-1]

                if (idx < 0)
                    return this->operator[](N + idx);
                else
                    return this->operator[](idx);
            }

            /**
             * @brief operator () const
             *
             * @param idx index
             * @return const T& reference of element
             */
            constexpr const T& operator()(int idx) const
            {
                if (idx < -static_cast<int>(N) || idx >= static_cast<int>(N))
                    throw std::out_of_range("Index out of range in Vector<bool, N>::operator[]");
                // idx must be in range [-N, N-1]
                if (idx < 0)
                    return this->operator[](N + idx);
                else
                    return this->operator[](idx);
            }

            /**
             * @brief operator []
             *
             * @param idx index
             * @return constexpr T& reference of element
             */
            constexpr T& operator[](int idx)
            {
                if (idx < -static_cast<int>(N) || idx >= static_cast<int>(N))
                    throw std::out_of_range("Index out of range in Vector<bool, N>::operator[]");
                // idx must be in range [-N, N-1]
                if (idx < 0)
                    return std::array<T, N>::operator[](N + idx);
                else
                    return std::array<T, N>::operator[](idx);
            }

            /**
             * @brief operator [] const
             *
             * @param idx index
             * @return constexpr const T& reference of element
             */
            constexpr const T& operator[](int idx) const
            {
                if (idx < -static_cast<int>(N) || idx >= static_cast<int>(N))
                    throw std::out_of_range("Index out of range in Vector<bool, N>::operator[]");
                // idx must be in range [-N, N-1]
                if (idx < 0)
                    return std::array<T, N>::operator[](N + idx);
                else
                    return std::array<T, N>::operator[](idx);
            }

            /**
             * @brief operator << for std::ostream
             *
             * @param os
             * @param vec
             * @return std::ostream&
             */
            friend std::ostream& operator<<(std::ostream& os, const Vector<T, N>& vec)
            {
                os << "Vector<" << typeid(T).name() << "," << N << ">: ";
                os << "[";
                for (size_t i = 0; i < N; i++)
                {
                    os << vec[i];
                    if (i != N - 1)
                    {
                        os << ",";
                    }
                }
                os << "]\n";
                return os;
            }

            /**
             * @brief operator + for vector addition
             *
             * @param other other vector
             * @return Vector<T, N>
             */
            constexpr Vector<T, N> operator+(const Vector<T, N>& other) const
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) + other[i];
                }
                return result;
            }

            /**
             * @brief operator - for vector subtraction
             *
             * @param other other vector
             * @return Vector<T, N>
             */
            constexpr Vector<T, N> operator-(const Vector<T, N>& other) const
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) - other[i];
                }
                return result;
            }

            /**
             * @brief operator +
             *
             * @return Vector<T, N>
             */
            constexpr Vector<T, N> operator+() const
            {
                return *this;
            }

            /**
             * @brief operator - for vector negation
             *
             * @return Vector<T, N>
             */
            constexpr Vector<T, N> operator-() const
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = -this->operator[](i);
                }
                return result;
            }

            /**
             * @brief vector batch multiplication
             *
             * @param other other vector
             * @return Vector<T, N>
             */
            constexpr Vector<T, N> operator*(const Vector<T, N>& other) const
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) * other[i];
                }
                return result;
            }

            /**
             * @brief vector batch division
             *
             * @param other other vector
             * @return Vector<T, N>
             */
            constexpr Vector<T, N> operator/(const Vector<T, N>& other) const
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) / other[i];
                }
                return result;
            }

            /**
             * @brief vector addition with value
             *
             * @param val value
             * @return Vector<T, N>
             */
            constexpr Vector<T, N> operator+(T val) const
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) + val;
                }
                return result;
            }


            /**
             * @brief vector subtraction with value
             *
             * @param val value
             * @return Vector<T, N>
             */
            constexpr Vector<T, N> operator-(T val) const
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) - val;
                }
                return result;
            }

            /**
             * @brief vector multiplication with value
             *
             * @param val value
             * @return Vector<T, N>
             */
            constexpr Vector<T, N> operator*(T val) const
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) * val;
                }
                return result;
            }

            /**
             * @brief vector division with value
             *
             * @param val
             * @return Vector<T, N>
             */
            constexpr Vector<T, N> operator/(T val) const
            {
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) / val;
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator==(const Vector<T, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) == other[i];
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator==(T other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) == other;
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator!=(const Vector<T, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) != other[i];
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator!=(T other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) != other;
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator>(const Vector<T, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) > other[i];
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator>(T other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) > other;
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator<(const Vector<T, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) < other[i];
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator<(T other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) < other;
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator>=(const Vector<T, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) >= other[i];
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator>=(T other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) >= other;
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator<=(const Vector<T, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) <= other[i];
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator<=(T other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) <= other;
                }
                return result;
            }


            /**
             * @brief vector in-place addition
             *
             * @param other other vector
             * @return Vector<T, N>&
             */
            constexpr Vector<T, N>& operator+=(const Vector<T, N>& other)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) += other[i];
                }
                return *this;
            }

            /**
             * @brief vector in-place subtraction
             *
             * @param other other vector
             * @return Vector<T, N>&
             */
            constexpr Vector<T, N>& operator-=(const Vector<T, N>& other)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) -= other[i];
                }
                return *this;
            }

            /**
             * @brief vector in-place batch multiplication
             *
             * @param other
             * @return Vector<T, N>&
             */
            constexpr Vector<T, N>& operator*=(const Vector<T, N>& other)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) *= other[i];
                }
                return *this;
            }

            /**
             * @brief vector in-place batch division
             *
             * @param other
             * @return Vector<T, N>&
             */
            constexpr Vector<T, N>& operator/=(const Vector<T, N>& other)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) /= other[i];
                }
                return *this;
            }

            /**
             * @brief vector in-place addition with value
             *
             * @param val
             * @return Vector<T, N>&
             */
            constexpr Vector<T, N>& operator+=(T val)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) += val;
                }
                return *this;
            }

            /**
             * @brief vector in-place subtraction with value
             *
             * @param val
             * @return Vector<T, N>&
             */
            constexpr Vector<T, N>& operator-=(T val)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) -= val;
                }
                return *this;
            }

            /**
             * @brief vector in-place multiplication with value
             *
             * @param val
             * @return Vector<T, N>&
             */
            constexpr Vector<T, N>& operator*=(T val)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) *= val;
                }
                return *this;
            }

            /**
             * @brief vector in-place division with value
             *
             * @param val
             * @return Vector<T, N>&
             */
            constexpr Vector<T, N>& operator/=(T val)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) /= val;
                }
                return *this;
            }

            /**
             * @brief vector dot product
             *
             * @param other other vector
             * @return T dot product result
             */
            constexpr T dot(const Vector<T, N>& other) const
            {
                T result = 0;
                for (size_t i = 0; i < N; i++)
                {
                    result += this->operator[](i) * other[i];
                }
                return result;
            }

            /**
             * @brief vector length in L2 norm
             *
             * @return T length
             */
            T length() const
            {
                T result = 0;
                for (size_t i = 0; i < N; i++)
                {
                    result += this->operator[](i) * this->operator[](i);
                }
                return std::sqrt(result);
            }

            /**
             * @brief vector normalize in L2 norm
             *
             * @return Vector<T, N> normalized vector
             */
            Vector<T, N> normalize() const
            {
                T len = length();
                Vector<T, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) / len;
                }
                return result;
            }

            /**
             * @brief Max element in vector
             *
             * @return T
             */
            constexpr T max() const
            {
                T result = this->operator[](0);
                for (size_t i = 1; i < N; i++)
                {
                    if (this->operator[](i) > result)
                    {
                        result = this->operator[](i);
                    }
                }
                return result;
            }

            /**
             * @brief Min element in vector
             *
             * @return T
             */
            constexpr T min() const
            {
                T result = this->operator[](0);
                for (size_t i = 1; i < N; i++)
                {
                    if (this->operator[](i) < result)
                    {
                        result = this->operator[](i);
                    }
                }
                return result;
            }

            /**
             * @brief sum of all elements
             *
             * @return T
             */
            constexpr T sum() const
            {
                T result = 0;
                for (size_t i = 0; i < N; i++)
                {
                    result += this->operator[](i);
                }
                return result;
            }

            /**
             * @brief average of all elements
             *
             * @return T
             */
            constexpr T average() const
            {
                return sum() / N;
            }
            /**
             * @brief apply function to each element of the vector
             *
             */
            using SelfApplyFunc = std::function<void(T&, size_t)>;
            void apply(SelfApplyFunc func)
            {
                for (size_t i = 0; i < N; i++)
                {
                    func(this->operator[](i), i);
                }
            }

            template<size_t begin, size_t end, size_t step = 1>
            Vector<T, (end - begin) / step> slice() const
            {
                static_assert(begin < end, "begin must be less than end");
                static_assert(end <= N, "end must be less than or equal to N");
                static_assert(step > 0, "step must be greater than 0");
                static_assert((end - begin) / step > 0, "step must be less than slice length");
                Vector<T, (end - begin) / step> result;
                for (size_t i = begin, j = 0; i < end; i += step, j++)
                {
                    result[j] = this->operator[](i);
                }
                return result;
            }

            /**
             * @brief repeat the vector multiple times
             *
             * @tparam RepeatN
             * @return constexpr Vector<T, N* RepeatN>
             */
            template<size_t RepeatN>
            constexpr Vector<T, N* RepeatN> repeat() const
            {
                Vector<T, N* RepeatN> result;
                for (size_t i = 0; i < RepeatN; i++)
                {
                    for (size_t j = 0; j < N; j++)
                    {
                        result[i * N + j] = this->operator[](j);
                    }
                }
                return result;
            }

            /**
             * @brief remap the vector with given index
             * @details remap the vector with given index, for example, after remap with index {2,-1,1},
             * the origin vector {3,4,5} should be {5,5,4}
             *
             * @param idx new index
             * @return constexpr Vector<T, N>
             */
            constexpr Vector<T, N> remap(const Vector<int, N>& idx)
            {
                Vector<T, N> result;
                for (size_t i = 0;i < N;i++)
                {
                    if (idx[i] > static_cast<int>(N) || idx[i] < -static_cast<int>(N))
                    {
                        throw std::runtime_error("index out of range");
                    }

                    result[i] = this->operator[](idx[i]);
                }
                return result;
            }

            /**
             * @brief  cast the vector to another type
             *
             * @tparam Scalar
             * @return constexpr Vector<Scalar, N>
             */
            template<typename Scalar>
            constexpr Vector<Scalar, N> cast() const
            {
                static_assert(std::is_arithmetic_v<Scalar>, "Scalar must be an arithmetic type");
                static_assert(std::is_convertible_v<T, Scalar>, "T must be convertible to Scalar");
                Vector<Scalar, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = static_cast<Scalar>(this->operator[](i));
                }
                return result;
            }

            /**
             * @brief cast the vector to itself, this is a no-op function
             *
             * @return constexpr Vector<T, N>
             */
            constexpr Vector<T, N> cast() const
            {
                return *this;
            }

            /**
             * @brief convert the vector to another type
             *
             * @tparam Scalar
             * @return constexpr Vector<Scalar, N>
             */
            template<typename Scalar>
            constexpr Vector<Scalar, N> to() const
            {
                return cast<Scalar>();
            }
        };

        /**
         * @brief Vector class for bool type, support some vector logical operations etc.
         *
         * @tparam N length of vector
         */
        template<size_t N>
        class Vector<bool, N> : public std::array<bool, N>
        {
        public:
            /**
             * @brief create a vector with all elements set to false
             *
             * @return Vector<T, N>
             */
            static constexpr Vector<bool, N> zeros()
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = false;
                }
                return result;
            }

            /**
             * @brief create a vector with all elements set to true
             *
             * @return Vector<T, N>
             */
            static constexpr Vector<bool, N> ones()
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = true;
                }
                return result;
            }

            /// @brief apply function to vector
            using ApplyFunc = std::function<bool(const bool&, size_t)>;

            /**
             * @brief apply function to vector element
             *
             * @param val vector
             * @param func apply function
             * @return Vector<T, N> result vector
             */
            static Vector<bool, N> apply(const Vector<bool, N>& val, ApplyFunc func)
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = func(val[i], i);
                }
                return result;
            }

            /**
             * @brief create a vector with random elements within [0, 1]
             * @warning this function does not set seed, therefore the result is not completely random
             * user should set seed before call this function.
             *
             * @return Vector<T, N>
             */
            static Vector<bool, N> rand()
            {
                Vector<bool, N> result;
                //std::srand(std::time({}));//set seed
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = static_cast<bool>(std::rand() / (RAND_MAX / 2));
                }
                return result;
            }

            /**
             * @brief Tests if all elements in input evaluate to True.
             *
             * @param val input vector
             * @return true
             * @return false
             */
            static constexpr bool all(const Vector<bool, N>& val)
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
             *
             * @param val input vector
             * @return true
             * @return false
             */
            static constexpr bool any(const Vector<bool, N>& val)
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
             * @brief get the int type vector with the same size as this vector
             *
             */
            using IntType = Vector<int, N>;


        public:

            /**
             * @brief operator ()
             *
             * @param idx index
             * @return T& reference of element
             */
            constexpr bool& operator()(int idx)
            {
                if (idx < -static_cast<int>(N) || idx >= static_cast<int>(N))
                    throw std::out_of_range("Index out of range in Vector<bool, N>::operator[]");
                // idx must be in range [-N, N-1]
                if (idx < 0)
                    return this->operator[](N + idx);
                else
                    return this->operator[](idx);
            }

            /**
             * @brief operator () const
             *
             * @param idx index
             * @return const T& reference of element
             */
            constexpr const bool& operator()(int idx) const
            {
                if (idx < -static_cast<int>(N) || idx >= static_cast<int>(N))
                    throw std::out_of_range("Index out of range in Vector<bool, N>::operator[]");
                // idx must be in range [-N, N-1]
                if (idx < 0)
                    return this->operator[](N + idx);
                else
                    return this->operator[](idx);
            }

            /**
             * @brief operator []
             *
             * @param idx index
             * @return constexpr T& reference of element
             */
            constexpr bool& operator[](int idx)
            {
                if (idx < -static_cast<int>(N) || idx >= static_cast<int>(N))
                    throw std::out_of_range("Index out of range in Vector<bool, N>::operator[]");
                // idx must be in range [-N, N-1]
                if (idx < 0)
                    return std::array<bool, N>::operator[](N + idx);
                else
                    return std::array<bool, N>::operator[](idx);
            }

            /**
             * @brief operator [] const
             *
             * @param idx index
             * @return constexpr const T& reference of element
             */
            constexpr const bool& operator[](int idx) const
            {
                if (idx < -static_cast<int>(N) || idx >= static_cast<int>(N))
                    throw std::out_of_range("Index out of range in Vector<bool, N>::operator[]");

                // idx must be in range [-N, N-1]
                if (idx < 0)
                    return std::array<bool, N>::operator[](N + idx);
                else
                    return std::array<bool, N>::operator[](idx);
            }

            /**
             * @brief operator << for std::ostream
             *
             * @param os
             * @param vec
             * @return std::ostream&
             */
            friend std::ostream& operator<<(std::ostream& os, const Vector<bool, N>& vec)
            {
                os << "Vector<" << typeid(bool).name() << "," << N << ">: ";
                os << "[";
                for (size_t i = 0; i < N; i++)
                {
                    os << (vec[i] ? "true" : "false");
                    if (i != N - 1)
                    {
                        os << ", ";
                    }
                }
                os << "]\n";
                return os;
            }

            /**
             * @brief operator + for vector addition
             *
             * @param other other vector
             * @return Vector<T, N>
             */
            constexpr Vector<bool, N> operator+(const Vector<bool, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    //or operation
                    result[i] = this->operator[](i) || other[i];
                }
                return result;
            }

            /**
             * @brief operator - for vector subtraction
             *
             * @param other other vector
             * @return Vector<T, N>
             */
            constexpr Vector<bool, N> operator-(const Vector<bool, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    //xor operation
                    result[i] = (this->operator[](i) && !other[i]) || (!this->operator[](i) && other[i]);
                }
                return result;
            }

            /**
             * @brief operator +
             *
             * @return Vector<T, N>
             */
            constexpr Vector<bool, N> operator+() const
            {
                return *this;
            }

            /**
             * @brief operator - for vector negation
             *
             * @return Vector<T, N>
             */
            constexpr Vector<bool, N> operator-() const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    //not operation
                    result[i] = !this->operator[](i);
                }
                return result;
            }

            /**
             * @brief vector batch multiplication
             *
             * @param other other vector
             * @return Vector<T, N>
             */
            constexpr Vector<bool, N> operator*(const Vector<bool, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    //and operation
                    result[i] = this->operator[](i) && other[i];
                }
                return result;
            }

            /**
             * @brief vector batch division
             *
             * @param other other vector
             * @return Vector<T, N>
             */
            constexpr Vector<bool, N> operator/(const Vector<bool, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    //xnor
                    result[i] = (this->operator[](i) && other[i]) || (!this->operator[](i) && !other[i]);
                }
                return result;
            }

            /**
             * @brief vector addition with value
             *
             * @param val value
             * @return Vector<T, N>
             */
            constexpr Vector<bool, N> operator+(bool val) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) || val;
                }
                return result;
            }

            /**
             * @brief vector subtraction with value
             *
             * @param val value
             * @return Vector<T, N>
             */
            constexpr Vector<bool, N> operator-(bool val) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    //xor
                    result[i] = (this->operator[](i) && !val) || (!this->operator[](i) && val);
                }
                return result;
            }

            /**
             * @brief vector multiplication with value
             *
             * @param val value
             * @return Vector<T, N>
             */
            constexpr Vector<bool, N> operator*(bool val) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) && val;
                }
                return result;
            }

            /**
             * @brief vector division with value
             *
             * @param val
             * @return Vector<T, N>
             */
            constexpr Vector<bool, N> operator/(bool val) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    // xnor
                    result[i] = (this->operator[](i) && val) || (!this->operator[](i) && !val);
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator==(const Vector<bool, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) == other[i];
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator==(bool other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) == other;
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator!=(const Vector<bool, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) != other[i];
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator!=(bool other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) != other;
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator||(const Vector<bool, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) || other[i];
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator&&(const Vector<bool, N>& other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) && other[i];
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator||(bool other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) || other;
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator&&(bool other) const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = this->operator[](i) && other;
                }
                return result;
            }

            /**
             * @brief vector logical operations
             *
             * @param other other vector
             * @return constexpr Vector<bool, N>& result vector
             */
            constexpr Vector<bool, N> operator!() const
            {
                Vector<bool, N> result;
                for (size_t i = 0; i < N; i++)
                {
                    result[i] = !this->operator[](i);
                }
                return result;
            }

            /**
             * @brief vector in-place addition
             *
             * @param other other vector
             * @return Vector<T, N>&
             */
            constexpr Vector<bool, N>& operator+=(const Vector<bool, N>& other)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) = this->operator[](i) || other[i];
                }
                return *this;
            }

            /**
             * @brief vector in-place subtraction
             *
             * @param other other vector
             * @return Vector<T, N>&
             */
            constexpr Vector<bool, N>& operator-=(const Vector<bool, N>& other)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) = (this->operator[](i) && !other[i]) || (!this->operator[](i) && other[i]);
                }
                return *this;
            }

            /**
             * @brief vector in-place batch multiplication
             *
             * @param other
             * @return Vector<T, N>&
             */
            constexpr Vector<bool, N>& operator*=(const Vector<bool, N>& other)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) = this->operator[](i) && other[i];
                }
                return *this;
            }

            /**
             * @brief vector in-place batch division
             *
             * @param other
             * @return Vector<T, N>&
             */
            constexpr Vector<bool, N>& operator/=(const Vector<bool, N>& other)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) = (this->operator[](i) && other[i]) || (!this->operator[](i) && !other[i]);
                }
                return *this;
            }

            /**
             * @brief vector in-place addition with value
             *
             * @param val
             * @return Vector<T, N>&
             */
            constexpr Vector<bool, N>& operator+=(bool val)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) = this->operator[](i) || val;
                }
                return *this;
            }

            /**
             * @brief vector in-place subtraction with value
             *
             * @param val
             * @return Vector<T, N>&
             */
            constexpr Vector<bool, N>& operator-=(bool val)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) = (this->operator[](i) && !val) || (!this->operator[](i) && val);
                }
                return *this;
            }

            /**
             * @brief vector in-place multiplication with value
             *
             * @param val
             * @return Vector<T, N>&
             */
            constexpr Vector<bool, N>& operator*=(bool val)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) = this->operator[](i) && val;
                }
                return *this;
            }

            /**
             * @brief vector in-place division with value
             *
             * @param val
             * @return Vector<T, N>&
             */
            constexpr Vector<bool, N>& operator/=(bool val)
            {
                for (size_t i = 0; i < N; i++)
                {
                    this->operator[](i) = (this->operator[](i) && val) || (!this->operator[](i) && !val);
                }
                return *this;
            }

            /**
             * @brief check if any element is true
             *
             * @return true
             * @return false
             */
            bool any() const
            {
                for (size_t i = 0; i < N; i++)
                {
                    if (this->operator[](i))
                    {
                        return true;
                    }
                }
                return false;
            }

            /**
             * @brief check if all elements are true
             *
             * @return true
             * @return false
             */
            bool all() const
            {
                for (size_t i = 0; i < N; i++)
                {
                    if (!this->operator[](i))
                    {
                        return false;
                    }
                }
                return true;
            }


            /**
             * @brief sum of all elements
             *
             * @return T
             */
            constexpr size_t sum() const
            {
                size_t result = 0;
                for (size_t i = 0; i < N; i++)
                {
                    result += this->operator[](i) ? 1 : 0;
                }
                return result;
            }

            /**
             * @brief average of all elements
             *
             * @return T
             */
            constexpr size_t average() const
            {
                return sum() / N;
            }
            /**
             * @brief apply function to each element of the vector
             *
             */
            using SelfApplyFunc = std::function<void(bool&, size_t)>;
            void apply(SelfApplyFunc func)
            {
                for (size_t i = 0; i < N; i++)
                {
                    func(this->operator[](i), i);
                }
            }

            template<size_t begin, size_t end, size_t step = 1>
            Vector<bool, (end - begin) / step> slice() const
            {
                static_assert(begin < end, "begin must be less than end");
                static_assert(end <= N, "end must be less than or equal to N");
                static_assert(step > 0, "step must be greater than 0");
                static_assert((end - begin) / step > 0, "step must be less than slice length");
                Vector<bool, (end - begin) / step> result;
                for (size_t i = begin, j = 0; i < end; i += step, j++)
                {
                    result[j] = this->operator[](i);
                }
                return result;
            }

            /**
             * @brief repeat the vector multiple times
             *
             * @tparam RepeatN
             * @return constexpr Vector<bool, N* RepeatN>
             */
            template<size_t RepeatN>
            constexpr Vector<bool, N* RepeatN> repeat() const
            {
                Vector<bool, N* RepeatN> result;
                for (size_t i = 0; i < RepeatN; i++)
                {
                    for (size_t j = 0; j < N; j++)
                    {
                        result[i * N + j] = this->operator[](j);
                    }
                }
                return result;
            }
        };

        /**
         * @brief concatenate multiple std::array
         *
         * @tparam T type
         * @tparam Ns length of arrays
         * @param arrays arrays
         * @return auto
         */
        template<typename T, size_t... Ns>
        constexpr auto cat(const std::array<T, Ns>&... arrays) {
            constexpr size_t total_size = (Ns + ...);
            std::array<T, total_size> result;
            size_t offset = 0;
            ((std::copy(arrays.begin(), arrays.end(), result.begin() + offset), offset += Ns), ...);
            return result;
        }

        /**
         * @brief concatenate multiple z Vector
         *
         * @tparam T type
         * @tparam Ns length of vectors
         * @param vectors vectors
         * @return auto
         */
        template<typename T, size_t ...Ns>
        constexpr auto cat(const Vector<T, Ns>&... vectors) {
            constexpr size_t total_size = (Ns + ...);
            Vector<T, total_size> result;
            size_t offset = 0;
            ((std::copy(vectors.begin(), vectors.end(), result.begin() + offset), offset += Ns), ...);
            return result;
        }
    };
};

namespace std {
    // 为 z::math::Vector<T, N> 特化 formatter
    template<typename T, size_t N>
    struct formatter<z::math::Vector<T, N>> {
        constexpr auto parse(format_parse_context& ctx) {
            return ctx.begin();  // 基础版本，不支持格式规范
        }

        auto format(const z::math::Vector<T, N>& vec, format_context& ctx) const {
            auto out = ctx.out();
            out = std::format_to(out, "Vector<{},{}>: [", typeid(T).name(), N);
            for (size_t i = 0; i < N; ++i) {
                out = std::format_to(out, "{}", vec[i]);
                if (i != N - 1) {
                    out = std::format_to(out, ",");
                }
            }
            out = std::format_to(out, "]\n");
            return out;
        }
    };

    // 为 z::math::Vector<bool, N> 特化 formatter
    template<size_t N>
    struct formatter<z::math::Vector<bool, N>> {
        constexpr auto parse(format_parse_context& ctx) {
            return ctx.begin();
        }

        auto format(const z::math::Vector<bool, N>& vec, format_context& ctx) const {
            auto out = ctx.out();
            out = std::format_to(out, "Vector<bool,{}>: [", N);
            for (size_t i = 0; i < N; ++i) {
                out = std::format_to(out, "{}", vec[i] ? "true" : "false");
                if (i != N - 1) {
                    out = std::format_to(out, ",");
                }
            }
            out = std::format_to(out, "]\n");
            return out;
        }
    };
}
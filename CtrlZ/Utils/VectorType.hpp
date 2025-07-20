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
            //using std::array<T, N>::_Elems;
        public:
            /**
             * @brief clamp val to min and max
             *
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
             *
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
             *
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
             *
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
             *
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
             *
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
             *
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
            //using std::array<T, N>::_Elems;
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
        constexpr auto cat(Vector<T, Ns>&... vectors) {
            constexpr size_t total_size = (Ns + ...);
            Vector<T, total_size> result;
            size_t offset = 0;
            ((std::copy(vectors.begin(), vectors.end(), result.begin() + offset), offset += Ns), ...);
            return result;
        }


    };
};
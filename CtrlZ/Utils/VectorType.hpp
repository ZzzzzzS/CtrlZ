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
        public:
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
                    //FIXME: we might need to use a better random function, like std::mt19937, but for now we just use std::rand
                    result[i] = static_cast<T>(std::rand()) / RAND_MAX;
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


        // Free math functions are now in MathFunction.hpp
    }; // namespace math
}; // namespace z

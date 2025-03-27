/**
 * @file MathTypes.hpp
 * @author zishun zhou
 * @brief
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

        public:

            /**
             * @brief operator ()
             *
             * @param idx index
             * @return T& reference of element
             */
            constexpr T& operator()(int idx)
            {
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
            constexpr Vector<T, N> remap(const std::array<int, N>& idx)
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



        /************************************************************************************************* */

        /**
         * @brief TensorShape struct, used to store tensor shape information
         *
         * @tparam Dims
         */
        template<int64_t... Dims>
        struct TensorShape {

            /// @brief tensor shape array
            static constexpr int64_t dims[] = { Dims... };

            /// @brief number of dimensions
            static constexpr size_t num_dims = sizeof...(Dims);

            /// @brief total size of tensor
            static constexpr size_t total_size = (Dims * ...);

            /// @brief tensor shape array
            static constexpr std::array<int64_t, num_dims> dims_array = { Dims... };
        };

        /**
         * @brief TensorBase class, base class for tensor
         *
         * @tparam T type of tensor element
         * @tparam Dims tensor shape
         */
        template<typename T, int64_t... Dims>
        class TensorBase {
        public:
            using Shape = TensorShape<Dims...>;
            using ValueType = T;

            /**
             * @brief Construct a new Tensor Base object with all elements set to default value
             *
             */
            TensorBase() {
                data__.fill(ValueType());
            }

            /**
             * @brief Construct a new Tensor Base object with all elements set to given value
             *
             */
            TensorBase(const T& val) {
                data__.fill(val);
            }

            /**
             * @brief Construct a new Tensor Base object, copy from std::array
             *
             * @param data an array of data
             */
            TensorBase(const std::array<ValueType, Shape::total_size>& data) : data__(data) {}

            /**
             * @brief Construct a new Tensor Base object, move from std::array
             *
             * @param data an array of data
             */
            TensorBase(std::array<ValueType, Shape::total_size>&& data) : data__(std::move(data)) {}

            /**
             * @brief Construct a new Tensor Base object, copy from another tensor
             *
             * @param other another tensor
             */
            TensorBase(const TensorBase& other) = default;

            /**
             * @brief Construct a new Tensor Base object, move from another tensor
             *
             * @param other another tensor
             */
            TensorBase(TensorBase&& other) = default;

            /**
             * @brief convert to std::array
             *
             * @return std::array<T, Shape::total_size>& reference of data array
             */
            std::array<ValueType, Shape::total_size>& Array()
            {
                return data__;
            }

            /**
             * @brief get total size of tensor
             *
             * @return constexpr size_t
             */
            static constexpr size_t size() {
                return Shape::total_size;
            }

            /**
             * @brief get shape of tensor
             *
             * @return constexpr std::array<size_t, Shape::num_dims>
             */
            static constexpr std::array<int64_t, Shape::num_dims> shape() {
                return Shape::dims_array;
            }

            static constexpr const int64_t* shape_ptr() {
                return Shape::dims_array.data();
            }

            /**
             * @brief get data pointer
             *
             * @return ValueType* the pointer of data
             */
            ValueType* data() {
                return data__.data();
            }

            /**
             * @brief get the number of dimensions
             *
             * @return constexpr size_t number of dimensions
             */
            static constexpr size_t num_dims() {
                return Shape::num_dims;
            }

            /**
             * @brief get data according to index, this function will ignore the shape of tensor,
             * the index is the offset in the memory.
             *
             * @param index data index
             * @return T& reference of data
             */
            T& operator[](size_t index) {
                return data__[index];
            }

            /**
             * @brief this function is a overload of operator[], it will return the data according to the index.
             *
             * @param index data index
             * @return const T& reference of data
             */
            const T& operator[](size_t index) const {
                return data__[index];
            }

            /**
             * @brief get data according to indices, this function will calculate the offset according to the shape of tensor.
             * for example, a tensor with shape {2, 3, 4}, the index (1, 2, 3) will be calculated as 1*3*4 + 2*4 + 3 = 35.
             *
             * @tparam Indices indices
             * @param indices indices
             * @return T& reference of data
             */
            template<typename... Indices>
            T& operator()(Indices... indices) {
                static_assert(sizeof...(Indices) == Shape::num_dims, "Number of indices must match number of dimensions");
                size_t index = calculate_index(indices...);
                return data__[index];
            }

            /**
             * @brief this function is a overload of operator(), it will return the data according to the indices.
             *
             * @tparam Indices indices
             * @param indices indices
             * @return const T& reference of data
             */
            template<typename... Indices>
            const T& operator()(Indices... indices) const {
                static_assert(sizeof...(Indices) == Shape::num_dims, "Number of indices must match number of dimensions");
                size_t index = calculate_index(indices...);
                return data__[index];
            }

            /**
             * @brief get data according to indices, this function will calculate the offset according to the shape of tensor.
             * for example, a tensor with shape {2, 3, 4}, the index (1, 2, 3) will be calculated as 1*3*4 + 2*4 + 3 = 35.
             *
             * @tparam Indices indices
             * @param indices indices
             * @return T& reference of data
             */
            template<typename... Indices>
            T& at(Indices... indices) {
                static_assert(sizeof...(Indices) == Shape::num_dims, "Number of indices must match number of dimensions");
                size_t index = calculate_index(indices...);
                return data__[index];
            }

            /**
             * @brief this function is a overload of at, it will return the data according to the indices.
             *
             * @tparam Indices indices
             * @param indices indices
             * @return const T& reference of data
             */
            template<typename... Indices>
            const T& at(Indices... indices) const {
                static_assert(sizeof...(Indices) == Shape::num_dims, "Number of indices must match number of dimensions");
                size_t index = calculate_index(indices...);
                return data__[index];
            }

            /**
             * @brief operator << for std::ostream, used to output tensor data
             *
             * @param os    std::ostream
             * @param tensor    TensorBase
             * @return std::ostream&
             */
            friend std::ostream& operator<<(std::ostream& os, const TensorBase& tensor) {
                os << "Tensor<" << typeid(T).name() << ", ";
                for (size_t i = 0; i < Shape::num_dims; ++i) {
                    os << Shape::dims[i];
                    if (i + 1 < Shape::num_dims) {
                        os << ", ";
                    }
                }
                os << ">\n";

                PrintTensorElements(os, tensor, 0, 0);
                return os;
            }

        protected:
            /// @brief data array, used to store tensor data
            std::array<T, Shape::total_size> data__;

            /**
             * @brief calculate the index according to the indices
             *
             * @tparam Indices
             * @param indices
             * @return constexpr size_t
             */
             //TODO: add support for compile time index calculation
            template<typename... Indices>
            static constexpr size_t calculate_index(Indices... indices) {
                static_assert(sizeof...(Indices) == Shape::num_dims, "Number of indices must match number of dimensions");
                std::array<int64_t, Shape::num_dims> indices_array = { indices... };

                // calculate the index
                size_t index = 0;
                size_t factor = 1;

                for (int i = Shape::num_dims - 1; i >= 0; i--) {
                    std::cout << "i: " << i << std::endl;
                    index += indices_array[i] * factor;
                    factor *= Shape::dims_array[i];

                    if (indices_array[i] >= Shape::dims_array[i])
                    {
                        throw std::out_of_range("Index out of range");
                    }
                }
                std::cout << "index: " << index << std::endl;
                return index;
            }

            /**
             * @brief print tensor elements recursively
             *
             * @param os output stream
             * @param tensor tensor to print
             * @param index index of elements
             * @param level level of elements
             */
            static void PrintTensorElements(std::ostream& os, const TensorBase& tensor, size_t index, size_t level)
            {
                if (level == Shape::num_dims - 1) {
                    os << "[";
                    for (size_t i = 0; i < Shape::dims[level] - 1; ++i) {
                        os << tensor.data__[index + i] << ", ";
                    }
                    os << tensor.data__[index + Shape::dims[level] - 1];
                    os << "]";
                }
                else {
                    os << "[";
                    for (size_t i = 0; i < Shape::dims[level] - 1; ++i) {

                        PrintTensorElements(os, tensor, index + i * Shape::dims[level + 1], level + 1);
                        os << ",\n";
                    }
                    PrintTensorElements(os, tensor, index + (Shape::dims[level] - 1) * Shape::dims[level + 1], level + 1);
                    os << "]\n";
                }
            }
        };

        /**
         * @brief Tensor class, used to store tensor data
         *
         * @tparam T type of tensor element
         * @tparam Dims
         */
        template<typename T, int64_t... Dims>
        class Tensor : public TensorBase<T, Dims...> {
        public:
            using Base = TensorBase<T, Dims...>;
            using Shape = typename Base::Shape;
            using ValueType = typename Base::ValueType;
            using Base::Base;

            using Ptr = std::shared_ptr<Tensor<T, Dims...>>;
        public:

            /**
             * @brief create a new tensor pointer
             *
             * @return Ptr Pointer of new tensor
             */
            static Ptr Create()
            {
                return std::make_shared<Tensor<T, Dims...>>();
            }

            /**
             * @brief create a new tensor pointer with given value for all elements
             *
             * @param val value for all elements
             * @return Ptr Pointer of new tensor
             */
            static Ptr Create(const T& val)
            {
                return std::make_shared<Tensor<T, Dims...>>(val);
            }

            /**
             * @brief create a new tensor pointer with data
             *
             * @param data data array
             */
            static Ptr Create(const std::array<ValueType, Shape::total_size>& data)
            {
                return std::make_shared<Tensor<T, Dims...>>(data);
            }

            /**
             * @brief create a new tensor pointer with data
             *
             * @param data data array
             */
            static Ptr Create(std::array<ValueType, Shape::total_size>&& data)
            {
                return std::make_shared<Tensor<T, Dims...>>(std::move(data));
            }

        public:

            /**
             * @brief convert tensor to z vector
             *
             * @return Vector<ValueType, Shape::total_size> z vector
             */
            Vector<ValueType, Shape::total_size> toVector()
            {
                return Vector<ValueType, Shape::total_size>(this->data__);
            }
        };
    };
};
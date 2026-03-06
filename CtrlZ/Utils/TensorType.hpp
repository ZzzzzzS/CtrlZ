/**
 * @file TensorType.hpp
 * @author zishun zhou
 * @brief 定义了一些张量类型
 * @date 2025-05-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <array>
#include <memory>
#include <atomic>
#include <random>
#include <cmath>
#include "VectorType.hpp"

namespace z
{
    namespace math
    {
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
            TensorBase()
            {
                //std::cout << "default construct" << std::endl;
                this->ref_count__ = new std::atomic<size_t>(1);
                //this->ref_count__->load(1);

                this->data_ptr__ = new std::array<ValueType, Shape::total_size>();
                this->data_ptr__->fill(ValueType());
            }

            ~TensorBase()
            {
                if (this->ref_count__ == nullptr || this->data_ptr__ == nullptr) {
                    return;
                }
                //std::cout << "ref_count=" << *(this->ref_count__) << std::endl;
                this->ref_count__->fetch_sub(1);
                if (this->ref_count__->load() == 0) {
                    //std::cout << "delete data_ptr__" << std::endl;
                    delete this->data_ptr__;
                    delete this->ref_count__;
                }
            }

            /**
             * @brief Construct a new Tensor Base object with all elements set to given value
             *
             */
            TensorBase(const T& val)
            {
                this->ref_count__ = new std::atomic<size_t>(1);
                this->data_ptr__ = new std::array<ValueType, Shape::total_size>();
                this->data_ptr__->fill(val);
            }

            /**
             * @brief Construct a new Tensor Base object, copy from std::array
             *
             * @param data an array of data
             */
            TensorBase(const std::array<ValueType, Shape::total_size>& data)
            {
                this->ref_count__ = new std::atomic<size_t>(1);
                this->data_ptr__ = new std::array<ValueType, Shape::total_size>(data);
            }

            /**
             * @brief Construct a new Tensor Base object, move from std::array
             *
             * @param data an array of data
             */
            TensorBase(std::array<ValueType, Shape::total_size>&& data)
            {
                this->ref_count__ = new std::atomic<size_t>(1);
                this->data_ptr__ = new std::array<ValueType, Shape::total_size>(std::move(data));
            }

            /**
             * @brief Construct a new Tensor Base object, copy from another tensor
             *
             * @param other another tensor
             */
            TensorBase(const TensorBase& other) noexcept
            {
                //std::cout << "copy construct" << std::endl;
                this->ref_count__ = other.ref_count__;
                this->data_ptr__ = other.data_ptr__;
                this->ref_count__->fetch_add(1);
            }

            /**
             * @brief Construct a new Tensor Base object, move from another tensor
             *
             * @param other another tensor
             */
            TensorBase(TensorBase&& other) noexcept
            {
                //std::cout << "move construct" << std::endl;
                this->ref_count__ = other.ref_count__;
                this->data_ptr__ = other.data_ptr__;
                other.ref_count__ = nullptr;
                other.data_ptr__ = nullptr;
            }

            /**
             * @brief assignment operator, copy from another tensor
             *
             * @param other another tensor
             * @return TensorBase& reference of this tensor
             */
            TensorBase<T, Dims...>& operator=(const TensorBase& other) noexcept
            {
                //std::cout << "copy assign" << std::endl;
                if (this != &other) {
                    this->ref_count__->fetch_sub(1);
                    if (this->ref_count__->load() == 0) {
                        //std::cout << "delete data_ptr__" << std::endl;
                        delete this->data_ptr__;
                        delete this->ref_count__;
                    }
                    this->ref_count__ = other.ref_count__;
                    this->data_ptr__ = other.data_ptr__;
                    this->ref_count__->fetch_add(1);
                }
                return *this;
            }

            /**
             * @brief assignment operator, move from another tensor
             *
             * @param other another tensor
             * @return TensorBase& reference of this tensor
             */
            TensorBase<T, Dims...>& operator=(TensorBase&& other) noexcept
            {
                //std::cout << "move assign" << std::endl;
                if (this != &other) {
                    this->ref_count__->fetch_sub(1);
                    if (this->ref_count__->load() == 0) {
                        delete this->data_ptr__;
                        delete this->ref_count__;
                    }
                    this->ref_count__ = other.ref_count__;
                    this->data_ptr__ = other.data_ptr__;
                    other.ref_count__ = nullptr;
                    other.data_ptr__ = nullptr;
                }
                return *this;
            }


            /**
             * @brief clone function, used to deepcopy a tensor
             *
             * @return TensorBase<T, Dims...>
             */
            TensorBase<T, Dims...> clone() const
            {
                TensorBase<T, Dims...> tensor;
                tensor.Array() = this->Array();
                return tensor;
            }

            /**
             * @brief DeepCopy function, used to deepcopy a tensor
             *
             * @return TensorBase<T, Dims...>
             */
            TensorBase<T, Dims...> DeepCopy() const
            {
                return this->clone();
            }

            /**
             * @brief deep copy from other tensor without change data_ptr address,
             * this is useful when the original tensor's data ptr is already registerd in somewhere else.
             * (e.g. warped in onnx runtime)
             *
             * @param other
             */
            void DeepCopy(TensorBase<T, Dims...>& other)
            {
                if ((this->data_ptr__ == other.data_ptr__) && (this->ref_count__ == other.ref_count__))
                    return;

                this->Array() = other.Array();
                return;
            }

            /**
             * @brief deep copy from other tensor without change data_ptr address,
             * this is useful when the original tensor's data ptr is already registerd in somewhere else.
             * (e.g. warped in onnx runtime)
             *
             * @param other
             */
            void clone(TensorBase<T, Dims...>& other)
            {

                this->DeepCopy(other);
                return;
            }

            /**
             * @brief compare two tensors, used to check if the given two tensors are the same tensor.
             *
             * @param other another tensor
             * @return true same
             * @return false not the same
             */
            bool same(const TensorBase& other) const
            {
                return (this->data_ptr__ == other.data_ptr__) && (this->ref_count__ == other.ref_count__);
            }

            /**
             * @brief compare two tensors, used to check if the given two tensors are the same tensor.
             *
             * @param other another tensor
             * @return true same
             * @return false not the same
             */
            bool equal(const TensorBase& other) const
            {
                return this->same(other);
            }

            /**
             * @brief convert to std::array
             *
             * @return std::array<T, Shape::total_size>& reference of data array
             */
            std::array<ValueType, Shape::total_size>& Array()
            {
                return *(this->data_ptr__);
            }

            const std::array<ValueType, Shape::total_size>& Array() const
            {
                return *(this->data_ptr__);
            }

            /**
             * @brief get total size of tensor
             *
             * @return constexpr size_t
             */
            static constexpr size_t size()
            {
                return Shape::total_size;
            }

            /**
             * @brief get shape of tensor
             *
             * @return constexpr std::array<size_t, Shape::num_dims>
             */
            static constexpr std::array<int64_t, Shape::num_dims> shape()
            {
                return Shape::dims_array;
            }

            static constexpr const int64_t* shape_ptr()
            {
                return Shape::dims_array.data();
            }

            /**
             * @brief get data pointer
             *
             * @return ValueType* the pointer of data
             */
            ValueType* data()
            {
                return this->data_ptr__->data();
            }

            /**
             * @brief get the number of dimensions
             *
             * @return constexpr size_t number of dimensions
             */
            static constexpr size_t num_dims()
            {
                return Shape::num_dims;
            }

            /**
             * @brief get data according to index, this function will ignore the shape of tensor,
             * the index is the offset in the memory.
             *
             * @param index data index
             * @return T& reference of data
             */
            T& operator[](size_t index)
            {
                return this->data_ptr__->operator[](index);
            }

            /**
             * @brief this function is a overload of operator[], it will return the data according to the index.
             *
             * @param index data index
             * @return const T& reference of data
             */
            const T& operator[](size_t index) const
            {
                return this->data_ptr__->operator[](index);
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
                return this->data_ptr__->operator[](index);
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
                return this->data_ptr__->operator[](index);
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
                return this->data_ptr__->operator[](index);
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
                return this->data_ptr__->operator[](index);
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

            std::array<T, Shape::total_size>* data_ptr__ = nullptr;

            /// @brief reference count
            std::atomic<size_t>* ref_count__ = nullptr;

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

                for (size_t i = 0;i < Shape::num_dims;i++)
                {
                    if (indices_array[i] < 0)
                        indices_array[i] += Shape::dims_array[i];
                }

                // calculate the index
                size_t index = 0;
                size_t factor = 1;

                for (int i = Shape::num_dims - 1; i >= 0; i--) {
                    //std::cout << "i: " << i << std::endl;
                    index += indices_array[i] * factor;
                    factor *= Shape::dims_array[i];

                    if (indices_array[i] >= Shape::dims_array[i] || indices_array[i] < 0)
                    {
                        throw std::out_of_range("Index out of range");
                    }
                }
                //std::cout << "index: " << index << std::endl;
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
                        os << tensor.data_ptr__->operator[](index + i) << ", ";
                    }
                    os << tensor.data_ptr__->operator[](index + Shape::dims[level] - 1);
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
        class Tensor : public TensorBase<T, Dims...>
        {
        public:
            using Base = TensorBase<T, Dims...>;
            using Shape = typename Base::Shape;
            using ValueType = typename Base::ValueType;
            using Base::Base;
            using Base::clone;
            using Base::DeepCopy;

        public:

            /**
             * @brief convert tensor to z vector
             *
             * @return Vector<ValueType, Shape::total_size> z vector
             */
            Vector<ValueType, Shape::total_size> toVector() const
            {
                return Vector<ValueType, Shape::total_size>(*(this->data_ptr__));
            }

        public: //static factory methods

            /**
             * @brief Create a tensor filled with zeros
             * @return Tensor<T, Dims...> tensor filled with zeros
             */
            static Tensor<T, Dims...> zeros()
            {
                return Tensor<T, Dims...>(T(0));
            }

            /**
             * @brief Create a tensor filled with ones
             * @return Tensor<T, Dims...> tensor filled with ones
             */
            static Tensor<T, Dims...> ones()
            {
                return Tensor<T, Dims...>(T(1));
            }

            /**
             * @brief Create a tensor filled with a specific value
             * @param value the value to fill
             * @return Tensor<T, Dims...> tensor filled with value
             */
            static Tensor<T, Dims...> full(const T& value)
            {
                return Tensor<T, Dims...>(value);
            }

            /**
             * @brief Create a tensor with random values in [0, 1)
             * @return Tensor<T, Dims...> tensor with random values
             */
            static Tensor<T, Dims...> rand()
            {
                Tensor<T, Dims...> result;
                static std::random_device rd;
                static std::mt19937 gen(rd());
                std::uniform_real_distribution<T> dis(T(0), T(1));
                for (size_t i = 0; i < result.size(); ++i)
                {
                    result[i] = dis(gen);
                }
                return result;
            }

            /**
             * @brief Create a tensor with random values from standard normal distribution
             * @param mean mean of normal distribution
             * @param std standard deviation of normal distribution
             * @return Tensor<T, Dims...> tensor with random values
             */
            static Tensor<T, Dims...> randn(T mean = T(0), T std = T(1))
            {
                Tensor<T, Dims...> result;
                static std::random_device rd;
                static std::mt19937 gen(rd());
                std::normal_distribution<T> dis(mean, std);
                for (size_t i = 0; i < result.size(); ++i)
                {
                    result[i] = dis(gen);
                }
                return result;
            }

            /**
             * @brief Create a tensor with random integer values in [low, high)
             * @param low lower bound (inclusive)
             * @param high upper bound (exclusive)
             * @return Tensor<T, Dims...> tensor with random integer values
             */
            static Tensor<T, Dims...> randint(T low, T high)
            {
                Tensor<T, Dims...> result;
                static std::random_device rd;
                static std::mt19937 gen(rd());
                std::uniform_int_distribution<T> dis(low, high - 1);
                for (size_t i = 0; i < result.size(); ++i)
                {
                    result[i] = dis(gen);
                }
                return result;
            }

            /**
             * @brief Create an empty tensor (default initialized)
             * @return Tensor<T, Dims...> empty tensor
             */
            static Tensor<T, Dims...> empty()
            {
                return Tensor<T, Dims...>();
            }

        public: //numerical operations

            /**
             * @brief operator +, used to add two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>
             */
            Tensor<ValueType, Dims...> operator+(const Tensor<ValueType, Dims...>& other) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) + other.data_ptr__->operator[](i);
                }
                return result;
            }

            /**
             * @brief operator +, used to add a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>
             */
            Tensor<ValueType, Dims...> operator+(const ValueType& other) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) + other;
                }
                return result;
            }

            /**
             * @brief operator +=, used to add two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>&
             */
            Tensor<ValueType, Dims...>& operator+=(const Tensor<ValueType, Dims...>& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    this->data_ptr__->operator[](i) += other.data_ptr__->operator[](i);
                }
                return *this;
            }

            /**
             * @brief operator +=, used to add a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>&
             */
            Tensor<ValueType, Dims...>& operator+=(const ValueType& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    this->data_ptr__->operator[](i) += other;
                }
                return *this;
            }

            /**
             * @brief operator -, used to subtract two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>
             */
            Tensor<ValueType, Dims...> operator-(const Tensor<ValueType, Dims...>& other) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) - other.data_ptr__->operator[](i);
                }
                return result;
            }

            /**
             * @brief operator -, used to subtract a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>
             */
            Tensor<ValueType, Dims...> operator-(const ValueType& other) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) - other;
                }
                return result;
            }

            /**
             * @brief operator -=, used to subtract two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>&
             */
            Tensor<ValueType, Dims...>& operator-=(const Tensor<ValueType, Dims...>& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    this->data_ptr__->operator[](i) -= other.data_ptr__->operator[](i);
                }
                return *this;
            }

            /**
             * @brief operator -=, used to subtract a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>&
             */
            Tensor<ValueType, Dims...>& operator-=(const ValueType& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    this->data_ptr__->operator[](i) -= other;
                }
                return *this;
            }

            /**
             * @brief operator *, used to multiply two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>
             */
            Tensor<ValueType, Dims...> operator*(const Tensor<ValueType, Dims...>& other) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) * other.data_ptr__->operator[](i);
                }
                return result;
            }

            /**
             * @brief operator *, used to multiply a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>
             */
            Tensor<ValueType, Dims...> operator*(const ValueType& other) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) * other;
                }
                return result;
            }

            /**
             * @brief operator *=, used to multiply two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>&
             */
            Tensor<ValueType, Dims...>& operator*=(const Tensor<ValueType, Dims...>& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    this->data_ptr__->operator[](i) *= other.data_ptr__->operator[](i);
                }
                return *this;
            }

            /**
             * @brief operator *=, used to multiply a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>&
             */
            Tensor<ValueType, Dims...>& operator*=(const ValueType& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    this->data_ptr__->operator[](i) *= other;
                }
                return *this;
            }

            /**
             * @brief operator /, used to divide two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>
             */
            Tensor<ValueType, Dims...> operator/(const Tensor<ValueType, Dims...>& other) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) / other.data_ptr__->operator[](i);
                }
                return result;
            }

            /**
             * @brief operator /, used to divide a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>
             */
            Tensor<ValueType, Dims...> operator/(const ValueType& other) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) / other;
                }
                return result;
            }

            /**
             * @brief operator /=, used to divide two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>&
             */
            Tensor<ValueType, Dims...>& operator/=(const Tensor<ValueType, Dims...>& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    this->data_ptr__->operator[](i) /= other.data_ptr__->operator[](i);
                }
                return *this;
            }

            /**
             * @brief operator /=, used to divide a tensor and a value
             *
             * @param other
             * @return Tensor<ValueType, Dims...>&
             */
            Tensor<ValueType, Dims...>& operator/=(const ValueType& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    this->data_ptr__->operator[](i) /= other;
                }
                return *this;
            }

            /**
             * @brief operator +, used to return the tensor itself
             *
             * @return Tensor<ValueType, Dims...>
             */
            Tensor<ValueType, Dims...> operator+() const
            {
                return this->clone();
            }

            /**
             * @brief operator -, used to return the negative of the tensor
             *
             * @return Tensor<ValueType, Dims...>
             */
            Tensor<ValueType, Dims...> operator-() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = -this->data_ptr__->operator[](i);
                }
                return result;
            }

            /**
             * @brief matrix multiplication (mm)
             * @details Perform matrix multiplication on two 2D tensors.
             *          this: (M, K), other: (K, N), result: (M, N)
             *
             * @tparam OtherDims dimensions of other tensor, must be (K, N)
             * @param other second matrix with shape (K, N)
             * @return Tensor<ValueType, M, N> result matrix
             */
            template<int64_t K, int64_t N>
            Tensor<ValueType, Shape::dims[0], N> mm(const Tensor<ValueType, K, N>& other) const
            {
                static_assert(Shape::num_dims == 2, "mm requires 2D tensor");
                static_assert(Shape::dims[1] == K, "Matrix dimensions mismatch for mm: (M, K) @ (K, N)");

                constexpr int64_t M = Shape::dims[0];
                Tensor<ValueType, M, N> result;

                for (int64_t i = 0; i < M; ++i)
                {
                    for (int64_t j = 0; j < N; ++j)
                    {
                        ValueType sum = 0;
                        for (int64_t k = 0; k < K; ++k)
                        {
                            sum += (*this)(i, k) * other(k, j);
                        }
                        result(i, j) = sum;
                    }
                }
                return result;
            }

            /**
             * @brief batch matrix multiplication (bmm)
             * @details Perform batch matrix multiplication on two 3D tensors.
             *          this: (B, M, K), other: (B, K, N), result: (B, M, N)
             *
             * @tparam B batch size
             * @tparam M first matrix rows (this tensor)
             * @tparam K inner dimension
             * @tparam N second matrix columns
             * @param other second batch matrix with shape (B, K, N)
             * @return Tensor<ValueType, B, M, N> result batch matrix
             */
            template<int64_t B, int64_t M, int64_t K, int64_t N>
            Tensor<ValueType, B, M, N> bmm(const Tensor<ValueType, B, K, N>& other) const
            {
                static_assert(Shape::num_dims == 3, "bmm requires 3D tensor");
                static_assert(Shape::dims[0] == B, "Batch size mismatch for bmm");
                static_assert(Shape::dims[1] == M, "First matrix rows mismatch for bmm");
                static_assert(Shape::dims[2] == K, "Matrix dimensions mismatch for bmm: (B, M, K) @ (B, K, N)");

                Tensor<ValueType, B, M, N> result;

                for (int64_t b = 0; b < B; ++b)
                {
                    for (int64_t i = 0; i < M; ++i)
                    {
                        for (int64_t j = 0; j < N; ++j)
                        {
                            ValueType sum = 0;
                            for (int64_t k = 0; k < K; ++k)
                            {
                                sum += (*this)(b, i, k) * other(b, k, j);
                            }
                            result(b, i, j) = sum;
                        }
                    }
                }
                return result;
            }

        public: //element-wise math operations

            /**
             * @brief Element-wise absolute value
             * @return Tensor<ValueType, Dims...> tensor with absolute values
             */
            Tensor<ValueType, Dims...> abs() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::abs((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise square root
             * @return Tensor<ValueType, Dims...> tensor with square root values
             */
            Tensor<ValueType, Dims...> sqrt() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::sqrt((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise cubic root
             * @return Tensor<ValueType, Dims...> tensor with cubic root values
             */
            Tensor<ValueType, Dims...> cbrt() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::cbrt((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise exponential
             * @return Tensor<ValueType, Dims...> tensor with exponential values
             */
            Tensor<ValueType, Dims...> exp() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::exp((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise 2^x
             * @return Tensor<ValueType, Dims...> tensor with 2^x values
             */
            Tensor<ValueType, Dims...> exp2() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::exp2((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise exp(x) - 1
             * @return Tensor<ValueType, Dims...> tensor with exp(x) - 1 values
             */
            Tensor<ValueType, Dims...> expm1() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::expm1((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise natural logarithm
             * @return Tensor<ValueType, Dims...> tensor with log values
             */
            Tensor<ValueType, Dims...> log() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::log((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise base-10 logarithm
             * @return Tensor<ValueType, Dims...> tensor with log10 values
             */
            Tensor<ValueType, Dims...> log10() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::log10((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise base-2 logarithm
             * @return Tensor<ValueType, Dims...> tensor with log2 values
             */
            Tensor<ValueType, Dims...> log2() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::log2((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise log(1 + x)
             * @return Tensor<ValueType, Dims...> tensor with log1p values
             */
            Tensor<ValueType, Dims...> log1p() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::log1p((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise sine
             * @return Tensor<ValueType, Dims...> tensor with sine values
             */
            Tensor<ValueType, Dims...> sin() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::sin((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise cosine
             * @return Tensor<ValueType, Dims...> tensor with cosine values
             */
            Tensor<ValueType, Dims...> cos() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::cos((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise tangent
             * @return Tensor<ValueType, Dims...> tensor with tangent values
             */
            Tensor<ValueType, Dims...> tan() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::tan((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise arcsine
             * @return Tensor<ValueType, Dims...> tensor with arcsine values
             */
            Tensor<ValueType, Dims...> asin() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::asin((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise arccosine
             * @return Tensor<ValueType, Dims...> tensor with arccosine values
             */
            Tensor<ValueType, Dims...> acos() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::acos((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise arctangent
             * @return Tensor<ValueType, Dims...> tensor with arctangent values
             */
            Tensor<ValueType, Dims...> atan() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::atan((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise hyperbolic sine
             * @return Tensor<ValueType, Dims...> tensor with hyperbolic sine values
             */
            Tensor<ValueType, Dims...> sinh() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::sinh((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise hyperbolic cosine
             * @return Tensor<ValueType, Dims...> tensor with hyperbolic cosine values
             */
            Tensor<ValueType, Dims...> cosh() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::cosh((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise hyperbolic tangent
             * @return Tensor<ValueType, Dims...> tensor with hyperbolic tangent values
             */
            Tensor<ValueType, Dims...> tanh() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::tanh((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise inverse hyperbolic sine
             * @return Tensor<ValueType, Dims...> tensor with inverse hyperbolic sine values
             */
            Tensor<ValueType, Dims...> asinh() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::asinh((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise inverse hyperbolic cosine
             * @return Tensor<ValueType, Dims...> tensor with inverse hyperbolic cosine values
             */
            Tensor<ValueType, Dims...> acosh() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::acosh((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise inverse hyperbolic tangent
             * @return Tensor<ValueType, Dims...> tensor with inverse hyperbolic tangent values
             */
            Tensor<ValueType, Dims...> atanh() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::atanh((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise floor
             * @return Tensor<ValueType, Dims...> tensor with floor values
             */
            Tensor<ValueType, Dims...> floor() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::floor((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise ceiling
             * @return Tensor<ValueType, Dims...> tensor with ceil values
             */
            Tensor<ValueType, Dims...> ceil() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::ceil((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise round to nearest
             * @return Tensor<ValueType, Dims...> tensor with round values
             */
            Tensor<ValueType, Dims...> round() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::round((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise truncation toward zero
             * @return Tensor<ValueType, Dims...> tensor with trunc values
             */
            Tensor<ValueType, Dims...> trunc() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::trunc((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise power
             * @param exponent exponent value
             * @return Tensor<ValueType, Dims...> tensor with power values
             */
            Tensor<ValueType, Dims...> pow(ValueType exponent) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::pow((*this)[i], exponent);
                }
                return result;
            }

            /**
             * @brief Element-wise error function
             * @return Tensor<ValueType, Dims...> tensor with erf values
             */
            Tensor<ValueType, Dims...> erf() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::erf((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise complementary error function
             * @return Tensor<ValueType, Dims...> tensor with erfc values
             */
            Tensor<ValueType, Dims...> erfc() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::erfc((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise gamma function
             * @return Tensor<ValueType, Dims...> tensor with tgamma values
             */
            Tensor<ValueType, Dims...> tgamma() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::tgamma((*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise log-gamma function
             * @return Tensor<ValueType, Dims...> tensor with lgamma values
             */
            Tensor<ValueType, Dims...> lgamma() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::lgamma((*this)[i]);
                }
                return result;
            }

        public: //reduction operations

            /**
             * @brief Sum all elements of tensor
             * @return ValueType sum of all elements
             */
            ValueType sum() const
            {
                ValueType result = ValueType(0);
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result += (*this)[i];
                }
                return result;
            }

            /**
             * @brief Compute product of all elements
             * @return ValueType product of all elements
             */
            ValueType prod() const
            {
                ValueType result = ValueType(1);
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result *= (*this)[i];
                }
                return result;
            }

            /**
             * @brief Compute mean of all elements
             * @return ValueType mean of all elements
             */
            ValueType mean() const
            {
                return sum() / static_cast<ValueType>(Shape::total_size);
            }

            /**
             * @brief Compute variance of all elements
             * @return ValueType variance of all elements
             */
            ValueType var() const
            {
                ValueType m = mean();
                ValueType result = ValueType(0);
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    ValueType diff = (*this)[i] - m;
                    result += diff * diff;
                }
                return result / static_cast<ValueType>(Shape::total_size);
            }

            /**
             * @brief Compute standard deviation of all elements
             * @return ValueType standard deviation of all elements
             */
            ValueType stddev() const
            {
                return std::sqrt(var());
            }

            /**
             * @brief Compute minimum element
             * @return ValueType minimum element
             */
            ValueType min() const
            {
                ValueType result = (*this)[0];
                for (size_t i = 1; i < Shape::total_size; ++i)
                {
                    if ((*this)[i] < result)
                        result = (*this)[i];
                }
                return result;
            }

            /**
             * @brief Compute maximum element
             * @return ValueType maximum element
             */
            ValueType max() const
            {
                ValueType result = (*this)[0];
                for (size_t i = 1; i < Shape::total_size; ++i)
                {
                    if ((*this)[i] > result)
                        result = (*this)[i];
                }
                return result;
            }

            /**
             * @brief Element-wise square
             * @return Tensor<ValueType, Dims...> tensor with squared values
             */
            Tensor<ValueType, Dims...> square() const
            {
                return (*this) * (*this);
            }

            /**
             * @brief Element-wise reciprocal (1/x)
             * @return Tensor<ValueType, Dims...> tensor with reciprocal values
             */
            Tensor<ValueType, Dims...> reciprocal() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = ValueType(1) / (*this)[i];
                }
                return result;
            }

            /**
             * @brief Element-wise sign function
             * @return Tensor<ValueType, Dims...> tensor with sign values (-1, 0, 1)
             */
            Tensor<ValueType, Dims...> sign() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = ((*this)[i] > ValueType(0)) ? ValueType(1) : (((*this)[i] < ValueType(0)) ? ValueType(-1) : ValueType(0));
                }
                return result;
            }

            /**
             * @brief Clamp tensor values to a range [min_val, max_val]
             * @param min_val minimum value
             * @param max_val maximum value
             * @return Tensor<ValueType, Dims...> clamped tensor
             */
            Tensor<ValueType, Dims...> clamp(ValueType min_val, ValueType max_val) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::max(min_val, std::min((*this)[i], max_val));
                }
                return result;
            }

            /**
             * @brief Element-wise clamp between 0 and 1
             * @return Tensor<ValueType, Dims...> tensor with values in [0, 1]
             */
            Tensor<ValueType, Dims...> clamp01() const
            {
                return clamp(ValueType(0), ValueType(1));
            }

        public: //activation functions

            /**
             * @brief Element-wise ReLU activation max(0, x)
             * @return Tensor<ValueType, Dims...> tensor with ReLU values
             */
            Tensor<ValueType, Dims...> relu() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::max((*this)[i], ValueType(0));
                }
                return result;
            }

            /**
             * @brief Element-wise leaky ReLU activation
             * @param negative_slope slope for x < 0
             * @return Tensor<ValueType, Dims...> tensor with leaky ReLU values
             */
            Tensor<ValueType, Dims...> leaky_relu(ValueType negative_slope = ValueType(0.01)) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = ((*this)[i] >= ValueType(0)) ? (*this)[i] : (negative_slope * (*this)[i]);
                }
                return result;
            }

            /**
             * @brief Element-wise sigmoid activation 1 / (1 + exp(-x))
             * @return Tensor<ValueType, Dims...> tensor with sigmoid values
             */
            Tensor<ValueType, Dims...> sigmoid() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = ValueType(1) / (ValueType(1) + std::exp(-(*this)[i]));
                }
                return result;
            }

            /**
             * @brief Element-wise softplus activation ln(1 + exp(x))
             * @return Tensor<ValueType, Dims...> tensor with softplus values
             */
            Tensor<ValueType, Dims...> softplus() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::log(ValueType(1) + std::exp((*this)[i]));
                }
                return result;
            }

            /**
             * @brief Element-wise ELU activation
             * @param alpha scale for x < 0
             * @return Tensor<ValueType, Dims...> tensor with ELU values
             */
            Tensor<ValueType, Dims...> elu(ValueType alpha = ValueType(1)) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = ((*this)[i] >= ValueType(0)) ? (*this)[i] : (alpha * (std::exp((*this)[i]) - ValueType(1)));
                }
                return result;
            }

            /**
             * @brief Element-wise SELU activation
             * @param lambda scale for all x
             * @param alpha scale for x < 0
             * @return Tensor<ValueType, Dims...> tensor with SELU values
             */
            Tensor<ValueType, Dims...> selu(ValueType lambda = ValueType(1.0507009873554805),
                ValueType alpha = ValueType(1.6732632423543772)) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    ValueType v = ((*this)[i] >= ValueType(0)) ? (*this)[i] : (alpha * (std::exp((*this)[i]) - ValueType(1)));
                    result[i] = lambda * v;
                }
                return result;
            }

            /**
             * @brief Element-wise GELU activation
             * @return Tensor<ValueType, Dims...> tensor with GELU values
             */
            Tensor<ValueType, Dims...> gelu() const
            {
                Tensor<ValueType, Dims...> result;
                constexpr ValueType k0 = ValueType(0.5);
                constexpr ValueType k1 = ValueType(0.7978845608028654); // sqrt(2/pi)
                constexpr ValueType k2 = ValueType(0.044715);
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    ValueType x = (*this)[i];
                    ValueType inner = k1 * (x + k2 * x * x * x);
                    result[i] = k0 * x * (ValueType(1) + std::tanh(inner));
                }
                return result;
            }

            /**
             * @brief Element-wise Swish activation x * sigmoid(x)
             * @return Tensor<ValueType, Dims...> tensor with Swish values
             */
            Tensor<ValueType, Dims...> swish() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    ValueType x = (*this)[i];
                    result[i] = x / (ValueType(1) + std::exp(-x));
                }
                return result;
            }

            /**
             * @brief Element-wise Mish activation x * tanh(softplus(x))
             * @return Tensor<ValueType, Dims...> tensor with Mish values
             */
            Tensor<ValueType, Dims...> mish() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    ValueType x = (*this)[i];
                    result[i] = x * std::tanh(std::log(ValueType(1) + std::exp(x)));
                }
                return result;
            }

            /**
             * @brief Element-wise softsign activation x / (1 + |x|)
             * @return Tensor<ValueType, Dims...> tensor with softsign values
             */
            Tensor<ValueType, Dims...> softsign() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    ValueType x = (*this)[i];
                    result[i] = x / (ValueType(1) + std::abs(x));
                }
                return result;
            }

            /**
             * @brief Element-wise hard sigmoid activation
             * @param slope slope for linear region
             * @param offset offset for linear region
             * @return Tensor<ValueType, Dims...> tensor with hard sigmoid values
             */
            Tensor<ValueType, Dims...> hard_sigmoid(ValueType slope = ValueType(0.2),
                ValueType offset = ValueType(0.5)) const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = std::clamp(slope * (*this)[i] + offset, ValueType(0), ValueType(1));
                }
                return result;
            }

            /**
             * @brief Element-wise hard swish activation x * relu6(x + 3) / 6
             * @return Tensor<ValueType, Dims...> tensor with hard swish values
             */
            Tensor<ValueType, Dims...> hard_swish() const
            {
                Tensor<ValueType, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    ValueType x = (*this)[i];
                    ValueType relu6 = std::clamp(x + ValueType(3), ValueType(0), ValueType(6));
                    result[i] = x * relu6 / ValueType(6);
                }
                return result;
            }

            /**
             * @brief Cast tensor to another type
             *
             * @tparam Scalar target type
             * @tparam T source type
             * @tparam Dims dimensions of tensor
             * @param t input tensor
             * @return Tensor<Scalar, Dims...> casted tensor
             */
            template<typename Scalar>
            Tensor<Scalar, Dims...> cast() const
            {
                static_assert(std::is_arithmetic_v<Scalar>, "Scalar must be an arithmetic type");
                static_assert(std::is_convertible_v<ValueType, Scalar>, "T must be convertible to Scalar");
                Tensor<Scalar, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = static_cast<Scalar>(this->data_ptr__->operator[](i));
                }
                return result;
            }

            /**
             * @brief Alias for cast function
             *
             * @tparam Scalar target type
             * @tparam T source type
             * @tparam Dims dimensions of tensor
             * @param t input tensor
             * @return Tensor<Scalar, Dims...> casted tensor
             */
            template<typename Scalar>
            Tensor<Scalar, Dims...> to()
            {
                return this->cast<Scalar>();
            }

        public: //logical operations

            /**
             * @brief operator >, used to compare two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator>(const Tensor<ValueType, Dims...>& other) const
            {
                if (this->equal(other))
                {
                    return Tensor<bool, Dims...>(false);
                }

                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) > other.data_ptr__->operator[](i));
                }
                return result;
            };

            /**
             * @brief operator >, used to compare a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator>(const ValueType& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) > other);
                }
                return result;
            };

            /**
             * @brief operator >=, used to compare two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator>=(const Tensor<ValueType, Dims...>& other) const
            {
                if (this->equal(other))
                {
                    return Tensor<bool, Dims...>(true);
                }

                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) >= other.data_ptr__->operator[](i));
                }
                return result;
            };

            /**
             * @brief operator >=, used to compare a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator>=(const ValueType& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) >= other);
                }
                return result;
            };

            /**
             * @brief operator <, used to compare two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator<(const Tensor<ValueType, Dims...>& other) const
            {
                if (this->equal(other))
                {
                    return Tensor<bool, Dims...>(false);
                }

                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) < other.data_ptr__->operator[](i));
                }
                return result;
            };

            /**
             * @brief operator <, used to compare a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator<(const ValueType& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) < other);
                }
                return result;
            };

            /**
             * @brief operator <=, used to compare two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator<=(const Tensor<ValueType, Dims...>& other) const
            {
                if (this->equal(other))
                {
                    return Tensor<bool, Dims...>(true);
                }
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) <= other.data_ptr__->operator[](i));
                }
                return result;
            };

            /**
             * @brief operator <=, used to compare a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator<=(const ValueType& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) <= other);
                }
                return result;
            };

            /**
             * @brief operator ==, used to compare two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator==(const Tensor<ValueType, Dims...>& other) const
            {
                if (this->equal(other))
                {
                    return Tensor<bool, Dims...>(true);
                }
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) == other.data_ptr__->operator[](i));
                }
                return result;
            };

            /**
             * @brief operator ==, used to compare a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator==(const ValueType& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) == other);
                }
                return result;
            };

            /**
             * @brief operator !=, used to compare two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator!=(const Tensor<ValueType, Dims...>& other) const
            {
                if (this->equal(other))
                {
                    return Tensor<bool, Dims...>(false);
                }
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) != other.data_ptr__->operator[](i));
                }
                return result;
            };

            /**
             * @brief operator !=, used to compare a tensor and a value
             *
             * @param other value
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator!=(const ValueType& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) != other);
                }
                return result;
            };
        };


        /**********************************bool tensor*****************************/

        /**
         * @brief Bool Tensor class, used to store bool type tensor data
         *
         * @tparam T type of tensor element
         * @tparam Dims
         */
        template<int64_t... Dims>
        class Tensor<bool, Dims...> : public TensorBase<bool, Dims...> {
        public:
            using Base = TensorBase<bool, Dims...>;
            using Shape = typename Base::Shape;
            using ValueType = typename Base::ValueType;
            using Base::Base;
            using Base::clone;
            using Base::DeepCopy;

        public:

            /**
             * @brief convert tensor to z vector
             *
             * @return Vector<ValueType, Shape::total_size> z vector
             */
            Vector<ValueType, Shape::total_size> toVector() const
            {
                return Vector<ValueType, Shape::total_size>(*(this->data_ptr__));
            }

        public: //static factory methods

            /**
             * @brief Create a bool tensor filled with false
             * @return Tensor<bool, Dims...> tensor filled with false
             */
            static Tensor<bool, Dims...> zeros()
            {
                return Tensor<bool, Dims...>(false);
            }

            /**
             * @brief Create a bool tensor filled with true
             * @return Tensor<bool, Dims...> tensor filled with true
             */
            static Tensor<bool, Dims...> ones()
            {
                return Tensor<bool, Dims...>(true);
            }

            /**
             * @brief Create a bool tensor filled with a specific value
             * @param value the value to fill
             * @return Tensor<bool, Dims...> tensor filled with value
             */
            static Tensor<bool, Dims...> full(bool value)
            {
                return Tensor<bool, Dims...>(value);
            }

            /**
             * @brief Create a bool tensor with random values
             * @return Tensor<bool, Dims...> tensor with random bool values
             */
            static Tensor<bool, Dims...> rand()
            {
                Tensor<bool, Dims...> result;
                static std::random_device rd;
                static std::mt19937 gen(rd());
                std::uniform_int_distribution<int> dis(0, 1);
                for (size_t i = 0; i < result.size(); ++i)
                {
                    result[i] = dis(gen) == 1;
                }
                return result;
            }

            /**
             * @brief Create an empty tensor (default initialized to false)
             * @return Tensor<bool, Dims...> empty tensor
             */
            static Tensor<bool, Dims...> empty()
            {
                return Tensor<bool, Dims...>();
            }

        public: //numerical operations
            /**
             * @brief operator +, used to add two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator+(const Tensor<bool, Dims...>& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) || other.data_ptr__->operator[](i);
                }
                return result;
            }

            /**
             * @brief operator +, used to add a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator+(const bool& other) const
            {
                if (other == false)
                {
                    return this->clone();
                }
                else//(other == true)
                {
                    return Tensor<bool, Dims...>(true);
                }
            }

            /**
             * @brief operator +=, used to add two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>&
             */
            Tensor<bool, Dims...>& operator+=(const Tensor<bool, Dims...>& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    this->data_ptr__->operator[](i) = this->data_ptr__->operator[](i) || other.data_ptr__->operator[](i);
                }
                return *this;
            }

            /**
             * @brief operator +=, used to add a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>&
             */
            Tensor<bool, Dims...>& operator+=(const bool& other)
            {
                if (other == false)
                {
                    return *this;
                }
                else//(other == true)
                {
                    for (size_t i = 0; i < Shape::total_size; i++)
                    {
                        this->data_ptr__->operator[](i) = true;
                    }
                    return *this;
                }
            }

            //- xor
            /**
             * @brief operator -, used to subtract two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator-(const Tensor<bool, Dims...>& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    //xor
                    result[i] = (!this->data_ptr__->operator[](i) && other.data_ptr__->operator[](i)) ||
                        (this->data_ptr__->operator[](i) && !other.data_ptr__->operator[](i));
                }
                return result;
            }

            /**
             * @brief operator -, used to subtract a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator-(const bool& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    //xor
                    result[i] = (!this->data_ptr__->operator[](i) && other) ||
                        (this->data_ptr__->operator[](i) && !other);
                }
                return result;
            }

            /**
             * @brief operator -=, used to subtract two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>&
             */
            Tensor<bool, Dims...>& operator-=(const Tensor<bool, Dims...>& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    //xor
                    this->data_ptr__->operator[](i) = (!this->data_ptr__->operator[](i) && other.data_ptr__->operator[](i)) ||
                        (this->data_ptr__->operator[](i) && !other.data_ptr__->operator[](i));
                }
                return *this;
            }

            /**
             * @brief operator -=, used to subtract a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>&
             */
            Tensor<bool, Dims...>& operator-=(const bool& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    //xor
                    this->data_ptr__->operator[](i) = (!this->data_ptr__->operator[](i) && other) ||
                        (this->data_ptr__->operator[](i) && !other);
                }
                return *this;
            }

            //* -> and
            /**
             * @brief operator *, used to multiply two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator*(const Tensor<bool, Dims...>& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) && other.data_ptr__->operator[](i);
                }
                return result;
            }

            /**
             * @brief operator *, used to multiply a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator*(const bool& other) const
            {
                if (other == false)
                {
                    return Tensor<bool, Dims...>(false);
                }
                else//(other == true)
                {
                    return this->clone();
                }
            }

            /**
             * @brief operator *=, used to multiply two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>&
             */
            Tensor<bool, Dims...>& operator*=(const Tensor<bool, Dims...>& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    this->data_ptr__->operator[](i) = this->data_ptr__->operator[](i) && other.data_ptr__->operator[](i);
                }
                return *this;
            }

            /**
             * @brief operator *=, used to multiply a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>&
             */
            Tensor<bool, Dims...>& operator*=(const bool& other)
            {
                if (other == false)
                {
                    for (size_t i = 0; i < Shape::total_size; i++)
                    {
                        this->data_ptr__->operator[](i) = false;
                    }
                    return *this;
                }
                else//(other == true)
                {
                    return *this;
                }
            }

            // /->nxor
            /**
             * @brief operator /, used to divide two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator/(const Tensor<bool, Dims...>& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    //nxor
                    result[i] = (this->data_ptr__->operator[](i) == other.data_ptr__->operator[](i)) ||
                        (!this->data_ptr__->operator[](i) && !other.data_ptr__->operator[](i));
                }
                return result;
            }

            /**
             * @brief operator /, used to divide a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator/(const bool& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    //nxor
                    result[i] = (this->data_ptr__->operator[](i) == other) ||
                        (!this->data_ptr__->operator[](i) && !other);
                }
                return result;
            }

            /**
             * @brief operator /=, used to divide two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>&
             */
            Tensor<bool, Dims...>& operator/=(const Tensor<bool, Dims...>& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    //nxor
                    this->data_ptr__->operator[](i) = (this->data_ptr__->operator[](i) == other.data_ptr__->operator[](i)) ||
                        (!this->data_ptr__->operator[](i) && !other.data_ptr__->operator[](i));
                }
                return *this;
            }

            /**
             * @brief operator /=, used to divide a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>&
             */
            Tensor<bool, Dims...>& operator/=(const bool& other)
            {
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    //nxor
                    this->data_ptr__->operator[](i) = (this->data_ptr__->operator[](i) == other) ||
                        (!this->data_ptr__->operator[](i) && !other);
                }
                return *this;
            }

            // - -> !
            /**
             * @brief operator -, used to return the negative of the tensor
             *
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator-() const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = !this->data_ptr__->operator[](i);
                }
                return result;
            }

            //+ -> this
            /**
             * @brief operator +, used to return the tensor itself
             *
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator+() const
            {
                return this->clone();
            }

            /**
             * @brief Cast tensor to another type
             *
             * @tparam Scalar target type
             * @tparam T source type
             * @tparam Dims dimensions of tensor
             * @param t input tensor
             * @return Tensor<Scalar, Dims...> casted tensor
             */
            template<typename Scalar>
            Tensor<Scalar, Dims...> cast() const
            {
                static_assert(std::is_arithmetic_v<Scalar>, "Scalar must be an arithmetic type");
                static_assert(std::is_convertible_v<ValueType, Scalar>, "T must be convertible to Scalar");
                Tensor<Scalar, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; ++i)
                {
                    result[i] = static_cast<Scalar>(this->data_ptr__->operator[](i));
                }
                return result;
            }

            /**
             * @brief Alias for cast function
             *
             * @tparam Scalar target type
             * @tparam T source type
             * @tparam Dims dimensions of tensor
             * @param t input tensor
             * @return Tensor<Scalar, Dims...> casted tensor
             */
            template<typename Scalar>
            Tensor<Scalar, Dims...> to()
            {
                return this->cast<Scalar>();
            }


        public: //logical operations
            /**
             * @brief operator &&, used to compare two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator&&(const Tensor<bool, Dims...>& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) && other.data_ptr__->operator[](i);
                }
                return result;
            };

            /**
             * @brief operator &&, used to compare a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator&&(const bool& other) const
            {
                if (other == false)
                {
                    return Tensor<bool, Dims...>(false);
                }
                else//(other == true)
                {
                    return this->clone();
                }
            }

            /**
             * @brief operator ||, used to compare two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator||(const Tensor<bool, Dims...>& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = this->data_ptr__->operator[](i) || other.data_ptr__->operator[](i);
                }
                return result;
            };

            /**
             * @brief operator ||, used to compare a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator||(const bool& other) const
            {
                if (other == false)
                {
                    return this->clone();
                }
                else//(other == true)
                {
                    return Tensor<bool, Dims...>(true);
                }
            }

            /**
             * @brief operator !, used to compare a tensor and a value
             *
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator!() const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = !this->data_ptr__->operator[](i);
                }
                return result;
            };

        public: //logical operation == !=
            /**
             * @brief operator ==, used to compare two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator==(const Tensor<bool, Dims...>& other) const
            {
                if (this->equal(other))
                {
                    return Tensor<bool, Dims...>(true);
                }
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) == other.data_ptr__->operator[](i));
                }
                return result;
            };

            /**
             * @brief operator ==, used to compare a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator==(const bool& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) == other);
                }
                return result;
            };

            /**
             * @brief operator !=, used to compare two tensors or a tensor and a value
             *
             * @param other
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator!=(const Tensor<bool, Dims...>& other) const
            {
                if (this->equal(other))
                {
                    return Tensor<bool, Dims...>(false);
                }
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) != other.data_ptr__->operator[](i));
                }
                return result;
            };

            /**
             * @brief operator !=, used to compare a tensor and a value
             *
             * @param other value
             * @return Tensor<bool, Dims...>
             */
            Tensor<bool, Dims...> operator!=(const bool& other) const
            {
                Tensor<bool, Dims...> result;
                for (size_t i = 0; i < Shape::total_size; i++)
                {
                    result[i] = (this->data_ptr__->operator[](i) != other);
                }
                return result;
            }
        };

        /************************** utility functions **************************/

        /**
         * @brief Create a 1D tensor with values from start to stop with step
         *
         * @tparam T type of tensor element
         * @tparam N size of tensor
         * @param start start value
         * @param step step size
         * @return Tensor<T, N> 1D tensor with arange values
         */
        template<typename T, int64_t N>
        Tensor<T, N> arange(T start = T(0), T step = T(1))
        {
            Tensor<T, N> result;
            for (int64_t i = 0; i < N; ++i)
            {
                result[i] = start + static_cast<T>(i) * step;
            }
            return result;
        }

        /**
         * @brief Create a 1D tensor with evenly spaced values from start to end
         *
         * @tparam T type of tensor element
         * @tparam N size of tensor
         * @param start start value
         * @param end end value
         * @return Tensor<T, N> 1D tensor with linspace values
         */
        template<typename T, int64_t N>
        Tensor<T, N> linspace(T start, T end)
        {
            Tensor<T, N> result;
            if (N == 1)
            {
                result[0] = start;
                return result;
            }
            T step = (end - start) / static_cast<T>(N - 1);
            for (int64_t i = 0; i < N; ++i)
            {
                result[i] = start + static_cast<T>(i) * step;
            }
            return result;
        }

        /**
         * @brief Create a 2D identity matrix
         *
         * @tparam T type of tensor element
         * @tparam N size of square matrix
         * @return Tensor<T, N, N> identity matrix
         */
        template<typename T, int64_t N>
        Tensor<T, N, N> eye()
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
         *
         * @tparam T type of tensor element
         * @tparam N size of square matrix
         * @return Tensor<T, N, N> identity matrix
         */
        template<typename T, int64_t N>
        Tensor<T, N, N> Identity()
        {
            return eye<T, N>();
        }

        /**
         * @brief Create a diagonal matrix from a 1D tensor
         *
         * @tparam T type of tensor element
         * @tparam N size of diagonal
         * @param vec 1D tensor containing diagonal elements
         * @return Tensor<T, N, N> diagonal matrix
         */
        template<typename T, int64_t N>
        Tensor<T, N, N> diag(const Tensor<T, N>& vec)
        {
            Tensor<T, N, N> result(T(0));
            for (int64_t i = 0; i < N; ++i)
            {
                result(i, i) = vec[i];
            }
            return result;
        }

        /**
         * @brief Extract diagonal elements from a 2D tensor
         *
         * @tparam T type of tensor element
         * @tparam N size of square matrix
         * @param mat 2D tensor (square matrix)
         * @return Tensor<T, N> 1D tensor containing diagonal elements
         */
        template<typename T, int64_t N>
        Tensor<T, N> diag(const Tensor<T, N, N>& mat)
        {
            Tensor<T, N> result;
            for (int64_t i = 0; i < N; ++i)
            {
                result[i] = mat(i, i);
            }
            return result;
        }

        /**
         * @brief Compute the transpose of a 2D tensor
         *
         * @tparam T type of tensor element
         * @tparam M number of rows
         * @tparam N number of columns
         * @param mat input matrix
         * @return Tensor<T, N, M> transposed matrix
         */
        template<typename T, int64_t M, int64_t N>
        Tensor<T, N, M> transpose(const Tensor<T, M, N>& mat)
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

        /**
         * @brief Element-wise absolute value
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with absolute values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> abs(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with square root values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> sqrt(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::sqrt(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise exponential
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with exponential values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> exp(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::exp(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise natural logarithm
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with log values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> log(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::log(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise power
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param exponent exponent value
         * @return Tensor<T, Dims...> tensor with power values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> pow(const Tensor<T, Dims...>& t, T exponent)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::pow(t[i], exponent);
            }
            return result;
        }

        /**
         * @brief Element-wise sine
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with sine values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> sin(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with cosine values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> cos(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with tangent values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> tan(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::tan(t[i]);
            }
            return result;
        }

        /**
         * @brief Sum all elements of tensor
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T sum of all elements
         */
        template<typename T, int64_t... Dims>
        T sum(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T mean of all elements
         */
        template<typename T, int64_t... Dims>
        T mean(const Tensor<T, Dims...>& t)
        {
            return sum(t) / static_cast<T>(t.size());
        }

        /**
         * @brief Compute minimum element
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T minimum element
         */
        template<typename T, int64_t... Dims>
        T min(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T maximum element
         */
        template<typename T, int64_t... Dims>
        T max(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param min_val minimum value
         * @param max_val maximum value
         * @return Tensor<T, Dims...> clamped tensor
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> clamp(const Tensor<T, Dims...>& t, T min_val, T max_val)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::max(min_val, std::min(t[i], max_val));
            }
            return result;
        }

        /**
         * @brief Element-wise clamp between 0 and 1
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with values in [0, 1]
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> clamp01(const Tensor<T, Dims...>& t)
        {
            return clamp(t, T(0), T(1));
        }

        /**
         * @brief Element-wise minimum of two tensors
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param a first tensor
         * @param b second tensor
         * @return Tensor<T, Dims...> tensor with minimum values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> min(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param val scalar value
         * @return Tensor<T, Dims...> tensor with minimum values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> min(const Tensor<T, Dims...>& t, T val)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param a first tensor
         * @param b second tensor
         * @return Tensor<T, Dims...> tensor with maximum values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> max(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param val scalar value
         * @return Tensor<T, Dims...> tensor with maximum values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> max(const Tensor<T, Dims...>& t, T val)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::max(t[i], val);
            }
            return result;
        }

        /**
         * @brief Element-wise cubic root
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with cubic root values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> cbrt(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::cbrt(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise arcsine
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with arcsine values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> asin(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with arccosine values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> acos(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with arctangent values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> atan(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param y first tensor
         * @param x second tensor
         * @return Tensor<T, Dims...> tensor with arctangent values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> atan2(const Tensor<T, Dims...>& y, const Tensor<T, Dims...>& x)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with hyperbolic sine values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> sinh(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with hyperbolic cosine values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> cosh(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with hyperbolic tangent values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> tanh(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with inverse hyperbolic sine values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> asinh(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with inverse hyperbolic cosine values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> acosh(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with inverse hyperbolic tangent values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> atanh(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::atanh(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise 2^x
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with 2^x values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> exp2(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with exp(x) - 1 values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> expm1(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::expm1(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise base-10 logarithm
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with log10 values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> log10(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with log2 values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> log2(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with log1p values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> log1p(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::log1p(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise floor
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with floor values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> floor(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with ceil values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> ceil(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with round values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> round(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with trunc values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> trunc(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param x dividend tensor
         * @param y divisor tensor
         * @return Tensor<T, Dims...> tensor with fmod values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> fmod(const Tensor<T, Dims...>& x, const Tensor<T, Dims...>& y)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param x dividend tensor
         * @param y divisor scalar
         * @return Tensor<T, Dims...> tensor with fmod values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> fmod(const Tensor<T, Dims...>& x, T y)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param x first tensor
         * @param y second tensor
         * @return Tensor<T, Dims...> tensor with hypot values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> hypot(const Tensor<T, Dims...>& x, const Tensor<T, Dims...>& y)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with erf values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> erf(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with erfc values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> erfc(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with tgamma values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> tgamma(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with lgamma values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> lgamma(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = std::lgamma(t[i]);
            }
            return result;
        }

        /**
         * @brief Element-wise power with tensor-tensor
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param base base tensor
         * @param exp exponent tensor
         * @return Tensor<T, Dims...> tensor with power values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> pow(const Tensor<T, Dims...>& base, const Tensor<T, Dims...>& exp)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < base.size(); ++i)
            {
                result[i] = std::pow(base[i], exp[i]);
            }
            return result;
        }

        /************************** activation functions **************************/

        /**
         * @brief Element-wise ReLU activation max(0, x)
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with ReLU values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> relu(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param negative_slope slope for x < 0
         * @return Tensor<T, Dims...> tensor with leaky ReLU values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> leaky_relu(const Tensor<T, Dims...>& t, T negative_slope = T(0.01))
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with sigmoid values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> sigmoid(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with softplus values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> softplus(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param alpha scale for x < 0
         * @return Tensor<T, Dims...> tensor with ELU values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> elu(const Tensor<T, Dims...>& t, T alpha = T(1))
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param lambda scale for all x
         * @param alpha scale for x < 0
         * @return Tensor<T, Dims...> tensor with SELU values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> selu(const Tensor<T, Dims...>& t,
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with GELU values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> gelu(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with Swish values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> swish(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with Mish values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> mish(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with softsign values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> softsign(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @param slope slope for linear region
         * @param offset offset for linear region
         * @return Tensor<T, Dims...> tensor with hard sigmoid values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> hard_sigmoid(const Tensor<T, Dims...>& t,
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with hard swish values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> hard_swish(const Tensor<T, Dims...>& t)
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

        /************************** utility functions **************************/

        /**
         * @brief Check if all elements are true
         *
         * @tparam Dims dimensions of tensor
         * @param t input bool tensor
         * @return true if all elements are true
         * @return false otherwise
         */
        template<int64_t... Dims>
        bool all(const Tensor<bool, Dims...>& t)
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
         *
         * @tparam Dims dimensions of tensor
         * @param t input bool tensor
         * @return true if any element is true
         * @return false otherwise
         */
        template<int64_t... Dims>
        bool any(const Tensor<bool, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param cond condition tensor
         * @param x values where condition is true
         * @param y values where condition is false
         * @return Tensor<T, Dims...> selected values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> where(const Tensor<bool, Dims...>& cond,
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T product of all elements
         */
        template<typename T, int64_t... Dims>
        T prod(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T variance of all elements
         */
        template<typename T, int64_t... Dims>
        T var(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return T standard deviation of all elements
         */
        template<typename T, int64_t... Dims>
        T stddev(const Tensor<T, Dims...>& t)
        {
            return std::sqrt(var(t));
        }

        /**
         * @brief Element-wise square
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with squared values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> square(const Tensor<T, Dims...>& t)
        {
            return t * t;
        }

        /**
         * @brief Element-wise reciprocal (1/x)
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with reciprocal values
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> reciprocal(const Tensor<T, Dims...>& t)
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
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param t input tensor
         * @return Tensor<T, Dims...> tensor with sign values (-1, 0, 1)
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> sign(const Tensor<T, Dims...>& t)
        {
            Tensor<T, Dims...> result;
            for (size_t i = 0; i < t.size(); ++i)
            {
                result[i] = (t[i] > T(0)) ? T(1) : ((t[i] < T(0)) ? T(-1) : T(0));
            }
            return result;
        }

        /**
         * @brief Create a tensor filled with a specific value (alias for full)
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @param value the value to fill
         * @return Tensor<T, Dims...> tensor filled with value
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> fill(const T& value)
        {
            return full<T, Dims...>(value);
        }

        /**
         * @brief Create an empty tensor (uninitialized, for performance)
         * Note: Elements are default-initialized, use with caution
         *
         * @tparam T type of tensor element
         * @tparam Dims dimensions of tensor
         * @return Tensor<T, Dims...> empty tensor
         */
        template<typename T, int64_t... Dims>
        Tensor<T, Dims...> empty()
        {
            return Tensor<T, Dims...>();
        }

        /************************** free function bmm **************************/

        /**
         * @brief batch matrix multiplication (bmm) - free function version
         * @details Perform batch matrix multiplication on two 3D tensors.
         *          a: (B, M, K), b: (B, K, N), result: (B, M, N)
         *          Usage: auto result = bmm(a, b); // automatic type deduction
         *
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
        Tensor<T, B, M, N> bmm(const Tensor<T, B, M, K>& a, const Tensor<T, B, K, N>& b)
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
         *
         * @tparam T type of tensor element
         * @tparam M first matrix rows
         * @tparam K inner dimension
         * @tparam N second matrix columns
         * @param a first matrix with shape (M, K)
         * @param b second matrix with shape (K, N)
         * @return Tensor<T, M, N> result matrix
         */
        template<typename T, int64_t M, int64_t K, int64_t N>
        Tensor<T, M, N> mm(const Tensor<T, M, K>& a, const Tensor<T, K, N>& b)
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
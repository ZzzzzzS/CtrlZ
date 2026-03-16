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
             * @brief calculate the index according to the indices (compile-time supported)
             *
             * @tparam Indices
             * @param indices
             * @return constexpr size_t
             */
            template<typename... Indices>
            static constexpr size_t calculate_index(Indices... indices) {
                static_assert(sizeof...(Indices) == Shape::num_dims, 
                              "Number of indices must match number of dimensions");
                return calculate_index_impl(std::make_index_sequence<Shape::num_dims>{}, 
                                            static_cast<int64_t>(indices)...);
            }

        private:
            /**
             * @brief Implementation of calculate_index with compile-time support
             */
            template<size_t... Is, typename... Indices>
            static constexpr size_t calculate_index_impl(std::index_sequence<Is...>, Indices... indices) {
                std::array<int64_t, Shape::num_dims> indices_array = { indices... };

                // Handle negative indices
                for (size_t i = 0; i < Shape::num_dims; i++) {
                    if (indices_array[i] < 0)
                        indices_array[i] += Shape::dims_array[i];
                }

                // Runtime bounds check (only when not in constant evaluation)
                if (!std::is_constant_evaluated()) {
                    for (size_t i = 0; i < Shape::num_dims; i++) {
                        if (indices_array[i] >= Shape::dims_array[i] || indices_array[i] < 0) {
                            throw std::out_of_range("Index out of range");
                        }
                    }
                }

                // Calculate strides for each dimension
                size_t strides[Shape::num_dims] = {};
                size_t factor = 1;
                for (int i = Shape::num_dims - 1; i >= 0; --i) {
                    strides[i] = factor;
                    factor *= Shape::dims_array[i];
                }

                // Calculate the final index
                size_t index = 0;
                ((index += static_cast<size_t>(indices_array[Is]) * strides[Is]), ...);

                return index;
            }

        protected:

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
                thread_local static std::random_device rd;
                thread_local static std::mt19937 gen(rd());
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

        // Free math functions are now in MathFunction.hpp
    }; // namespace math
}; // namespace z
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
    };
};
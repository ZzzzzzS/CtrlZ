#pragma once
#include <array>
#include <memory>
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
            TensorBase() {
                this->ref_count__ = new std::atomic<size_t>(1);
                this->data_ptr__ = new std::array<ValueType, Shape::total_size>();
                this->data_ptr__->fill(ValueType());
            }

            ~TensorBase()
            {
                std::cout << "ref_count=" << *(this->ref_count__) << std::endl;
                if (--(*this->ref_count__) == 0) {
                    delete this->data_ptr__;
                    delete this->ref_count__;
                }
            }

            /**
             * @brief Construct a new Tensor Base object with all elements set to given value
             *
             */
            TensorBase(const T& val) {
                this->ref_count__ = new size_t(1);
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
                this->ref_count__ = new size_t(1);
                this->data_ptr__ = new std::array<ValueType, Shape::total_size>(data);
            }

            /**
             * @brief Construct a new Tensor Base object, move from std::array
             *
             * @param data an array of data
             */
            TensorBase(std::array<ValueType, Shape::total_size>&& data)
            {
                this->ref_count__ = new size_t(1);
                this->data_ptr__ = new std::array<ValueType, Shape::total_size>(std::move(data));
            }

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
                return *(this->data_ptr__);
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
                return this->data_ptr__->data();
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
                return *(this->data_ptr__)[index];
            }

            /**
             * @brief this function is a overload of operator[], it will return the data according to the index.
             *
             * @param index data index
             * @return const T& reference of data
             */
            const T& operator[](size_t index) const {
                return *(this->data_ptr__)[index];
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
                return *(this->data_ptr__)[index];
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
                return *(this->data_ptr__)[index];
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
                return *(this->data_ptr__)[index];
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
                return *(this->data_ptr__)[index];
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
                        os << *(tensor.data_ptr__)[index + i] << ", ";
                    }
                    os << *(tensor.data_ptr__)[index + Shape::dims[level] - 1];
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
                return Vector<ValueType, Shape::total_size>(*(this->data_ptr__));
            }
        };
    };
};
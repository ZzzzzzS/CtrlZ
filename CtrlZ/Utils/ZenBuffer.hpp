/**
 * @file ZenBuffer.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <vector>
#include <iostream>
#include "ZObject.hpp"

namespace z
{
    /**
     * @brief ring buffer class, **this buffer does NOT garantee thread safety**
     *
     * @tparam T type of data
     */
    template <typename T>
    class RingBuffer : public ZObject
    {
    public:
        /**
         * @brief Construct a new Ring Buffer object with size
         *
         * @param _Size size of the buffer
         */
        RingBuffer(int _Size)
        {
            this->Size = _Size;
            this->Data.resize(_Size);
            this->CurrentIndex = 0;
        }

        /**
         * @brief Construct a new Ring Buffer object with size and initial value
         *
         * @param _Size size of the buffer
         * @param object initial value
         */
        RingBuffer(int _Size, T& object)
        {
            this->Size = _Size;
            this->Data.reserve(this->Size);
            for (size_t i = 0; i < this->Size; i++)
            {
                this->Data.push_back(object); // need to push object multiple times, so can not use move here.
            }
            this->CurrentIndex = 0;
        }

        /**
         * @brief push data to the buffer
         *
         * @param _Data data to push
         */
        void push(const T& _Data)
        {
            this->Data[this->CurrentIndex] = _Data;
            this->CurrentIndex = (this->CurrentIndex + 1) % this->Size;
        }

        /**
         * @brief get data from the buffer
         *
         * @param _Index index of the data
         * @return T& data
         */
        T& operator()(int _Index)
        {
            return this->Data[(_Index + this->CurrentIndex) % this->Size];
        }

        /**
         * @brief get data from the buffer
         *
         * @param _Index index of the data
         * @return T& data
         */
        T& operator[](int _Index)
        {
            return this->Data[(_Index + this->CurrentIndex) % this->Size];
        }

        /**
         * @brief get the oldest data from the buffer
         *
         * @return T& oldest data reference
         */
        T& front()
        {
            return this->Data[this->CurrentIndex];
        }

        /**
         * @brief get the newest data from the buffer
         *
         * @return T& newest data reference
         */
        T& back()
        {
            return this->Data[(this->CurrentIndex - 1 + this->Size) % this->Size];
        }

        /**
         * @brief get the size of the buffer
         *
         * @return int size of the buffer
         */
        int size()
        {
            return this->Size;
        }

        /**
         * @brief flush the buffer with default value
         *
         */
        void flush()
        {
            for (auto& i : this->Data)
            {
                i = T();
            }
        }

    private:
        std::vector<T> Data;
        int Size;
        int CurrentIndex;
    };

    /**
     * @brief filter class with simple average filter
     *
     * @tparam T
     */
    template <typename T>
    class filter : public ZObject
    {
    public:
        /**
         * @brief Construct a new filter object
         *
         * @param buffer_sz filter length
         */
        filter(int buffer_sz = 3)
        {
            this->buffer_sz = buffer_sz;
            this->buffer.resize(buffer_sz);
            for (auto& i : this->buffer)
                i = 0;
        }
        /**
         * @brief clear the buffer and reset the pointer with default value
         *
         */
        void clear()
        {
            for (auto& i : this->buffer)
            {
                i = T();
            }
        }

        /**
         * @brief operator () to filter the input
         *
         * @param input input data
         * @return T filtered data
         */
        T operator()(T input)
        {
            buffer[this->pointer] = input;
            this->pointer++;
            this->pointer %= this->buffer_sz;
            T sum = 0;
            for (auto& i : this->buffer)
            {
                sum += i;
            }
            return sum / static_cast<T>(buffer_sz);
        }

    private:
        int buffer_sz;
        std::vector<T> buffer;
        int pointer = 0;
    };

    /// @brief alias for double filter
    using filterd = filter<double>;
    /// @brief alias for float filter
    using filterf = filter<float>;


    /**
     * @brief filter class with weight filter
     *
     * @tparam T type of data
     */
    template <typename T>
    class WeightFilter : public ZObject
    {
    public:
        /**
         * @brief Construct a new Weight Filter object with weight
         *
         * @param param array of weight WeightFilter<double> filter({0.1, 0.2, 0.3, 0.4});
         */
        WeightFilter(std::initializer_list<T> param)
            : buffer(param.size())
        {
            this->weight = param;
            this->buffer_sz = param.size();
            this->SumWeight = T();
            for (auto&& i : this->weight)
            {
                this->SumWeight += i;
            }
        }

        WeightFilter(std::vector<T> param)
            : buffer(param.size())
        {
            this->weight = param;
            this->buffer_sz = param.size();
            this->SumWeight = T();
            for (auto&& i : this->weight)
            {
                this->SumWeight += i;
            }
        }

        /**
         * @brief clear the buffer and reset the pointer with default value
         *
         */
        void clear()
        {
            this->buffer.flush();
        }

        /**
         * @brief operator () to filter the input
         *
         * @param input input data
         * @return T filtered data
         */
        T operator()(T input)
        {
            buffer.push(input);

            T sum = T();
            for (size_t i = 0; i < this->buffer_sz; i++)
            {
                sum += buffer[i] * weight[i];
            }

            return sum / SumWeight;
        }

    private:
        size_t buffer_sz;
        std::vector<T> weight;
        RingBuffer<T> buffer;
        T SumWeight;
    };
};
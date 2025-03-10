/**
 * @file DataCenter.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <string>
#include <vector>
#include <map>
#include <array>
#include <any>
#include <mutex>
#include "../Utils/ZObject.hpp"
#include "../Utils/StaticStringUtils.hpp"
#include "atomic"

namespace z
{
    //TODO: 无锁ring buffer真tmd难，稍后再实现这个吧
    /**
     * @brief 数据中心类型，用于存储工作流水线中的数据。
     *
     * @tparam CTS 用于定义数据中心中的数据类型，是一系列键-值对，键是CTString类型，值是任意类型。
     * 具体定义方式参见CTSPair类型。
     */
    template<CTSPair ...CTS>
    class DataCenter
    {
    public:
        /**
         * @brief Construct a new Data Center object
         *
         */
        DataCenter()
        {

        }
        /**
         * @brief Destroy the Data Center object
         *
         */
        ~DataCenter()
        {
        }

        /**
         * @brief 向数据中心中写入数据
         *
         * @tparam CT 待写入数据的键，编译期字符串
         * @tparam T 待写入数据的类型
         * @param TimeStamp 写入的时间戳
         * @param data 待写入的数据
         */
        template<CTString CT, typename T>
        void SetData(size_t TimeStamp, const T& data)
        {
            size_t idx = this->dataBuffers_.template index<CT>();
            this->timeStamps_[idx] = TimeStamp;

            this->writelocks_[idx].lock();
            this->dataBuffers_.template set<CT, T>(data);
            this->writelocks_[idx].unlock();
        }

        /**
         * @brief 从数据中心中读取数据
         *
         * @tparam CT 待读取数据的键，编译期字符串
         * @tparam T 待读取数据的类型
         * @param data 读取到的数据
         * @return size_t 读取到的数据的时间戳
         */
        template<CTString CT, typename T>
        size_t GetData(T& data)
        {
            size_t idx = this->dataBuffers_.template index<CT>();
            this->writelocks_[idx].lock();
            this->dataBuffers_.template get<CT, T>(data);
            this->writelocks_[idx].unlock();
            return this->timeStamps_[idx];
        }


    private:
        /// @brief 用于存储数据的时间戳
        std::array<std::atomic_size_t, sizeof...(CTS)> timeStamps_{ 0 };

        /// @brief 用于保护数据的写锁
        std::array<std::mutex, sizeof...(CTS)> writelocks_;

        /// @brief 用于存储数据的缓冲区
        CTSMap<CTS...> dataBuffers_{};

    };
};
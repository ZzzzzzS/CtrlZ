/**
 * @file AsyncLoggerWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <thread>
#include <string>
#include <iostream>
#include <array>
#include <tuple>
#include <vector>
#include <type_traits>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <fstream>
#include <limits>
#include <filesystem>
#include "readerwriterqueue/readerwriterqueue.h"
 //#include "readerwriterqueue.h"

#include "../Utils/StaticStringUtils.hpp"
#include "../Utils/ZenBuffer.hpp"
#include "AbstractWorker.hpp"

namespace z
{
    /**
     * @brief 异步日志工人类型，用户可以通过这个工人类型来异步的实现记录数据的功能。
     * @details 异步日志工人类型，用户可以通过这个工人类型来异步的实现记录数据的功能。
     * 这个工人类型的优点是可以异步的记录数据，不会阻塞主线程，适用于一些需要记录数据的场景。
     * 比如需要记录一些传感器的数据，或者需要记录一些模型的输出数据。同时，用户可以通过配置文件来配置日志的路径和写入频率。
     * 所有的数据都会被记录到一个csv文件中，用户可以通过这个文件来查看数据。
     * @details config.json配置文件示例：
     * @code {.json}
     * {
     *   "Workers": {
     *        "AsyncLogger": {
     *           "LogPath": "./log/", //日志文件的路径
     *           "WriteBackFrequency": 1000 //日志的写入频率，单位为ms
     *             }
     *        }
     *  }
     * @endcode
     *
     * @tparam SchedulerType 调度器类型
     * @tparam LogPrecision 日志精度，用户可以通过这个参数来指定日志的精度，比如可以指定为float或者double
     * @tparam Args 一个或者多个CTSPair类型的模板参数，用户可以通过这个参数来指定需要记录的数据的名称和类型，
     * 请注意这些数据类型必须是在调度器的数据总线中注册过的数据类型。在这里记录的数据类型将会自动被正确记录并保存在CSV中。
     *
     */
    template<typename SchedulerType, typename LogPrecision, CTSPair ...Args>
    class AsyncLoggerWorker : public AbstractWorker<SchedulerType>
    {
        static_assert(std::is_arithmetic<LogPrecision>::value, "LogPrecision must be a number type");
    public:

        /**
         * @brief 创建一个异步日志工人类型
         *
         * @param scheduler 调度器的指针
         * @param root_cfg 配置文件
         */
        AsyncLoggerWorker(SchedulerType::Ptr scheduler, const nlohmann::json& root_cfg)
            :AbstractWorker<SchedulerType>(scheduler)
        {
            nlohmann::json cfg = root_cfg;
            //TODO: add static assert to check if the value type is a valid type (number or array of numbers)
            std::time_t now = std::time(nullptr);
            std::tm tm = *std::localtime(&now);
            std::string time_str = std::to_string(tm.tm_year + 1900) + "-" + std::to_string(tm.tm_mon + 1) + "-" + std::to_string(tm.tm_mday) + "-" + std::to_string(tm.tm_hour) + "-" + std::to_string(tm.tm_min) + "-" + std::to_string(tm.tm_sec);
            this->PrintSplitLine();
            std::cout << "AsyncLoggerWorker" << std::endl;
            std::string path;
            try
            {
                path = cfg["LogPath"].get<std::string>();
                this->LogPath__ = path;
            }
            catch (const std::exception&)
            {
                std::cerr << "AsyncLoggerWorker: Failed to get LogPath from config, use default path" << std::endl;
                this->LogPath__ = "./";
            }
            this->LogPath__ = path + time_str + ".csv";

            try
            {
                this->WriteBackFrequency__ = cfg["WriteBackFrequency"].get<size_t>();
            }
            catch (const std::exception&)
            {
                std::cerr << "AsyncLoggerWorker: Failed to get WriteBackFrequency from config, use default value" << std::endl;
                this->WriteBackFrequency__ = 1000;
            }

            std::cout << "LogPath:" << this->LogPath__ << std::endl;
            std::cout << "WriteBackFrequency:" << this->WriteBackFrequency__ << std::endl;
            std::cout << "\nLogged Data Pairs: \n";
            (PrintCTSPairInfo<Args>(), ...);
            this->PrintSplitLine();

        }

        /**
         * @brief 析构函数，虚函数，用于释放资源
         *
         */
        virtual ~AsyncLoggerWorker()
        {
            TaskDestroy();
        }

        // TODO: add a static assert to check if the value type is a valid type (number or array of numbers)
        // static constexpr void checkType()
        // {
        //     static_assert(std::conjunction_v<std::is_arithmetic<Args::type>...> , "All Args must be a number type");
        // }

        /**
         * @brief TaskRun方法，在每次任务队列循环中被调用，但是这个方法在这里没有实际的作用，因为数据记录通常发生在流水线的结尾，也就是在TaskCycleEnd方法中。
         *
         */
        virtual void TaskRun() override {}

        /**
         * @brief TaskCreate方法，在任务队列创建的时候会被调度器调用，这里用于初始化资源，打开文件流，生成表头，创建写入线程等。
         *
         */
        virtual void TaskCreate() override
        {
            std::lock_guard<std::mutex> lock(this->CreateMutex__);
            if (this->AlreadyCreated__)
            {
                return;
            }

            if (!std::filesystem::exists(this->LogPath__.substr(0, this->LogPath__.find_last_of("/"))))
            {
                std::cout << "AsyncLoggerWorker: Folder not exist, creating folder " << this->LogPath__.substr(0, this->LogPath__.find_last_of("/")) << std::endl;
                std::filesystem::create_directories(this->LogPath__.substr(0, this->LogPath__.find_last_of("/")));
            }

            this->FileStream__.open(this->LogPath__, std::fstream::out);
            if (!this->FileStream__.is_open())
            {
                std::cerr << "AsyncLoggerWorker: Failed to open file " << this->LogPath__ << std::endl;
                return;
            }
            GenerateHeader();

            this->LogThreadRun__ = true;
            size_t CurrentWriteBackCount__ = 0;
            std::atomic<bool> LogThreadNeedWrite__ = false;
            this->WriteLogThread__ = std::thread(&AsyncLoggerWorker::WriteLogThreadRun, this);
            this->AlreadyCreated__ = true;
        }

        /**
         * @brief TaskDestroy方法，在任务队列删除的时候会被调度器调用
         *
         */
        virtual void TaskDestroy() override
        {
            this->LogThreadSyncMutex__.lock();
            this->LogThreadRun__ = false;
            this->LogThreadSyncMutex__.unlock();
            this->LogThreadSyncCV__.notify_all();
            if (this->WriteLogThread__.joinable())
            {
                this->WriteLogThread__.join();
            }
        }

        /**
         * @brief TaskCycleEnd方法，在每次任务队列循环的结束会被调度器调用，用于将数据总线中的数据复制到数据队列中，供写入线程写入文件。
         *
         */
        virtual void TaskCycleEnd() override
        {
            size_t TS = this->Scheduler->getTimeStamp();
            if (TS == this->last_call_time.load())
            {
                //您可以在多个tasklist中注册logger，但是您需要确保正在运行的tasklist中该logger在每周期只会被调用一次，否则会出现数据重复记录的问题
                //如果您知道您在做什么，可以注释掉这个异常
                throw std::runtime_error("Mutiple call in one cycle, only one call is allowed in one cycle, check your task list!");
                this->last_call_time.store(TS);
                return;
            }
            this->last_call_time.store(TS);

            LogFrameType DataFrame;
            getValues(DataFrame);
            if (!this->DataQueue__.enqueue(DataFrame))
            {
                std::cerr << "AsyncLoggerWorker: DataQueue is full, data lost" << std::endl;
            }

            {
                std::unique_lock<std::mutex> lock(this->LogThreadSyncMutex__); //lock
                this->CurrentWriteBackCount__++;
                if (this->CurrentWriteBackCount__ >= this->WriteBackFrequency__)
                {
                    this->LogThreadNeedWrite__ = true;
                    lock.unlock();
                    this->LogThreadSyncCV__.notify_all();
                    this->CurrentWriteBackCount__ = 0;
                }
            }
        }

    private:
        static constexpr std::tuple<decltype(Args)...> TypeInfo__ = std::make_tuple(Args...);
        //static constexpr std::array<std::string, sizeof...(Args)> DataName__ = { Args.str.value... };

        std::string LogPath__;
        std::fstream FileStream__; //文件流
        size_t WriteBackFrequency__;
        size_t CurrentWriteBackCount__ = 0;
        std::atomic<bool> LogThreadNeedWrite__ = false; //写入标志

        std::thread WriteLogThread__;
        std::mutex LogThreadSyncMutex__;
        std::condition_variable LogThreadSyncCV__; //线程同步条件变量
        std::atomic<bool> LogThreadRun__ = false; //线程运行标志


        static constexpr size_t KeySize() { return sizeof...(Args); }
        static constexpr size_t HeaderSize()
        {
            size_t size = 0;
            std::apply([&size](auto&&... args) {((size += args.dim), ...); }, TypeInfo__);
            return size;
        }

        using LogFrameType = std::array<LogPrecision, HeaderSize()>;

        std::array<std::string, HeaderSize()> HeaderList__; //表头列表
        moodycamel2::ReaderWriterQueue<LogFrameType> DataQueue__; //数据队列

        bool AlreadyCreated__ = false;
        std::mutex CreateMutex__;
        std::atomic<size_t> last_call_time = { std::numeric_limits<size_t>::max() };

    private:
        void GenerateHeader()
        {
            //this->HeaderList__.clear();
            size_t idx = 0;
            auto lambda = [this, &idx]<typename T>(T & t)
            {
                if constexpr (T::isArray)
                {
                    for (size_t i = 0; i < T::dim;i++)
                    {
                        std::string str_idx = T::str.value;
                        str_idx += "[" + std::to_string(i) + "]";
                        this->HeaderList__[idx++] = str_idx;
                    }
                }
                else
                {
                    std::string str_idx = T::str.value;
                    this->HeaderList__[idx++] = str_idx;
                }
            };
            std::apply([&lambda](auto&& ...args) {((lambda(args)), ...);}, TypeInfo__);
        }

        void getValues(LogFrameType& DataFrame)
        {
            size_t idx = 0;
            auto lambda = [&DataFrame, this, &idx]<typename T>(T & t)
            {
                std::remove_pointer_t<decltype(t.type)> v;
                this->Scheduler->template GetData<t.str>(v);
                if constexpr (T::isArray)
                {
                    for (size_t i = 0; i < T::dim;i++)
                    {
                        DataFrame[idx++] = static_cast<LogPrecision>(v[i]);
                    }
                }
                else
                {
                    DataFrame[idx++] = static_cast<LogPrecision>(v);
                }
            };
            std::apply([&lambda](auto&& ...args) {((lambda(args)), ...);}, TypeInfo__);
        }

        void WriteHeader()
        {
            for (size_t i = 0; i < this->HeaderList__.size() - 1;i++)
            {
                this->FileStream__ << this->HeaderList__[i] << ",";
            }
            this->FileStream__ << this->HeaderList__[this->HeaderList__.size() - 1] << std::endl;
        }

        void WriteFile()
        {
            while (this->DataQueue__.size_approx())
            {
                LogFrameType DataFrame;
                if (!this->DataQueue__.try_dequeue(DataFrame))
                {
                    break;
                }

                for (size_t i = 0; i < DataFrame.size() - 1;i++)
                {
                    this->FileStream__ << DataFrame[i] << ",";
                }
                this->FileStream__ << DataFrame[DataFrame.size() - 1] << std::endl;
            }
        }

        void WriteLogThreadRun()
        {
            WriteHeader();
            while (this->LogThreadRun__)
            {
                {
                    std::unique_lock<std::mutex> lock(this->LogThreadSyncMutex__);
                    this->LogThreadSyncCV__.wait(lock, [this] {return this->LogThreadNeedWrite__ || !this->LogThreadRun__; });
                }
                if (!this->LogThreadRun__) break;
                //std::cout << "WriteLogThreadRun" << std::endl;
                WriteFile();
                this->LogThreadNeedWrite__ = false;
            }
            //std::cout << "WriteLogThreadRun exit" << std::endl;
            WriteFile();
            this->FileStream__.close();
        }
    };
};

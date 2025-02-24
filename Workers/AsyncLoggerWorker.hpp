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
#include "readerwriterqueue.h"

#include "../Utils/StaticStringUtils.hpp"
#include "../Utils/ZenBuffer.hpp"
#include "AbstractWorker.hpp"

namespace zzs
{
    template<typename SchedulerType, typename LogPrecision, CTSPair ...Args>
    class AsyncLoggerWorker : public AbstractWorker<SchedulerType>
    {
        static_assert(std::is_arithmetic<LogPrecision>::value, "LogPrecision must be a number type");
    public:
        AsyncLoggerWorker(SchedulerType* scheduler, const nlohmann::json& root_cfg)
            :AbstractWorker<SchedulerType>(scheduler)
        {
			nlohmann::json cfg = root_cfg["Workers"]["AsyncLogger"];
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
            this->PrintSplitLine();

        }

        virtual ~AsyncLoggerWorker()
        {
            TaskDestroy();
        }

        // TODO: add a static assert to check if the value type is a valid type (number or array of numbers)
        // static constexpr void checkType()
        // {
        //     static_assert(std::conjunction_v<std::is_arithmetic<Args::type>...> , "All Args must be a number type");
        // }

        virtual void TaskRun() override{}

        virtual void TaskCreate() override
        {
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
        }

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

        virtual void TaskCycleEnd() override
        {
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
        moodycamel::ReaderWriterQueue<LogFrameType> DataQueue__; //数据队列

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
			std::cout << "WriteLogThreadRun exit" << std::endl;
            WriteFile();
            this->FileStream__.close();
        }
    };
};

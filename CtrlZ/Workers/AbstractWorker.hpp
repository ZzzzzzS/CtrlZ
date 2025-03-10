/**
 * @file AbstractWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "../Utils/ZObject.hpp"
#include "../Utils/StaticStringUtils.hpp"
#include <memory>
#include <nlohmann/json.hpp>

namespace z
{
    /**
     * @brief AbstractWorker 类型是一切工人类型的基类，在这个类中指定了一些必须要实现的基本方法，这些方法将在调度器的调度流水线下被依次调用
     * 来实现工人的工作逻辑。通过将多个Worker类型的工人注册到调度器中，形成一个TaskList，用户可以实现复杂的工作流程，在构成一个流水线的同时，
     * 保证每个工人模块的逻辑是独立的，从而实现了工作流程的模块化和可扩展性。
     * @details 这个类的主要目的是为了实现工人的工作逻辑，用户可以通过继承这个类来实现自己的工人类型，然后通过调度器来调度这些工人的工作流程。
     * 具体来说，调度器在任务队列创建和删除的时候会调用抽象工人类型的TaskCreate和TaskDestroy方法；而在任务队列的循环中，调度器会依次调用
     * TaskCycleBegin，TaskRun和TaskCycleEnd方法，其中TaskRun方法是用户必须要实现的方法，在用户继承的子类中应当实现当前工作的具体逻辑。
     * 举例来说，如果这个Worker是一个用来处理电机的Worker，那么可能的实现是在TaskRun方法中获取当前电机的状态和下发电机的目标状态，
     * 在TaskCreate方法中初始化电机的通信接口，在TaskDestroy方法中关闭电机的通信接口。
     *
     * @tparam SchedulerType 调度器类型，这个调度器类型会根据用户定义的调度逻辑依次调度工人的工作流程。
     */
    template<typename SchedulerType>
    class AbstractWorker : public ZObject
    {
    public:
        AbstractWorker() {}

        /**
         * @brief 构造一个抽象工人类型
         *
         * @param scheduler 调度器的指针，可以通过调度器来获得一些全局的数据，比如当前任务的时间戳，前级工人的输出数据等
         * @param cfg 配置文件，用户可以通过配置文件来配置工人的一些参数。
         */
        AbstractWorker(SchedulerType* scheduler, const nlohmann::json& cfg = nlohmann::json())
        {
            this->Scheduler = scheduler;
        }

        /**
         * @brief 设置调度器的指针
         *
         * @param scheduler 调度器指针
         */
        void setScheduler(SchedulerType* scheduler) { this->scheduler = scheduler; }

        /**
         * @brief 析构函数，虚函数，用于释放资源
         *
         */
        virtual ~AbstractWorker() {}

        /**
         * @brief TaskCreate方法，在任务队列创建的时候会被调度器调用，用户可以在这个方法中初始化一些资源
         *
         */
        virtual void TaskCreate() {}

        /**
         * @brief TaskDestroy方法，在任务队列删除的时候会被调度器调用，用户可以在这个方法中释放一些资源
         *
         */
        virtual void TaskDestroy() {}

        /**
         * @brief TaskCycleBegin方法，在每次任务队列循环的开始会被调度器调用。
         *
         */
        virtual void TaskCycleBegin() {}

        /**
         * @brief TaskRun方法，在每次任务队列循环中被调用，
         * 具体的调用时机取决于用户在调度器中设置的当前工人在流水线中的顺序，
         * 用户必须要实现的方法，用于实现当前工人的具体工作逻辑。
         *
         */
        virtual void TaskRun() = 0;

        /**
         * @brief TaskCycleEnd方法，在每次任务队列循环的结束会被调度器调用。
         *
         */
        virtual void TaskCycleEnd() {}

    protected:
        /**
         * @brief 调度器的指针，用户可以通过这个指针来获取一些全局的数据，
         * 比如当前任务的时间戳，前级工人的输出数据等，也可以通过这个指针来设置一些全局的数据。
         */
        SchedulerType* Scheduler = nullptr;
    };


    /**
     * @brief SimpleCallbackWorker 类型是一个简单的回调工人类型，用户可以通过这个工人类型来实现一些简单的工作逻辑。
     * @details SimpleCallbackWorker 类型是一个简单的回调工人类型，用户可以通过这个工人类型来实现一些简单的工作逻辑。
     * 这个工人类型的优点是简单易用，用户只需要实现一个回调函数，就可以实现一个工人的工作逻辑，适用于一些简单的工作逻辑。
     * 比如需要临时修改流水线中的一些数据，或者需要在流水线中插入一些简单的逻辑。
     *
     * @tparam SchedulerType 调度器类型。
     */
    template<typename SchedulerType>
    class SimpleCallbackWorker : public AbstractWorker<SchedulerType>
    {
        /// @brief 回调函数类型，函数签名为void(SchedulerType*)，包含一个调度器的指针
        using CallbackType = std::function<void(SchedulerType*)>;
    public:

        /**
         * @brief 构造一个简单的回调工人类型
         *
         * @param scheduler 调度器的指针
         * @param func 回调函数，用户可以通过这个回调函数来实现工人的工作逻辑，该函数在TaskRun方法中被调用
         * @param cfg 配置文件，用户可以通过配置文件来配置工人的一些参数。
         */
        SimpleCallbackWorker(SchedulerType* scheduler, CallbackType func, const nlohmann::json& cfg = nlohmann::json())
            :AbstractWorker<SchedulerType>(scheduler), callback__(func), cfg__(cfg) {}

        /**
         * @brief 析构函数，虚函数，用于释放资源
         *
         */
        virtual ~SimpleCallbackWorker() {}

        /**
         * @brief TaskRun方法，在每次任务队列循环中被调用，这个方法会调用用户传入的回调函数
         *
         */
        virtual void TaskRun() override
        {
            callback__(this->Scheduler);
        }

        /**
         * @brief 获取配置文件
         *
         * @return 配置文件
         */
        nlohmann::json& getConfig() { return this->cfg__; }

    private:
        /// @brief 回调函数
        CallbackType callback__;

        /// @brief 配置文件
        nlohmann::json cfg__;
    };
};
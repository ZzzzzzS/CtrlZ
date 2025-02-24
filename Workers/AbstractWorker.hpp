#pragma once
#include "../Utils/ZObject.hpp"
#include "../Utils/StaticStringUtils.hpp"
#include <memory>
#include <nlohmann/json.hpp>

namespace zzs
{
    /**
     * @brief
     *
     * @tparam SchedulerType
     */
    template<typename SchedulerType>
    class AbstractWorker : public ZObject
    {
    public:
        AbstractWorker() {}

        AbstractWorker(SchedulerType* scheduler, const nlohmann::json& cfg = nlohmann::json())
        {
            this->Scheduler = scheduler;
        }

        void setScheduler(SchedulerType* scheduler) { this->scheduler = scheduler; }

        virtual ~AbstractWorker() {}

        virtual void TaskCreate() {}

        virtual void TaskDestroy() {}

        virtual void TaskCycleBegin() {}
        virtual void TaskRun() = 0;
        virtual void TaskCycleEnd() {}

    protected:
        SchedulerType* Scheduler = nullptr;
    };

    template<typename SchedulerType>
    class SimpleTestWorker : public AbstractWorker<SchedulerType>
    {
    public:
        SimpleTestWorker(SchedulerType* scheduler)
            :AbstractWorker<SchedulerType>(scheduler) {}
        virtual ~SimpleTestWorker() {}
    protected:
        virtual void TaskRun() override
        {
            std::cout << "SimpleTestWorker::TaskRun" << std::endl;
            this->Scheduler->template SetData<"1">(123);
            this->Scheduler->template SetData<"1">(123);
            int data;
            this->Scheduler->template GetData<"1">(data);
            std::cout << "data:" << data << std::endl;
        }
    };

    template<typename SchedulerType>
    class SimpleCallbackWorker : public AbstractWorker<SchedulerType>
    {
        using CallbackType = std::function<void(SchedulerType*)>;
    public:
        SimpleCallbackWorker(SchedulerType* scheduler, CallbackType func, const nlohmann::json& cfg = nlohmann::json())
            :AbstractWorker<SchedulerType>(scheduler), callback__(func), cfg__(cfg) {}
        virtual ~SimpleCallbackWorker() {}

        virtual void TaskRun() override
        {
            callback__(this->Scheduler);
        }
        nlohmann::json& getConfig() { return this->cfg__; }

    private:
        CallbackType callback__;
        nlohmann::json cfg__;
    };
};
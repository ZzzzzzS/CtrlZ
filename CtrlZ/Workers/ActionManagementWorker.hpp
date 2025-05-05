/**
 * @file ActionManagementWorker.hpp
 * @author Zishun Zhou
 * @brief 该文件定义了一个类ActionManagementWorker，用于管理和选择单个或多个推理网络的输出。
 * @date 2025-04-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <tuple>
#include <array>
#include <string>
#include <memory>
#include <iostream>
#include <nlohmann/json.hpp>
#include <type_traits>
#include <initializer_list>
#include <functional>
#include <limits>

#include "Schedulers/AbstractScheduler.hpp"
#include "Workers/AbstractWorker.hpp"
#include "Utils/MathTypes.hpp"
#include "Utils/StaticStringUtils.hpp"
#include "Workers/NN/AbstractInferenceWorker.hpp"

namespace z
{
    /**
     * @brief ActionManagementWorker类型是一个用于管理和选择单个或多个推理网络的输出的工人类型。
     *
     * @details ActionManagementWorker类型是一个用于管理和选择单个或多个推理网络的输出的工人类型。
     * 用户可以通过这个工人类型来实现多个推理网络的输出的选择和管理。用户可以通过配置文件来配置需要管理的网络输出的名称和类型，
     * 用户可以通过SwitchTo方法来切换到指定的网络输出，用户也可以通过SetActionRemapFunction方法来设置网络输出的重映射函数。
     * 该类型实现了一个默认的重映射函数，默认重映射函数将网络输出的值直接写入数据总线的"TargetMotorPosition"中。
     *
     * * @details config.json配置文件示例：
     * {
     *    "Workers": {
     *       "ActionManagement": {
     *          "SwitchIntervalTime": 1, //默认切换间隔时间 1s
     *       }
     * }
     *
     * @tparam SchedulerType 调度器类型
     * @tparam InferencePrecision 推理精度，用户可以通过这个参数来指定推理的精度，比如可以指定为float。
     * @tparam ActionPairs 多个网络输出的ActionPair类型，用户可以通过这个参数来指定需要管理的网络输出的名称和类型。
     */
    template<typename SchedulerType, typename InferencePrecision, CTSPair ...ActionPairs>
    class ActionManagementWorker : public AbstractWorker<SchedulerType>
    {
    private:
        template<CTSPair First>
        static constexpr size_t ElementSize()
        {
            return First.dim;
        }

        template<CTSPair First, CTSPair ...Rest>
        static constexpr size_t ElementSize()
        {
            return First.dim;
        }
        static constexpr size_t ActionElementSize__ = ElementSize<ActionPairs...>();


    public:
        /// @brief 定义一个函数类型，用于重映射网络输出的函数类型，该函数类型接受一个调度器指针和一个网络输出的值，返回void。
        using OutPutRemapFunction = std::function<void(SchedulerType*, const z::math::Vector<InferencePrecision, ActionElementSize__>)>;

        using InferenceWorkerType = AbstractNetInferenceWorker<SchedulerType*, InferencePrecision>;

    public:
        /**
         * @brief 构造一个ActionManagementWorker类型
         *
         * @param scheduler 调度器的指针
         * @param worker_cfg 配置文件，用户可以通过配置文件来配置工人的一些参数。
         * @param scheduler_cfg 调度器的配置文件，用户可以通过配置文件来配置调度器的一些参数。
         */
        ActionManagementWorker(SchedulerType* scheduler, const nlohmann::json& worker_cfg, const nlohmann::json& scheduler_cfg)
            :AbstractWorker<SchedulerType>(scheduler, worker_cfg)
        {
            this->block__.store(true); //默认阻塞输出
            this->CycleTime__ = scheduler_cfg["dt"].get<InferencePrecision>();
            this->ActionRemapFunctions__.fill(std::bind(&ActionManagementWorker::DefaultActionRemapFunction, this, this->Scheduler, std::placeholders::_2));
            InferencePrecision DefaultSwitchIntervalTime = worker_cfg["SwitchIntervalTime"].get<InferencePrecision>();
            this->DefaultNextActionCycleCnt__ = static_cast<decltype(this->DefaultNextActionCycleCnt__)>(DefaultSwitchIntervalTime / this->CycleTime__);
        }

        /**
         * @brief 任务运行方法，在每次任务队列循环中被调用。
         *
         */
        void TaskRun()
        {
            if (this->block__)
            {
                return;
            }
            if (this->NextActionCycleCnt__ > 0)
            {
                this->NextActionCycleCnt__--;
                if (this->NextActionCycleCnt__ == 0)
                {
                    this->ActionIndex__ = this->NextActionIndex__;
                }
            }
            else
            {
                this->ActionIndex__ = this->NextActionIndex__;
            }
            (this->ProcessAction<ActionPairs.str>(), ...);
        }

        /**
         * @brief 设置网络输出的重映射函数，用户可以通过这个方法来设置网络输出的重映射函数。
         *
         * @tparam CT 编译期字符串常量，表示网络输出的名称。
         * @param func 重映射函数，用户可以通过这个函数来实现网络输出的重映射逻辑。
         */
        template<CTString CT>
        void SetActionRemapFunction(OutPutRemapFunction func)
        {
            constexpr size_t idx = ActionPairs__.template index<CT>();
            static_assert(idx != sizeof...(ActionPairs), "ActionPair not found in ActionManagementWorker");
            ActionRemapFunctions__[idx] = func;
        }

        /**
         * @brief 切换到指定的网络输出，用户可以通过这个方法来切换到指定的网络输出。
         *
         * @param ActionName 网络输出的名称，用户可以通过这个名称来指定需要切换到的网络输出。
         * @param SwitchIntervalTime 切换间隔时间，用户可以通过这个参数来指定切换的间隔时间，网络将会在用户指定的时间后切换到指定的网络输出。
         * @details 如果SwitchIntervalTime为-1，则使用默认的切换间隔时间。（切换单位为秒）
         * @return true 切换成功
         * @return false 切换失败，网络输出的名称未找到。
         */
        bool SwitchTo(std::string ActionName, InferencePrecision SwitchIntervalTime = -1)
        {
            this->block__ = false;
            size_t idx = ActionPairs__.index(ActionName);
            if (idx == sizeof...(ActionPairs))
            {
                std::cerr << "ActionManagementWorker: Action not found in ActionManagementWorker" << std::endl;
                return false;
            }

            if (this->ActionIndex__ == idx)
            {
                return true;
            }

            this->NextActionIndex__ = idx;

            if (this->NextActionCycleCnt__ == -1)
            {
                this->NextActionCycleCnt__ = this->DefaultNextActionCycleCnt__;
            }
            else
            {
                this->NextActionCycleCnt__ = SwitchIntervalTime / this->CycleTime__;
            }
            return true;
        }

        /**
         * @brief 阻止输出，用户可以通过这个方法来阻止输出。输出被阻止后，网络输出将不会更新电机目标位置，
         * 用户可以通过SwitchTo方法来切换到指定的网络输出来重新更新位置。
         *
         */
        void BlockOutput()
        {
            this->NextActionCycleCnt__ = std::numeric_limits<size_t>::max();
            this->block__ = true;
        }

    private:
        void DefaultActionRemapFunction(SchedulerType* scheduler, const z::math::Vector<InferencePrecision, ActionElementSize__>& data)
        {
            scheduler->template SetData<"TargetMotorPosition">(data);
        }

        template<CTString CT>
        void ProcessAction()
        {
            constexpr size_t Idx = ActionPairs__.template index<CT>();
            if (Idx == this->ActionIndex__)
            {
                z::math::Vector<InferencePrecision, ActionElementSize__> NetOut;
                this->Scheduler->template GetData<CT>(NetOut);
                this->ActionRemapFunctions__[Idx](this->Scheduler, NetOut);
            }
        }

    private:
        std::array<OutPutRemapFunction, sizeof...(ActionPairs)> ActionRemapFunctions__;
        CTSMap<ActionPairs...> ActionPairs__;

        size_t ActionIndex__ = 0;
        std::atomic<size_t> NextActionIndex__ = 0;
        std::atomic<size_t> NextActionCycleCnt__ = 0;
        size_t DefaultNextActionCycleCnt__;
        std::atomic<bool> block__ = true;
        InferencePrecision CycleTime__;


        //static_assert(isAllSameType<ActionPairs...>(), "All ActionPairs must be the same type.");
        //static_assert(std::is_same_v<InferencePrecision, typename ActionPairs::type>, "InferencePrecision must be the same as ActionPairs type.");
        static_assert(sizeof...(ActionPairs) > 0, "ActionPairs must be at least one.");
    };
};


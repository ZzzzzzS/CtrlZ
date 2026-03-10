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
     * @details config.json配置文件示例：
     * @code {.json}
     * {
     *    "Workers": {
     *       "ActionManagement": {
     *          "SwitchIntervalTime": 1, //默认切换间隔时间 1s
     *       }
     * }
     * @endcode
     *
     * @tparam SchedulerType 调度器类型
     * @tparam InferencePrecision 推理精度，用户可以通过这个参数来指定推理的精度，比如可以指定为float。
     * @tparam ActionPairs 多个网络输出的ActionPair类型，用户可以通过这个参数来指定需要管理的网络输出的名称和类型。
     */
    template<typename SchedulerType, typename InferencePrecision, CTSPair ...ActionPairs>
    class ActionManagementWorker : public AbstractWorker<SchedulerType>
    {
    protected:
        template<CTSPair First, CTSPair ...Rest>
        static constexpr size_t ElementSize()
        {
            return First.dim;
        }
        static constexpr size_t ActionElementSize__ = ElementSize<ActionPairs...>();

    public:
        /// @brief 定义一个函数类型，用于重映射网络输出的函数类型，该函数类型接受一个调度器指针和一个网络输出的值，返回void。
        using OutPutRemapFunction = std::function<void(typename SchedulerType::Ptr, const z::math::Vector<InferencePrecision, ActionElementSize__>)>;

    public:
        /**
         * @brief 构造一个ActionManagementWorker类型
         *
         * @param scheduler 调度器的指针
         * @param worker_cfg 配置文件，用户可以通过配置文件来配置工人的一些参数。
         */
        ActionManagementWorker(SchedulerType::Ptr scheduler, const nlohmann::json& worker_cfg)
            :AbstractWorker<SchedulerType>(scheduler, worker_cfg)
        {
            this->PrintSplitLine();
            std::cout << "ActionManagementWorker" << std::endl;
            this->block__.store(true); //默认阻塞输出
            this->CycleTime__ = scheduler->getSpinOnceTime();
            this->ActionRemapFunctions__.fill(std::bind(&ActionManagementWorker::DefaultActionRemapFunction, this, this->Scheduler, std::placeholders::_2));
            InferencePrecision DefaultSwitchIntervalTime = worker_cfg["SwitchIntervalTime"].get<InferencePrecision>();
            this->DefaultNextActionCycleCnt__ = static_cast<decltype(this->DefaultNextActionCycleCnt__)>(DefaultSwitchIntervalTime / this->CycleTime__);

            std::cout << "DefaultSwitchIntervalTime=" << DefaultSwitchIntervalTime << std::endl;

            std::cout << "Managed Action Pairs: \n";
            // (std::cout << ActionPairs.str.value...);
            (PrintCTSPairInfo<ActionPairs>(), ...);
            this->PrintSplitLine();
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

            if (this->NextActionCycleCnt__.load() < this->Scheduler->getTimeStamp())
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
         * @brief 设置网络输出的重映射函数，用户可以通过这个方法来设置网络输出的重映射函数。
         *
         * @tparam CT 编译期动作对，表示网络输出的名称和对应的类型。
         * @param func 重映射函数，用户可以通过这个函数来实现网络输出的重映射逻辑。
         */
        template<CTSPair CT>
        void SetActionRemapFunction(OutPutRemapFunction func)
        {
            constexpr size_t idx = ActionPairs__.template index<CT.str>();
            static_assert(CT.dim == ActionElementSize__, "ActionPair size not match, the size of ActionPair must be the same as ActionElementSize__");
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

            return this->switchTo(idx, SwitchIntervalTime);
        }

        /**
         * @brief 切换到指定的网络输出，用户可以通过这个方法来切换到指定的网络输出。
         *
         * @tparam CTSPair ActionPair 网络输出的名称类型对，用户可以通过这个名称来指定需要切换到的网络输出。
         * @tparam InferencePrecision SwitchIntervalTime 切换间隔时间，用户可以通过这个参数来指定切换的间隔时间，网络将会在用户指定的时间后切换到指定的网络输出。
         * @details 如果SwitchIntervalTime为-1，则使用默认的切换间隔时间。（切换单位为秒）
         * @return true 切换成功
         * @return false 切换失败，网络输出的名称未找到。
         */
        template<CTSPair ActionPair, InferencePrecision SwitchIntervalTime = static_cast<InferencePrecision>(-1)>
        bool SwitchTo()
        {
            this->block__ = false;
            static_assert(ActionPair.dim == ActionElementSize__, "ActionPair size not match, the size of ActionPair must be the same as ActionElementSize__");
            constexpr size_t idx = ActionPairs__.template index<ActionPair.str>();
            if (idx == sizeof...(ActionPairs))
            {
                std::cerr << "ActionManagementWorker: Action not found in ActionManagementWorker" << std::endl;
                return false;
            }

            return this->switchTo(idx, SwitchIntervalTime);
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

    protected:
        void DefaultActionRemapFunction(SchedulerType::Ptr scheduler, const z::math::Vector<InferencePrecision, ActionElementSize__>& data)
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

        bool switchTo(const size_t idx, InferencePrecision SwitchIntervalTime)
        {
            if (this->ActionIndex__ == idx)
            {
                return true;
            }

            this->NextActionIndex__ = idx;

            if (SwitchIntervalTime == -1)
            {
                this->NextActionCycleCnt__ = this->DefaultNextActionCycleCnt__;
            }
            else
            {
                this->NextActionCycleCnt__ = static_cast<size_t>(SwitchIntervalTime / this->CycleTime__);
            }
            this->NextActionCycleCnt__ += this->Scheduler->getTimeStamp();
            return true;
        }

    protected:
        std::array<OutPutRemapFunction, sizeof...(ActionPairs)> ActionRemapFunctions__;
        CTSMap<ActionPairs...> ActionPairs__;

        size_t ActionIndex__ = std::numeric_limits<size_t>::max();
        std::atomic<size_t> NextActionIndex__ = 0;
        std::atomic<size_t> NextActionCycleCnt__ = 0;
        size_t DefaultNextActionCycleCnt__;
        std::atomic<bool> block__ = true;
        InferencePrecision CycleTime__;


        //static_assert(isAllSameType<ActionPairs...>(), "All ActionPairs must be the same type.");
        //static_assert(std::is_same_v<InferencePrecision, typename ActionPairs::type>, "InferencePrecision must be the same as ActionPairs type.");
        static_assert(sizeof...(ActionPairs) > 0, "ActionPairs must be at least one.");
    };


    /**
     * @brief ActionAndMotorPropertiesManagementWorker类型是一个用于管理和选择单个或多个推理网络的输出以及电机属性的工人类型。
     *
     * @details ActionManagementWorker类型是一个用于管理和选择单个或多个推理网络和电机属性的输出的工人类型。
     * 用户可以通过这个工人类型来实现多个推理网络的输出的选择和管理。用户可以通过配置文件来配置需要管理的网络输出的名称和类型，
     * 同时也可以管理电机的属性，比如位置增益，速度增益，位置偏移，动作增益等。该类型继承自ActionManagementWorker类型，共享其全部功能和配置文件。
     * 用户可以通过SwitchTo方法来切换到指定的网络输出，用户也可以通过SetActionRemapFunction方法来设置网络输出的重映射函数。
     * 该类型实现了一个默认的重映射函数，默认重映射函数将网络输出的值直接写入数据总线的"TargetMotorPosition"中。
     *
     * @details config.json配置文件示例：
     * @code {.json}
     * {
     *    "Workers": {
     *       "ActionManagement": {
     *          "SwitchIntervalTime": 1, //默认切换间隔时间 1s
     *          "MotorProperties":[
     *            {          //对应的网络名称,和ActionPairs中的名称一致
     *               "PolicyNetName":"name" //属性名称
     *               "Stiffness": [100,100,100,100,100,100]
     *               "Damping": [10,10,10,10,10,10],
     *               "ActionOffset": [0,0,0,0,0,0]，
     *               "ActionScale": [1,1,1,1,1,1]
     *              }
     *           ]
     *       }
     * }
     * @endcode
     *
     * @tparam SchedulerType 调度器类型
     * @tparam InferencePrecision 推理精度，用户可以通过这个参数来指定推理的精度，比如可以指定为float。
     * @tparam ActionPairs 多个网络输出的ActionPair类型，用户可以通过这个参数来指定需要管理的网络输出的名称和类型。
     */
    template<typename SchedulerType, typename InferencePrecision, CTSPair ...ActionPairs>
    class ActionAndMotorPropertiesManagementWorker : public ActionManagementWorker<SchedulerType, InferencePrecision, ActionPairs...>
    {
    public:
        using BaseType = ActionManagementWorker<SchedulerType, InferencePrecision, ActionPairs...>;
        using MotorVec = z::math::Vector<InferencePrecision, BaseType::ActionElementSize__>;
        using BaseType::ActionElementSize__;
        /**
         * @brief 构造一个ActionAndMotorPropertiesManagementWorker类型
         *
         * @param scheduler 调度器的指针
         * @param worker_cfg 配置文件，用户可以通过配置文件来配置工人的一些参数。
         */
        ActionAndMotorPropertiesManagementWorker(SchedulerType::Ptr scheduler, const nlohmann::json& worker_cfg)
            :BaseType(scheduler, worker_cfg)
        {
            this->PrintSplitLine();
            std::cout << "ActionAndMotorPropertiesManagementWorker" << std::endl;
            this->ActionRemapFunctions__.fill(std::bind(&ActionAndMotorPropertiesManagementWorker::DefaultActionRemapFunction, this, this->Scheduler, std::placeholders::_2));

            nlohmann::json motor_properties_cfg = worker_cfg["MotorProperties"];
            for (auto&& node : motor_properties_cfg)
            {
                std::string net_name = node["PolicyNetName"];
                size_t idx = this->ActionPairs__.index(net_name);
                size_t alter_idx = this->ActionPairs__.index(net_name + "Action");
                if (idx == sizeof...(ActionPairs) && alter_idx == sizeof...(ActionPairs))
                {
                    std::cerr << "ActionAndMotorPropertiesManagementWorker: ActionPair " << net_name << " not found in ActionManagementWorker" << std::endl;
                    throw std::runtime_error("ActionPair not found in ActionManagementWorker");
                    continue;
                }

                idx = (idx == sizeof...(ActionPairs)) ? alter_idx : idx;

                std::cout << "Configuring Motor Properties for ActionPair: " << net_name << ", with action index= " << idx << std::endl;

                nlohmann::json property_cfg = node;
                MotorVec PositionKp;
                MotorVec VelocityKd;
                MotorVec ActionOffset;
                MotorVec ActionScale;

                if (property_cfg.contains("Stiffness") && property_cfg.contains("Damping"))
                {
                    nlohmann::json pos_kp_cfg = property_cfg["Stiffness"];
                    nlohmann::json pos_kd_cfg = property_cfg["Damping"];
                    if (pos_kp_cfg.size() != BaseType::ActionElementSize__)
                        throw(std::runtime_error("Stiffness size is not equal!"));
                    if (pos_kd_cfg.size() != BaseType::ActionElementSize__)
                        throw(std::runtime_error("Damping size is not equal!"));
                    for (size_t i = 0; i < BaseType::ActionElementSize__; i++)
                    {
                        PositionKp[i] = pos_kp_cfg[i].get<InferencePrecision>();
                        VelocityKd[i] = pos_kd_cfg[i].get<InferencePrecision>();
                    }
                    std::cout << "  PositionKp: " << PositionKp << std::endl;
                    std::cout << "  VelocityKd: " << VelocityKd << std::endl;
                }
                else
                {
                    std::cout << "  Motor Properties for ActionPair: " << net_name << " not fully specified, using default values.\n";
                    PositionKp.fill(static_cast<InferencePrecision>(0));
                    VelocityKd.fill(static_cast<InferencePrecision>(0));
                }

                if (property_cfg.contains("ActionOffset"))
                {
                    nlohmann::json action_offset_cfg = property_cfg["ActionOffset"];
                    if (action_offset_cfg.size() != BaseType::ActionElementSize__)
                        throw(std::runtime_error("ActionOffset size is not equal!"));
                    for (size_t i = 0; i < BaseType::ActionElementSize__; i++)
                    {
                        ActionOffset[i] = action_offset_cfg[i].get<InferencePrecision>();
                    }
                    std::cout << "  ActionOffset: " << ActionOffset << std::endl;
                }
                else
                {
                    ActionOffset.fill(static_cast<InferencePrecision>(0));
                    std::cout << "  ActionOffset not specified, using default value 0.\n";
                }

                if (property_cfg.contains("ActionScale"))
                {
                    nlohmann::json action_scale_cfg = property_cfg["ActionScale"];
                    if (action_scale_cfg.size() != BaseType::ActionElementSize__)
                        throw(std::runtime_error("ActionScale size is not equal!"));
                    for (size_t i = 0; i < BaseType::ActionElementSize__; i++)
                    {
                        ActionScale[i] = action_scale_cfg[i].get<InferencePrecision>();
                    }
                    std::cout << "  ActionScale: " << ActionScale << std::endl;
                }
                else
                {
                    ActionScale.fill(static_cast<InferencePrecision>(1));
                    std::cout << "  ActionScale not specified, using default value 1.\n";
                }

                this->PositionKp__[idx] = PositionKp;
                this->VelocityKd__[idx] = VelocityKd;
                this->ActionOffset__[idx] = ActionOffset;
                this->ActionScale__[idx] = ActionScale;
                std::cout << std::endl;
            }
            this->PrintSplitLine();
        }

        void TaskCreate() override
        {
            BaseType::TaskCreate();
            MotorVec default_stiffness = MotorVec::zeros();
            MotorVec default_damping = MotorVec::zeros();
            this->Scheduler->template GetData<"TargetMotorStiffness">(default_stiffness);
            this->Scheduler->template GetData<"TargetMotorDamping">(default_damping);

            for (size_t i = 0; i < sizeof...(ActionPairs); i++)
            {

                if (z::math::all(this->PositionKp__[i] == MotorVec::zeros()))
                {
                    this->PositionKp__[i] = default_stiffness;
                }
                if (z::math::all(this->VelocityKd__[i] == MotorVec::zeros()))
                {
                    this->VelocityKd__[i] = default_damping;
                }
            }

        }

        void TaskRun()
        {
            if (this->block__)
            {
                return;
            }

            if (this->NextActionCycleCnt__.load() < this->Scheduler->getTimeStamp())
            {
                this->ActionIndex__ = this->NextActionIndex__;
            }


            (this->ProcessAction<ActionPairs.str>(), ...);
        }

    private:
        template<CTString CT>
        void ProcessAction()
        {
            constexpr size_t Idx = ActionPairs__.template index<CT>();
            if (Idx == this->ActionIndex__)
            {
                z::math::Vector<InferencePrecision, ActionElementSize__> NetOut;
                this->Scheduler->template GetData<CT>(NetOut);

                MotorVec PositionKp = this->PositionKp__[Idx];
                MotorVec VelocityKd = this->VelocityKd__[Idx];
                MotorVec ActionOffset = this->ActionOffset__[Idx];
                MotorVec ActionScale = this->ActionScale__[Idx];
                MotorVec adjusted_data = NetOut * ActionScale + ActionOffset;

                this->Scheduler->template SetData<"TargetMotorStiffness">(PositionKp);
                this->Scheduler->template SetData<"TargetMotorDamping">(VelocityKd);

                this->ActionRemapFunctions__[Idx](this->Scheduler, adjusted_data);
            }
        }

    private:
        std::array<MotorVec, sizeof...(ActionPairs)> PositionKp__;
        std::array<MotorVec, sizeof...(ActionPairs)> VelocityKd__;
        std::array<MotorVec, sizeof...(ActionPairs)> ActionOffset__;
        std::array<MotorVec, sizeof...(ActionPairs)> ActionScale__;
        CTSMap<ActionPairs...> ActionPairs__;

    };
};


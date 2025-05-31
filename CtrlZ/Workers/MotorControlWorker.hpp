/**
 * @file MotorControlWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "AbstractWorker.hpp"
#include "Schedulers/AbstractScheduler.hpp"
#include "Utils/StaticStringUtils.hpp"
#include "Utils/ZenBuffer.hpp"
#include "Utils/MathTypes.hpp"
#include <memory>
#include <iostream>


namespace z
{
    /**
     * @brief MotorPDControlWorker类型是一个电机PD控制工人类型，用户可以通过这个工人类型来实现电机的PD控制。

     * @details MotorPDControlWorker类型是一个电机PD控制工人类型，用户可以通过这个工人类型来实现电机的PD控制。
     * 通常来说如果电机本身不支持位置控制，可以在主任务队列中使用这个工人类型来实现电机的位置控制。如果电机支持位置控制，
     * 更建议使用电机内建的位置控制功能，这样可以减少控制延迟，提升控制精度。该类型需要从数据总线中读取
     * "TargetMotorPosition","TargetMotorVelocity","CurrentMotorPosition",CurrentMotorVelocity"这四个数据,并将
     * "TargetMotorTorque"这个数据写入到数据总线中。用户需要在数据总线中注册这些数据类型，以便于工人类型能够正确的读写数据。
     * 此外用户需要在配置文件中配置Kp和Kd参数。
     *
     * @details config.json配置文件示例：
     * @code {.json}
     * {
     *   "Workers": {
     *      "MotorPDLoop": {
     *         //注意这里的Kp和Kd参数需要和电机的数量相匹配
     *         "Kp": [40, 40, 40],  //位置控制参数
     *         "Kd": [1.5, 1.5, 1.5]   //速度控制参数
     *          }
     *      }
     * }
     * @endcode
     *
     * @tparam SchedulerType 调度器类型
     * @tparam MotorPrecision 电机数据精度，用户可以通过这个参数来指定电机数据的精度，比如可以指定为float或者double
     * @tparam JointNumber 电机数量，用户可以通过这个参数来指定电机的数量
     */
    template<typename SchedulerType, typename MotorPrecision, size_t JointNumber>
    class MotorPDControlWorker : public AbstractWorker<SchedulerType>
    {
        /// @brief 电机数据类型
        using MotorValVec = math::Vector<MotorPrecision, JointNumber>;
    public:
        /**
         * @brief 构造一个电机PD控制工人类型
         *
         * @param scheduler 调度器的指针
         * @param root_cfg 配置文件
         */
        MotorPDControlWorker(SchedulerType::Ptr scheduler, const nlohmann::json& root_cfg)
            :AbstractWorker<SchedulerType>(scheduler)
        {
            nlohmann::json cfg = root_cfg;
            nlohmann::json Kp_cfg = cfg["Kp"];
            nlohmann::json Kd_cfg = cfg["Kd"];
            if (Kp.size() != JointNumber || Kd_cfg.size() != JointNumber)
                throw(std::runtime_error("Joint size is not equal!"));

            for (size_t i = 0; i < JointNumber; i++)
            {
                Kp[i] = Kp_cfg[i].get<MotorPrecision>();
                Kd[i] = Kd_cfg[i].get<MotorPrecision>();
            }

            this->PrintSplitLine();
            std::cout << "MotorPDControlWorker" << std::endl;
            std::cout << "Kp=" << Kp << std::endl;
            std::cout << "Kd=" << Kd << std::endl;
            this->PrintSplitLine();
        }

        /**
         * @brief TaskRun方法，在每次任务队列循环中被调用，用来实现电机的PD控制逻辑
         *
         */
        void TaskRun()
        {
            MotorValVec TargetMotorPos;
            this->Scheduler->template GetData<"TargetMotorPosition">(TargetMotorPos);

            MotorValVec TargetMotorVel;
            this->Scheduler->template GetData<"TargetMotorVelocity">(TargetMotorVel);

            MotorValVec CurrentMotorPos;
            this->Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);

            MotorValVec CurrentMotorVel;
            this->Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel);

            MotorValVec PosErr = MotorValVec(TargetMotorPos) - MotorValVec(CurrentMotorPos);
            MotorValVec VelErr = MotorValVec(TargetMotorVel) - MotorValVec(CurrentMotorVel);

            MotorValVec Torque = Kp * PosErr + Kd * VelErr;
            this->Scheduler->template SetData<"TargetMotorTorque">(Torque);
        }

    private:
        /// @brief 位置控制参数
        MotorValVec Kp;

        /// @brief 速度控制参数
        MotorValVec Kd;
    };


    /**
     * @brief MotorControlWorker类型是一个电机控制类型，用于读取实际电机的位置、速度、电流等数据，并将控制指令写入到电机的控制接口。
     *
     * @details MotorControlWorker类型是一个电机控制类型，用于读取实际电机的位置、速度、电流等数据，并将控制指令写入到电机的控制接口。
     * 通常来说，用户可以将该类型放在主任务队列中，用于实现电机的控制逻辑。该类型会向数据总线中写入"CurrentMotorPosition",
     * "CurrentMotorVelocity","CurrentMotorTorque"这三个数据，并读取"TargetMotorPosition","TargetMotorVelocity","TargetMotorTorque"
     * 这三个数据并下发给电机，此外还会向数据总线中写入"LimitTargetMotorTorque"数据，以在力矩模式中展示被限制后的目标力矩输出。
     * 用户需要在数据总线中注册这些数据类型，以便于工人类型能够正确的读写数据。用户可以通过配置文件来配置电机的控制模式，限制力矩大小等。
     *
     * @details config.json配置文件示例：
     * @code {.json}
     * {
     *      "Workers": {
     *          "MotorControl": {
     *              //注意这里的ControlMode参数需要和电机的数量相匹配,支持的控制模式有"Torque","Position","Velocity"
     *              //若所有电机都是同一种控制模型,可以简写为一个字符串
     *              //"ControlMode":"Torque"
     *              "ControlMode": [ "Position", "Position", "Velocity", "Position", "Position", "Velocity" ],
     *              "TorqueLimit":[100,100,100,100,100,100], //非力矩模式该字段可以省略
     *              "PosFilterWeight": [0.9,0.1], //位置滤波权重
     *              "VelFilterWeight": [0.9,0.1], //速度滤波权重
     *              "DefaultPosition": [0,0,0,0,0,0] //默认位置，注意这里的默认位置需要和电机的数量相匹配
     *          }
     *      }
     * }
     * @endcode
     *
     * @tparam SchedulerType 调度器类型
     * @tparam JointType 电机类型指针，在这个类型中必须实现GetActualPosition(),GetActualTorque(),GetActualTorque(),
     * SetTargetTorque(),SetTargetPosition(),SetTargetVelocity()方法
     * @tparam MotorPrecision 电机数据精度，用户可以通过这个参数来指定电机数据的精度，比如可以指定为float或者double
     * @tparam JointNumber 关节电机数量
     */
    template<typename SchedulerType, typename JointType, typename MotorPrecision, size_t JointNumber>
    class MotorControlWorker : public AbstractWorker<SchedulerType>
    {
        /// @brief 电机数据类型
        using MotorValVec = math::Vector<MotorPrecision, JointNumber>;
    public:
        /**
         * @brief 构造一个电机控制工人类型
         *
         * @param scheduler 调度器的指针
         * @param root_cfg 配置文件
         * @param Joints 电机指针数组
         */
        MotorControlWorker(SchedulerType::Ptr scheduler, const nlohmann::json& root_cfg, const std::array<JointType, JointNumber>& Joints)
            :AbstractWorker<SchedulerType>(scheduler),
            Joints(Joints)
        {
            nlohmann::json cfg = root_cfg;
            this->PrintSplitLine();
            std::cout << "MotorControlWorker" << std::endl;
            std::cout << "JointNumber=" << JointNumber << std::endl;

            for (size_t i = 0; i < JointNumber; i++)
            {
                this->TorqueLimit[i] = 65535;
            }

            if (cfg["ControlMode"].is_string())
            {
                std::string ControlModeStr = cfg["ControlMode"].get<std::string>();
                if (ControlModeStr == "Torque")
                {
                    for (size_t i = 0; i < JointNumber; i++)
                    {
                        this->ControlModeArray[i] = ControlType::Torque;
                    }

                    nlohmann::json TorqueLimit_cfg = cfg["TorqueLimit"];
                    if (TorqueLimit_cfg.size() != JointNumber)
                        throw(std::runtime_error("TorqueLimit size is not equal!"));

                    for (size_t i = 0; i < JointNumber; i++)
                    {
                        TorqueLimit[i] = TorqueLimit_cfg[i].get<MotorPrecision>();
                    }
                    std::cout << "ControlMode=Torque" << std::endl;
                }
                else if (ControlModeStr == "Position")
                {
                    for (size_t i = 0; i < JointNumber; i++)
                    {
                        this->ControlModeArray[i] = ControlType::Position;
                    }
                    std::cout << "ControlMode=Position" << std::endl;
                }
                else if (ControlModeStr == "Velocity")
                {
                    for (size_t i = 0; i < JointNumber; i++)
                    {
                        this->ControlModeArray[i] = ControlType::Velocity;
                    }
                    std::cout << "ControlMode=Velocity" << std::endl;
                }
                else
                {
                    throw(std::runtime_error("ControlMode not supported!"));
                }

            }
            else if (cfg["ControlMode"].is_array())
            {
                if (cfg["ControlMode"].size() != JointNumber)
                    throw(std::runtime_error("ControlMode size is not equal!"));

                for (size_t i = 0; i < JointNumber; i++)
                {
                    std::string ControlModeStr = cfg["ControlMode"][i].get<std::string>();
                    if (ControlModeStr == "Torque")
                    {
                        this->ControlModeArray[i] = ControlType::Torque;
                        nlohmann::json TorqueLimit_cfg = cfg["TorqueLimit"][i];
                        TorqueLimit[i] = TorqueLimit_cfg.get<MotorPrecision>();
                        std::cout << "ControlMode=Torque" << std::endl;
                        std::cout << "TorqueLimit=" << TorqueLimit[i] << std::endl;
                    }
                    else if (ControlModeStr == "Position")
                    {
                        this->ControlModeArray[i] = ControlType::Position;
                        std::cout << "ControlMode=Position" << std::endl;
                    }
                    else if (ControlModeStr == "Velocity")
                    {
                        this->ControlModeArray[i] = ControlType::Velocity;
                        std::cout << "ControlMode=Velocity" << std::endl;
                    }
                    else
                    {
                        throw(std::runtime_error("ControlMode not supported!"));
                    }
                }
            }
            else
            {
                throw(std::runtime_error("ControlMode not supported!"));
            }


            try
            {
                std::vector<MotorValVec> PosFilterWeightVec;
                std::vector<MotorValVec> VelFilterWeightVec;
                nlohmann::json PosFilterWeight_cfg = cfg["PosFilterWeight"];
                nlohmann::json VelFilterWeight_cfg = cfg["VelFilterWeight"];

                for (auto&& val : PosFilterWeight_cfg)
                {
                    MotorPrecision weight = val.get<MotorPrecision>();
                    PosFilterWeightVec.emplace_back(MotorValVec::ones() * weight);
                }

                for (auto&& val : VelFilterWeight_cfg)
                {
                    MotorPrecision weight = val.get<MotorPrecision>();
                    VelFilterWeightVec.emplace_back(MotorValVec::ones() * weight);
                }

                this->JointPosFilter = std::make_unique<WeightFilter<MotorValVec>>(PosFilterWeightVec);
                this->JointVelFilter = std::make_unique<WeightFilter<MotorValVec>>(VelFilterWeightVec);

            }
            catch (const std::exception& e)
            {
                std::cerr << "Failed to get filter weight from config, use default value." << std::endl;
                this->JointPosFilter = std::make_unique<WeightFilter<MotorValVec>>(std::vector<MotorValVec>(1, MotorValVec::ones()));
                this->JointVelFilter = std::make_unique<WeightFilter<MotorValVec>>(std::vector<MotorValVec>(1, MotorValVec::ones()));
            }

            this->PrintSplitLine();
        }

        /**
         * @brief 析构函数
         *
         */
        ~MotorControlWorker() {}

        void TaskCreate()
        {
            MotorValVec MotorVel;
            MotorValVec MotorPos;
            MotorValVec MotorTorque;

            MotorVel.apply([this](MotorPrecision& val, size_t i) {
                val = this->Joints[i]->GetActualVelocity();
                });
            MotorPos.apply([this](MotorPrecision& val, size_t i) {
                val = this->Joints[i]->GetActualPosition();
                });
            MotorTorque.apply([this](MotorPrecision& val, size_t i) {
                val = this->Joints[i]->GetActualTorque();
                });


            this->Scheduler->template SetData<"CurrentMotorVelocityRaw">(MotorVel);
            this->Scheduler->template SetData<"CurrentMotorPositionRaw">(MotorPos);
            this->CurrentMotorVel = (*JointVelFilter)(MotorVel);
            this->CurrentMotorPos = (*JointPosFilter)(MotorPos);

            this->Scheduler->template SetData<"CurrentMotorVelocity">(CurrentMotorVel);
            this->Scheduler->template SetData<"CurrentMotorPosition">(CurrentMotorPos);
            this->Scheduler->template SetData<"CurrentMotorTorque">(MotorTorque);

            SetCurrentPositionAsTargetPosition();
        }

        /**
         * @brief 设置当前位置为目标位置
         *
         */
        void SetCurrentPositionAsTargetPosition()
        {
            MotorValVec TargetVel = MotorValVec::zeros();
            this->Scheduler->template SetData<"TargetMotorPosition">(CurrentMotorPos);
            this->Scheduler->template SetData<"TargetMotorVelocity">(TargetVel);
        }

        /**
         * @brief TaskCycleBegin方法，在每次任务队列循环开始时被调用，用来读取电机的位置、速度、电流等数据并写入到数据总线中
         *
         */
        void TaskCycleBegin() override
        {
            MotorValVec MotorVel;
            MotorValVec MotorPos;
            MotorValVec MotorTorque;

            MotorVel.apply([this](MotorPrecision& val, size_t i) {
                val = this->Joints[i]->GetActualVelocity();
                });
            MotorPos.apply([this](MotorPrecision& val, size_t i) {
                val = this->Joints[i]->GetActualPosition();
                });
            MotorTorque.apply([this](MotorPrecision& val, size_t i) {
                val = this->Joints[i]->GetActualTorque();
                });

            this->Scheduler->template SetData<"CurrentMotorVelocityRaw">(MotorVel);
            this->Scheduler->template SetData<"CurrentMotorPositionRaw">(MotorPos);
            this->CurrentMotorVel = (*JointVelFilter)(MotorVel);
            this->CurrentMotorPos = (*JointPosFilter)(MotorPos);

            this->Scheduler->template SetData<"CurrentMotorVelocity">(CurrentMotorVel);
            this->Scheduler->template SetData<"CurrentMotorPosition">(CurrentMotorPos);
            this->Scheduler->template SetData<"CurrentMotorTorque">(MotorTorque);

        }

        /**
         * @brief TaskRun方法，在每次任务队列循环中被调用，用来实现电机的控制逻辑，默认为空实现，因为电机的控制逻辑在TaskCycleEnd中实现
         */
        void TaskRun() override
        {
        }

        /**
         * @brief TaskCycleEnd方法，在每次任务队列循环结束时被调用，用来将控制指令写入到电机的控制接口。
         * 限制力矩大小，以及根据控制模式选择不同的控制方式。
         */
        void TaskCycleEnd() override
        {
            MotorValVec MotorTorque;
            this->Scheduler->template GetData<"TargetMotorTorque">(MotorTorque);

            MotorValVec MotorPos;
            this->Scheduler->template GetData<"TargetMotorPosition">(MotorPos);

            MotorValVec MotorVel;
            this->Scheduler->template GetData<"TargetMotorVelocity">(MotorVel);

            MotorValVec LimitMotorTorque = math::Vector<MotorPrecision, JointNumber>::clamp(MotorTorque, -TorqueLimit, TorqueLimit);
            this->Scheduler->template SetData<"LimitTargetMotorTorque">(LimitMotorTorque);

            for (size_t i = 0; i < JointNumber; i++)
            {
                switch (this->ControlModeArray[i])
                {
                case ControlType::Torque:
                    this->Joints[i]->SetTargetTorque(LimitMotorTorque[i]);
                    break;
                case ControlType::Position:
                    this->Joints[i]->SetTargetPosition(MotorPos[i]);
                    break;
                case ControlType::Velocity:
                    this->Joints[i]->SetTargetVelocity(MotorVel[i]);
                    break;
                default:
                    break;
                }
            }
        }

    private:
        /// @brief 控制模式
        enum class ControlType
        {
            Torque,
            Position,
            Velocity
        };

        /// @brief 电机指针数组
        std::array<JointType, JointNumber> Joints;

        /// @brief 电机位置滤波器
        std::unique_ptr<WeightFilter<MotorValVec>> JointPosFilter;

        /// @brief 电机速度滤波器
        std::unique_ptr<WeightFilter<MotorValVec>> JointVelFilter;

        /// @brief 当前电机位置
        MotorValVec CurrentMotorPos;

        /// @brief 当前电机速度
        MotorValVec CurrentMotorVel;

        /// @brief 电机力矩限制
        MotorValVec TorqueLimit;

        /// @brief 控制模式数组
        std::array<ControlType, JointNumber> ControlModeArray;
    };
};
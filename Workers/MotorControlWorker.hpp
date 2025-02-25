#pragma once
#include "AbstractWorker.hpp"
#include "Schedulers/AbstractScheduler.hpp"
#include "Utils/StaticStringUtils.hpp"
#include "Utils/ZenBuffer.hpp"
#include "Utils/MathTypes.hpp"
#include <memory>
#include <iostream>


namespace zzs
{
    template<typename SchedulerType, typename MotorPrecision, size_t JointNumber>
    class MotorPDControlWorker : public AbstractWorker<SchedulerType>
    {
        using MotorValVec = math::Vector<MotorPrecision, JointNumber>;
    public:
        MotorPDControlWorker(SchedulerType* scheduler, const nlohmann::json& root_cfg)
            :AbstractWorker<SchedulerType>(scheduler)
        {
            nlohmann::json cfg = root_cfg["Workers"]["MotorPDLoop"];
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
        MotorValVec Kp;
        MotorValVec Kd;
    };


    template<typename SchedulerType, typename JointType, typename MotorPrecision, size_t JointNumber>
    class MotorControlWorker : public AbstractWorker<SchedulerType>
    {
        using MotorValVec = math::Vector<MotorPrecision, JointNumber>;
    public:
        MotorControlWorker(SchedulerType* scheduler, const nlohmann::json& root_cfg, const std::array<JointType, JointNumber>& Joints)
            :AbstractWorker<SchedulerType>(scheduler),
            Joints(Joints)
        {
            nlohmann::json cfg = root_cfg["Workers"]["MotorControl"];
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

        ~MotorControlWorker() {}

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


            this->CurrentMotorVel = (*JointVelFilter)(MotorVel);
            this->CurrentMotorPos = (*JointPosFilter)(MotorPos);

            this->Scheduler->template SetData<"CurrentMotorVelocity">(CurrentMotorVel);
            this->Scheduler->template SetData<"CurrentMotorPosition">(CurrentMotorPos);
            this->Scheduler->template SetData<"CurrentMotorTorque">(MotorTorque);
        }

        void TaskRun() override
        {
        }

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
        enum class ControlType
        {
            Torque,
            Position,
            Velocity
        };

        std::array<JointType, JointNumber> Joints;

        std::unique_ptr<WeightFilter<MotorValVec>> JointPosFilter;
        std::unique_ptr<WeightFilter<MotorValVec>> JointVelFilter;

        MotorValVec CurrentMotorPos;
        MotorValVec CurrentMotorVel;

        MotorValVec TorqueLimit;

        std::array<ControlType, JointNumber> ControlModeArray;
    };
};
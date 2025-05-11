/**
 * @file EraxLikeInferenceWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "CommonLocoInferenceWorker.hpp"
#include "NetInferenceWorker.h"
#include "Utils/ZenBuffer.hpp"
#include "Utils/StaticStringUtils.hpp"
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace z
{
    template<typename SchedulerType, CTString NetName, typename InferencePrecision, size_t INPUT_STUCK_LENGTH, size_t EXTRA_INPUT_LENGTH, size_t JOINT_NUMBER>
    class EraxLikeInferenceWorker : public CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision, JOINT_NUMBER>
    {
    public:
        using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
        using ValVec3 = math::Vector<InferencePrecision, 3>;

    public:
        EraxLikeInferenceWorker(SchedulerType* scheduler, const nlohmann::json& Net_Config, const nlohmann::json& Motor_Config)
            :CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision, JOINT_NUMBER>(scheduler, Net_Config, Motor_Config),
            GravityVector({ 0.0,0.0,-1.0 }),
            HistoryInputBuffer(INPUT_STUCK_LENGTH)
        {
            //read cfg
            nlohmann::json InferenceCfg = Net_Config["Inference"];
            nlohmann::json NetworkCfg = Net_Config["Network"];
            this->CyctleTime = NetworkCfg["Cycle_time"].get<InferencePrecision>();
            this->dt = scheduler->getSpinOnceTime();

            this->PrintSplitLine();
            std::cout << "EraxInferenceWorker" << std::endl;
            std::cout << "JOINT_NUMBER=" << JOINT_NUMBER << std::endl;
            std::cout << "Cycle_Time=" << this->CyctleTime << std::endl;
            std::cout << "dt=" << this->dt << std::endl;
            this->PrintSplitLine();


            //concatenate all scales
            this->InputScaleVec = math::cat(
                this->Scales_ang_vel,
                this->Scales_project_gravity,
                this->Scales_dof_pos,
                this->Scales_dof_vel,
                this->Scales_last_action
            );
            this->OutputScaleVec = this->ActionScale;

            //warp input tensor
            this->InputOrtTensors__.push_back(this->WarpOrtTensor(InputTensor));
            this->OutputOrtTensors__.push_back(this->WarpOrtTensor(OutputTensor));
        }

        virtual ~EraxLikeInferenceWorker()
        {

        }

        void PreProcess() override
        {
            this->start_time = std::chrono::steady_clock::now();

            MotorValVec CurrentMotorVel;
            this->Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel);

            MotorValVec CurrentMotorPos;
            this->Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);
            CurrentMotorPos -= this->JointDefaultPos;

            MotorValVec LastAction;
            this->Scheduler->template GetData<concat(NetName, "NetLastAction")>(LastAction);

            ValVec3 UserCmd3;
            this->Scheduler->template GetData<concat(NetName, "NetUserCommand3")>(UserCmd3);
            UserCmd3 = UserCmd3 * this->Scales_command3;

            ValVec3 LinVel;
            this->Scheduler->template GetData<"LinearVelocityValue">(LinVel);

            ValVec3 AngVel;
            this->Scheduler->template GetData<"AngleVelocityValue">(AngVel);

            ValVec3 Ang;
            this->Scheduler->template GetData<"AngleValue">(Ang);

            ValVec3 ProjectedGravity = ComputeProjectedGravity(Ang, this->GravityVector);
            this->Scheduler->template SetData<concat(NetName, "NetProjectedGravity")>(ProjectedGravity);

            size_t t = this->Scheduler->getTimeStamp();
            InferencePrecision clock = 2 * M_PI / this->CyctleTime * this->dt * static_cast<InferencePrecision>(t);
            InferencePrecision clock_sin = std::sin(clock);
            InferencePrecision clock_cos = std::cos(clock);
            z::math::Vector<InferencePrecision, 2> ClockVector = { clock_sin, clock_cos };
            this->Scheduler->template SetData<concat(NetName, "NetClockVector")>(ClockVector);

            z::math::Vector< InferencePrecision, EXTRA_INPUT_LENGTH> ExtraInputVec = math::cat(
                UserCmd3, ClockVector
            );

            auto SingleInputVecScaled = math::cat(
                AngVel,
                ProjectedGravity,
                CurrentMotorPos,
                CurrentMotorVel,
                LastAction
            ) * this->InputScaleVec;

            this->HistoryInputBuffer.push(SingleInputVecScaled);


            math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH> InputVec;
            for (size_t i = 0; i < INPUT_STUCK_LENGTH; i++)
            {
                std::copy(this->HistoryInputBuffer[i].begin(), this->HistoryInputBuffer[i].end(), InputVec.begin() + i * INPUT_TENSOR_LENGTH_UNIT);
            }
            constexpr size_t EXTRA_OFFSET = INPUT_STUCK_LENGTH * INPUT_TENSOR_LENGTH_UNIT;
            std::copy(ExtraInputVec.begin(), ExtraInputVec.end(), InputVec.begin() + EXTRA_OFFSET);

            this->InputTensor.Array() = decltype(InputVec)::clamp(InputVec, -this->ClipObservation, this->ClipObservation);
        }

        void PostProcess() override
        {
            auto LastAction = this->OutputTensor.toVector();
            auto ClipedLastAction = MotorValVec::clamp(LastAction, -this->ClipAction, this->ClipAction);
            this->Scheduler->template SetData<concat(NetName, "NetLastAction")>(ClipedLastAction);

            auto ScaledAction = ClipedLastAction * this->OutputScaleVec + this->JointDefaultPos;
            this->Scheduler->template SetData<concat(NetName, "NetScaledAction")>(ScaledAction);

            auto clipedAction = MotorValVec::clamp(ScaledAction, this->JointClipLower, this->JointClipUpper);
            this->Scheduler->template SetData<concat(NetName, "Action")>(clipedAction);

            this->end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time);
            InferencePrecision inference_time = static_cast<InferencePrecision>(duration.count()) / 1000.0;
            this->Scheduler->template SetData<concat(NetName, "InferenceTime")>(inference_time);
        }

    private:
        //base base ang vel; proj grav; dof pos; dof vel; last action
        static constexpr size_t INPUT_TENSOR_LENGTH_UNIT = 3 + 3 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER;
        static constexpr size_t INPUT_TENSOR_LENGTH = INPUT_TENSOR_LENGTH_UNIT * INPUT_STUCK_LENGTH + EXTRA_INPUT_LENGTH;
        //joint number
        static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;

        //input tensor
        z::math::Tensor<InferencePrecision, 1, INPUT_TENSOR_LENGTH> InputTensor;
        z::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH_UNIT> InputScaleVec;
        z::RingBuffer<z::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH_UNIT>> HistoryInputBuffer;

        //output tensor
        z::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;
        z::math::Vector<InferencePrecision, OUTPUT_TENSOR_LENGTH> OutputScaleVec;

        const ValVec3 GravityVector;

        InferencePrecision CyctleTime;
        InferencePrecision dt;

        //compute time
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
    };
};


/**
 * @file UnitreeRlLabVelocityInferenceWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2026-01-09
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "CommonLocoInferenceWorker.hpp"
#include "NetInferenceWorker.h"
#include "Utils/ZenBuffer.hpp"
#include "Utils/StaticStringUtils.hpp"
#include <chrono>
#include <cmath>

namespace z
{
    template<typename SchedulerType, CTString NetName, typename InferencePrecision, size_t INPUT_STUCK_LENGTH, size_t JOINT_NUMBER>
    class UnitreeRlLabVelocityInferenceWorker : public CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision, JOINT_NUMBER>
    {
    public:
        using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
        using ValVec3 = math::Vector<InferencePrecision, 3>;

    public:
        UnitreeRlLabVelocityInferenceWorker(SchedulerType::Ptr scheduler, const nlohmann::json& Net_Config, const nlohmann::json& Motor_Config)
            :CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision, JOINT_NUMBER>(scheduler, Net_Config, Motor_Config),
            GravityVector({ 0.0,0.0,-1.0 }),
            ang_vel_buffer(INPUT_STUCK_LENGTH),
            projected_gravity_buffer(INPUT_STUCK_LENGTH),
            velo_cmd_buffer(INPUT_STUCK_LENGTH),
            motor_pos_buffer(INPUT_STUCK_LENGTH),
            motor_vel_buffer(INPUT_STUCK_LENGTH),
            last_action_buffer(INPUT_STUCK_LENGTH)
        {
            //read cfg
            nlohmann::json InferenceCfg = Net_Config["Inference"];
            nlohmann::json NetworkCfg = Net_Config["Network"];
            this->dt = scheduler->getSpinOnceTime();

            this->PrintSplitLine();
            std::cout << "UnitreeRlLabVelocityInferenceWorker" << std::endl;
            std::cout << "JOINT_NUMBER=" << JOINT_NUMBER << std::endl;
            std::cout << "dt=" << this->dt << std::endl;
            this->PrintSplitLine();


            //concatenate all scales

            auto Scales_ang_vel_rep = this->Scales_ang_vel.template repeat<INPUT_STUCK_LENGTH>();
            auto Scales_project_gravity_rep = this->Scales_project_gravity.template repeat<INPUT_STUCK_LENGTH>();
            auto Scales_dof_pos_rep = this->Scales_dof_pos.template repeat<INPUT_STUCK_LENGTH>();
            auto Scales_dof_vel_rep = this->Scales_dof_vel.template repeat<INPUT_STUCK_LENGTH>();
            auto Scales_last_action_rep = this->Scales_last_action.template repeat<INPUT_STUCK_LENGTH>();
            auto Scales_command3_rep = ValVec3::ones().template repeat<INPUT_STUCK_LENGTH>();


            this->InputScaleVec = math::cat(
                Scales_ang_vel_rep,
                Scales_project_gravity_rep,
                Scales_command3_rep,
                Scales_dof_pos_rep,
                Scales_dof_vel_rep,
                Scales_last_action_rep
            );

            this->OutputScaleVec = this->ActionScale;

            //warp input tensor
            this->InputOrtTensors__.push_back(this->WarpOrtTensor(InputTensor));
            this->OutputOrtTensors__.push_back(this->WarpOrtTensor(OutputTensor));
        }

        virtual ~UnitreeRlLabVelocityInferenceWorker()
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

            // ValVec3 LinVel;
            // this->Scheduler->template GetData<"LinearVelocityValue">(LinVel);

            ValVec3 AngVel;
            this->Scheduler->template GetData<"AngleVelocityValue">(AngVel);

            ValVec3 Ang;
            this->Scheduler->template GetData<"AngleValue">(Ang);

            ValVec3 ProjectedGravity = ComputeProjectedGravity(Ang, this->GravityVector);
            this->Scheduler->template SetData<concat(NetName, "NetProjectedGravity")>(ProjectedGravity);


            this->ang_vel_buffer.push(AngVel);
            this->projected_gravity_buffer.push(ProjectedGravity);
            this->velo_cmd_buffer.push(UserCmd3);
            this->motor_pos_buffer.push(CurrentMotorPos);
            this->motor_vel_buffer.push(CurrentMotorVel);
            this->last_action_buffer.push(LastAction);

            // std::cout << "Input Buffer State:" << std::endl;
            // std::cout << "AngVel Buffer: " << this->ang_vel_buffer;

            auto ang_vel_stacked = this->stackBuffer(this->ang_vel_buffer);
            auto projected_gravity_stacked = this->stackBuffer(this->projected_gravity_buffer);
            auto velo_cmd_stacked = this->stackBuffer(this->velo_cmd_buffer);
            auto motor_pos_stacked = this->stackBuffer(this->motor_pos_buffer);
            auto motor_vel_stacked = this->stackBuffer(this->motor_vel_buffer);
            auto last_action_stacked = this->stackBuffer(this->last_action_buffer);

            auto InputVec = math::cat(
                ang_vel_stacked,
                projected_gravity_stacked,
                velo_cmd_stacked,
                motor_pos_stacked,
                motor_vel_stacked,
                last_action_stacked
            ) * this->InputScaleVec;

            this->InputTensor.Array() = z::math::clamp(InputVec, -this->ClipObservation, this->ClipObservation);
        }

        template<typename Scalar, size_t N>
        z::math::Vector<Scalar, N* INPUT_STUCK_LENGTH> stackBuffer(z::RingBuffer<z::math::Vector<Scalar, N>> buffer)
        {
            z::math::Vector<Scalar, N* INPUT_STUCK_LENGTH> result;
            for (size_t i = 0; i < INPUT_STUCK_LENGTH; i++)
            {
                std::copy(buffer[i].begin(), buffer[i].end(), result.begin() + i * N);
            }
            return result;
        }

        void PostProcess() override
        {
            auto LastAction = this->OutputTensor.toVector();
            auto ClipedLastAction = z::math::clamp(LastAction, -this->ClipAction, this->ClipAction);
            this->Scheduler->template SetData<concat(NetName, "NetLastAction")>(ClipedLastAction);

            auto ScaledAction = ClipedLastAction * this->OutputScaleVec + this->JointDefaultPos;
            //this->Scheduler->template SetData<concat(NetName, "NetScaledAction")>(ScaledAction);

            auto clipedAction = z::math::clamp(ScaledAction, this->JointClipLower, this->JointClipUpper);
            this->Scheduler->template SetData<concat(NetName, "Action")>(clipedAction);

            this->end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time);
            InferencePrecision inference_time = static_cast<InferencePrecision>(duration.count()) / 1000.0;
            this->Scheduler->template SetData<concat(NetName, "InferenceTime")>(inference_time);
        }

    private:
        //base ang vel; proj grav;velo_cmd; dof pos; dof vel; last action
        static constexpr size_t INPUT_TENSOR_LENGTH_UNIT = 3 + 3 + 3 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER;
        static constexpr size_t INPUT_TENSOR_LENGTH = INPUT_TENSOR_LENGTH_UNIT * INPUT_STUCK_LENGTH;
        //joint number
        static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;

        //input tensor
        z::math::Tensor<InferencePrecision, 1, INPUT_TENSOR_LENGTH> InputTensor;
        z::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH> InputScaleVec;

        z::RingBuffer<ValVec3> ang_vel_buffer;
        z::RingBuffer<ValVec3> projected_gravity_buffer;
        z::RingBuffer<ValVec3> velo_cmd_buffer;
        z::RingBuffer<MotorValVec> motor_pos_buffer;
        z::RingBuffer<MotorValVec> motor_vel_buffer;
        z::RingBuffer<MotorValVec> last_action_buffer;

        //output tensor
        z::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;
        z::math::Vector<InferencePrecision, OUTPUT_TENSOR_LENGTH> OutputScaleVec;

        const ValVec3 GravityVector;

        InferencePrecision dt;

        //compute time
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
    };
};


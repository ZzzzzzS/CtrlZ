/**
 * @file ArmTrackingInferenceWorker.hpp
 * @author Zishun Zhou
 * @brief 机械臂控制程序所需控制网络
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "Workers/NN/CommonLocoInferenceWorker.hpp"
#include "Workers/NN/NetInferenceWorker.h"
#include "chrono"

namespace z
{
    template<typename SchedulerType, typename InferencePrecision, size_t JOINT_NUMBER>
    class ArmTrackingInferenceWorker : public CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>
    {
    public:
        using Base = CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>;
        /// @brief 电机向量类型
        using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
        /// @brief 七维向量类型
        using ValVec7 = math::Vector<InferencePrecision, 7>;
    public:
        /**
         * @brief 构造一个ArmTrackingInferenceWorker类型
         *
         * @param scheduler 调度器的指针
         * @param cfg 配置文件
         */
        ArmTrackingInferenceWorker(SchedulerType* scheduler, const nlohmann::json& cfg)
            :CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>(scheduler, cfg)
        {
            //concatenate all scales
            ValVec7 Scales_Cmd = ValVec7::Ones();
            this->InputScaleVec = math::cat(
                Scales_Cmd,
                this->Scales_dof_pos,
                this->Scales_dof_vel,
                this->Scales_last_action
            );
            this->OutputScaleVec = this->ActionScale;

            //warp input tensor
            this->InputOrtTensors__.push_back(this->WarpOrtTensor(InputTensor));
            this->OutputOrtTensors__.push_back(this->WarpOrtTensor(OutputTensor));
        }

        /**
         * @brief 析构函数
         *
         */
        virtual ~ArmTrackingInferenceWorker()
        {

        }

        /**
         * @brief 推理前的准备工作,主要是将数据从数据总线中读取出来，并将数据缩放到合适的范围
         *
         */
        void PreProcess() override
        {
            this->start_time = std::chrono::steady_clock::now();

            MotorValVec CurrentMotorVel;
            this->Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel);

            MotorValVec CurrentMotorPos;
            this->Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);
            CurrentMotorPos -= this->JointDefaultPos;

            MotorValVec LastAction;
            this->Scheduler->template GetData<"NetLastAction">(LastAction);

            ValVec7 UserCmd;
            this->Scheduler->template GetData<"NetUserCommand">(UserCmd);


            auto InputVecScaled = math::cat(
                UserCmd,
                CurrentMotorPos,
                CurrentMotorVel,
                LastAction
            ) * this->InputScaleVec;

            this->InputTensor.Array() = decltype(this->InputScaleVec)::clamp(InputVecScaled, -this->ClipObservation, this->ClipObservation);
        }

        /**
         * @brief 推理后的处理工作,主要是将推理的结果从数据总线中读取出来，并将数据缩放到合适的范围
         * 并将推理的结果写入到数据总线中
         */
        void PostProcess() override
        {
            auto LastAction = this->OutputTensor.toVector();
            auto ClipedLastAction = MotorValVec::clamp(LastAction, -this->ClipAction, this->ClipAction);
            this->Scheduler->template SetData<"NetLastAction">(ClipedLastAction);

            auto ScaledAction = ClipedLastAction * this->OutputScaleVec + this->JointDefaultPos;
            this->Scheduler->template SetData<"NetScaledAction">(ScaledAction);

            auto clipedAction = MotorValVec::clamp(ScaledAction, this->JointClipLower, this->JointClipUpper);
            this->Scheduler->template SetData<"TargetMotorPosition">(clipedAction);

            this->end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time);
            InferencePrecision inference_time = static_cast<InferencePrecision>(duration.count());
            this->Scheduler->template SetData<"InferenceTime">(inference_time);
        }

    private:
        //base lin vel; base ang vel; proj grav; cmd7; dof pos; dof vel; last action
        static constexpr size_t INPUT_TENSOR_LENGTH = 7 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER;
        //joint number
        static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;

        //input tensor
        z::math::Tensor<InferencePrecision, 1, INPUT_TENSOR_LENGTH> InputTensor;
        z::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH> InputScaleVec;


        //output tensor
        z::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;
        z::math::Vector<InferencePrecision, OUTPUT_TENSOR_LENGTH> OutputScaleVec;

        //compute time
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
    };
};


/**
 * @file PlainInferenceWorker.hpp
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
#include "chrono"
#include "Utils/StaticStringUtils.hpp"

namespace z
{
    /**
     * @brief 该类型实现了一个可用的平凡推理工人类型，用户可以通过这个工人类型来实现推理的功能。
     * @details 该类型实现了一个可用的平凡推理工人类型，用户可以通过这个工人类型来实现推理的功能。
     * 该类型网络输入包括：线速度，角速度，重力投影，用户命令，关节位置，关节速度，上一次动作，网络输出为动作。
     * 该类型需要从数据总线中读取"CurrentMotorVelocity","CurrentMotorPosition","NetLastAction","NetUserCommand3",
     * "LinearVelocityValue","AngleVelocityValue","AngleValue"这些数据，并将"NetLastAction","NetProjectedGravity","NetScaledAction",
     * "TargetMotorPosition","InferenceTime"这些数据写入到数据总线中。用户需要在数据总线中注册这些数据类型，以便于工人类型能够正确的读写数据。
     * 用户可以通过配置文件来配置推理的参数，比如观测量的缩放(observation scale)，动作量的缩放(action scale)，
     * 观测量的裁剪(observation clip)，动作量的裁剪(action clip)等。
     *
     * @details config.json配置文件示例：
     * 该类型无新增配置，具体配置请参考CommonLocoInferenceWorker中的配置。
     *
     * @details 网络输入：线速度(3)，角速度(3)，重力投影(3)，用户命令(3)，关节位置(N)，关节速度(N)，上一次动作(N)
     * 网络输出：动作(N), N为关节数量
     *
     * @tparam SchedulerType 调度器类型
     * @tparam NetName 网络名称，用户可以通过这个参数来指定网络的名称, 这在有多个网络时可以区分数据总线上的不同网络数据
     * @tparam InferencePrecision 推理精度，用户可以通过这个参数来指定推理的精度，比如可以指定为float或者double
     * @tparam JOINT_NUMBER 关节数量
     */
    template<typename SchedulerType, CTString NetName, typename InferencePrecision, size_t JOINT_NUMBER>
    class PlainInferenceWorker : public CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision, JOINT_NUMBER>
    {
    public:
        using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
        using ValVec3 = math::Vector<InferencePrecision, 3>;

    public:
        /**
         * @brief 构造一个PlainInferenceWorker类型
         *
         * @param scheduler 调度器的指针
         * @param cfg 配置文件
         */
        PlainInferenceWorker(SchedulerType::Ptr scheduler, const nlohmann::json& Net_cfg, const nlohmann::json& Motor_cfg)
            :CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision, JOINT_NUMBER>(scheduler, Net_cfg, Motor_cfg),
            GravityVector({ 0.0,0.0,-1.0 })
        {
            //concatenate all scales
            this->InputScaleVec = math::cat(
                this->Scales_lin_vel,
                this->Scales_ang_vel,
                this->Scales_project_gravity,
                this->Scales_command3,
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
        virtual ~PlainInferenceWorker()
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
            this->Scheduler->template GetData<concat(NetName, "NetLastAction")>(LastAction);

            ValVec3 UserCmd3;
            this->Scheduler->template GetData<concat(NetName, "NetUserCommand3")>(UserCmd3);

            ValVec3 LinVel;
            this->Scheduler->template GetData<"LinearVelocityValue">(LinVel);

            ValVec3 AngVel;
            this->Scheduler->template GetData<"AngleVelocityValue">(AngVel);

            ValVec3 Ang;
            this->Scheduler->template GetData<"AngleValue">(Ang);

            ValVec3 ProjectedGravity = ComputeProjectedGravity(Ang, this->GravityVector);
            this->Scheduler->template SetData<concat(NetName, "NetProjectedGravity")>(ProjectedGravity);


            auto InputVecScaled = math::cat(
                LinVel,
                AngVel,
                ProjectedGravity,
                UserCmd3,
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
            this->Scheduler->template SetData<concat(NetName, "NetLastAction")>(ClipedLastAction);

            auto ScaledAction = ClipedLastAction * this->OutputScaleVec + this->JointDefaultPos;
            this->Scheduler->template SetData<concat(NetName, "NetScaledAction")>(ScaledAction);

            auto clipedAction = MotorValVec::clamp(ScaledAction, this->JointClipLower, this->JointClipUpper);
            this->Scheduler->template SetData<concat(NetName, "Action")>(clipedAction);

            this->end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time);
            InferencePrecision inference_time = static_cast<InferencePrecision>(duration.count());
            this->Scheduler->template SetData<concat(NetName, "InferenceTime")>(inference_time);
        }

    private:
        //base lin vel; base ang vel; proj grav; cmd3; dof pos; dof vel; last action
        static constexpr size_t INPUT_TENSOR_LENGTH = 3 + 3 + 3 + 3 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER;
        //joint number
        static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;

        //input tensor
        z::math::Tensor<InferencePrecision, 1, INPUT_TENSOR_LENGTH> InputTensor;
        z::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH> InputScaleVec;


        //output tensor
        z::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;
        z::math::Vector<InferencePrecision, OUTPUT_TENSOR_LENGTH> OutputScaleVec;

        /// @brief 重力向量{0,0,-1}
        const ValVec3 GravityVector;

        //compute time
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
    };
};


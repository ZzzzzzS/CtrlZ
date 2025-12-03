/**
 * @file GeneralMimicWorker.hpp
 * @author Zhou Zishun
 * @brief TBD
 *
 * @date 2025
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "CommonLocoInferenceWorker.hpp"
#include "NetInferenceWorker.h"
#include "Utils/ZenBuffer.hpp"
#include "Utils/StaticStringUtils.hpp"
#include "Utils/CSVReader.hpp"
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace z
{
    template<typename SchedulerType, CTString NetName, typename InferencePrecision, size_t JOINT_NUMBER, size_t TRAJ_LENGTH>
    class BeyondMimicUnitreeInferenceWorker : public AbstractNetInferenceWorker<SchedulerType, NetName, InferencePrecision>
    {
    public:
        using Base = AbstractNetInferenceWorker<SchedulerType, NetName, InferencePrecision>;
        using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
        using Vec3 = math::Vector<InferencePrecision, 3>;
        using Vec6 = math::Vector<InferencePrecision, 6>;
        using Vec4 = math::Vector<InferencePrecision, 4>;

    public:
        /**
         * @brief 构造一个BeyondMimicUnitreeInferenceWorker类型
         *
         * @param scheduler 调度器的指针
         * @param cfg 配置文件
         */
        BeyondMimicUnitreeInferenceWorker(SchedulerType::Ptr scheduler, const nlohmann::json& Net_cfg, const nlohmann::json& Motor_cfg, const std::array<size_t, JOINT_NUMBER>& joint_id_map)
            :Base(scheduler, Net_cfg), JointIdMap(joint_id_map)
        {
            //compute reverse joint id map
            for (size_t i = 0; i < JOINT_NUMBER; i++)
            {
                this->ReverseJointIdMap[this->JointIdMap[i]] = i;
            }

            //read cfg
            nlohmann::json InferenceCfg = Net_cfg["Inference"];
            nlohmann::json NetworkCfg = Net_cfg["Network"];
            nlohmann::json PreprocessCfg = Net_cfg["Preprocess"];
            nlohmann::json PostprocessCfg = Net_cfg["Postprocess"];
            this->cycle_time = NetworkCfg["Cycle_time"].get<InferencePrecision>();
            this->dt = scheduler->getSpinOnceTime();

            this->PrintSplitLine();
            std::cout << "BeyondMimicUnitreeInferenceWorker" << std::endl;
            std::cout << "JOINT_NUMBER=" << JOINT_NUMBER << std::endl;
            std::cout << "TRAJECTORY_LENGTH=" << TRAJ_LENGTH << std::endl;
            std::cout << "Cycle_time=" << this->cycle_time << std::endl;
            std::cout << "dt=" << this->dt << std::endl;
            std::cout << "Trajectory Path=" << PreprocessCfg["trajectory_path"].get<std::string>() << std::endl;
            this->PrintSplitLine();

            //concatenate all scales and joint default pos
            this->InputScaleVec = decltype(this->InputScaleVec)::ones();
            for (size_t i = 0; i < JOINT_NUMBER;i++)
            {
                this->OutputScaleVec[i] = PostprocessCfg["action_scale"][i].get<InferencePrecision>();
            }
            if (Motor_cfg["DefaultPosition"].size() != JOINT_NUMBER)
                throw(std::runtime_error("default_pos size is not equal!"));
            for (size_t i = 0; i < JOINT_NUMBER; i++)
            {
                this->JointDefaultPos[i] = Motor_cfg["DefaultPosition"][i].get<InferencePrecision>();
            }


            //load trajectory tensor
            this->LoadTrajectoryTensor(PreprocessCfg["trajectory_path"].get<std::string>(), PreprocessCfg["include_velocity"].get<bool>());


            //warp input tensor
            this->InputOrtTensors__.push_back(this->WarpOrtTensor(this->InputTensor));
            this->OutputOrtTensors__.push_back(this->WarpOrtTensor(this->OutputTensor));
        }

        /**
         * @brief 析构函数
         *
         */
        virtual ~BeyondMimicUnitreeInferenceWorker()
        {

        }

        /**
         * @brief 开始运行跳舞轨迹
         *
         */
        void start()
        {
            this->compute_init_quat();
            this->start_timestamp = this->Scheduler->getTimeStamp();
            this->started = true;
        }

        /**
         * @brief 推理前的准备工作,主要是将数据从数据总线中读取出来，并将数据缩放到合适的范围
         * 构造堆叠的输入数据，并准备好输入张量。
         *
         */
        void PreProcess() override
        {
            this->start_time = std::chrono::steady_clock::now();

            MotorValVec CurrentMotorVel;
            this->Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel);

            MotorValVec CurrentMotorPos;
            MotorValVec CurrentMotorPosRaw;
            this->Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);
            CurrentMotorPosRaw = CurrentMotorPos;
            CurrentMotorPos -= this->JointDefaultPos;

            MotorValVec LastAction;
            this->Scheduler->template GetData<concat(NetName, "NetLastAction")>(LastAction);

            Vec3 AngVel;
            this->Scheduler->template GetData<"AngleVelocityValue">(AngVel);

            Vec3 Ang;
            this->Scheduler->template GetData<"AngleValue">(Ang);

            if (this->first_run == true)
            {
                this->compute_init_quat();
                this->first_run = false;
            }

            Vec4 torso_quat = this->compute_torso_quat_from_base_inertial_measurement_unit_measurement();

            MotorValVec ref_joint_pos;
            MotorValVec ref_joint_vel;
            Vec3 ref_root_pos;
            Vec4 ref_root_quat;
            this->ComputeReference(
                this->Scheduler->getTimeStamp(),
                ref_joint_pos,
                ref_joint_vel,
                ref_root_pos,
                ref_root_quat
            );

            this->Scheduler->template SetData<concat(NetName, "RefTraj")>(ref_joint_pos);
            this->Scheduler->template SetData<concat(NetName, "RefVel")>(ref_joint_vel);

            Vec6 motion_anchor_ori_b = this->compute_motion_anchor_ori_b(
                ref_root_quat,
                torso_quat
            );

            auto SingleInputVecScaled = math::cat(
                ref_joint_pos,
                ref_joint_vel,
                motion_anchor_ori_b,
                AngVel,
                CurrentMotorPos,
                CurrentMotorVel,
                LastAction
            ) * this->InputScaleVec;

            this->InputTensor.Array() = SingleInputVecScaled;

        }

        /**
         * @brief 推理后的处理工作,主要是将推理的结果从数据总线中读取出来，并将数据缩放到合适的范围
         *
         */
        void PostProcess() override
        {
            auto LastAction = this->OutputTensor.toVector();
            this->Scheduler->template SetData<concat(NetName, "NetLastAction")>(LastAction);

            auto ScaledAction = LastAction * this->OutputScaleVec + this->JointDefaultPos;

            MotorValVec MotorPositionTarget;
            this->Scheduler->template GetData<concat(NetName, "RefTraj")>(MotorPositionTarget);
            //MotorPositionTarget += this->JointDefaultPos;

            this->Scheduler->template SetData<concat(NetName, "Action")>(ScaledAction);
            //this->Scheduler->template SetData<concat(NetName, "Action")>(MotorPositionTarget);

            this->end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time);
            InferencePrecision inference_time = static_cast<InferencePrecision>(duration.count()) / 1000.0;
            this->Scheduler->template SetData<concat(NetName, "InferenceTime")>(inference_time);
        }

    private:
        Vec4 yawQuaternion(const Vec4& q) {
            float yaw = std::atan2(2.0f * (q[3] * q[2] + q[0] * q[1]), 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]));
            float half_yaw = yaw * 0.5f;
            return Vec4({ 0.0f, 0.0f, std::sin(half_yaw),std::cos(half_yaw) });
        };

        void compute_init_quat()
        {
            Vec3 Ang;
            this->Scheduler->template GetData<"AngleValue">(Ang);

            MotorValVec CurrentMotorPos;
            this->Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);

            Vec4 current_quat = this->compute_torso_quat_from_base_inertial_measurement_unit_measurement();

            auto robot_yaw = this->yawQuaternion(current_quat);
            Vec4 init_quat_e({ this->TrajectoryTensor(0, 3), this->TrajectoryTensor(0, 4), this->TrajectoryTensor(0, 5),this->TrajectoryTensor(0, 6) });
            auto ref_yaw = this->yawQuaternion(init_quat_e);

            Vec4 init_quat_ = z::math::quat_mul(robot_yaw, z::math::quat_conjugate(ref_yaw));

            this->init_quat = init_quat_;
        }

        void LoadTrajectoryTensor(const std::string& path, bool include_velocity = false)
        {
            auto Reader = z::CSVReader<InferencePrecision>::Create(path, false);
            if (Reader->RowSize() != TRAJ_LENGTH)
            {
                std::cout << "Trajectory length not match!" << std::endl;
                throw(std::runtime_error("Trajectory length not match!"));
            }
            for (size_t i = 0; i < TRAJ_LENGTH; i++)
            {
                auto row = Reader->getRow(i);
                //root pos
                this->TrajectoryTensor(i, 0) = static_cast<InferencePrecision>(row[0]);
                this->TrajectoryTensor(i, 1) = static_cast<InferencePrecision>(row[1]);
                this->TrajectoryTensor(i, 2) = static_cast<InferencePrecision>(row[2]);
                //root quat
                this->TrajectoryTensor(i, 3) = static_cast<InferencePrecision>(row[3]);
                this->TrajectoryTensor(i, 4) = static_cast<InferencePrecision>(row[4]);
                this->TrajectoryTensor(i, 5) = static_cast<InferencePrecision>(row[5]);
                this->TrajectoryTensor(i, 6) = static_cast<InferencePrecision>(row[6]);
                //joint pos and vel
                for (size_t j = 0; j < JOINT_NUMBER; j++)
                {
                    this->TrajectoryTensor(i, 7 + j) = static_cast<InferencePrecision>(row[7 + j]);
                    if (include_velocity)
                    {
                        this->TrajectoryTensor(i, 7 + JOINT_NUMBER + j) = static_cast<InferencePrecision>(row[7 + JOINT_NUMBER + j]);
                    }
                    else
                    {
                        if (i == 0)
                            this->TrajectoryTensor(i, 7 + JOINT_NUMBER + j) = 0.0;
                        else
                        {
                            InferencePrecision pos_now = static_cast<InferencePrecision>(row[7 + j]);
                            InferencePrecision pos_last = this->TrajectoryTensor(i - 1, 7 + j);
                            InferencePrecision vel = (pos_now - pos_last) / this->cycle_time;
                            this->TrajectoryTensor(i, 7 + JOINT_NUMBER + j) = vel;
                        }
                    }
                }
            }
        }

        void ComputeReference(size_t time_stamp,
            math::Vector<InferencePrecision, JOINT_NUMBER>& ref_joint_pos,
            math::Vector<InferencePrecision, JOINT_NUMBER>& ref_joint_vel,
            math::Vector<InferencePrecision, 3>& ref_root_pos,
            math::Vector<InferencePrecision, 4>& ref_root_quat)
        {
            InferencePrecision time_in_cycle = static_cast<InferencePrecision>(time_stamp - this->start_timestamp) * this->dt;
            // index = time_in_cycle / (this->cycle_time * TRAJ_LENGTH);
            InferencePrecision index_f = time_in_cycle / (this->cycle_time);
            size_t index = static_cast<size_t>(std::floor(index_f));

            if (this->started == false)
                index = 0;

            // std::cout << "time_stamp: " << time_stamp << ", time_in_cycle: " << time_in_cycle << ", index: " << index << "traj:" << (this->cycle_time * TRAJ_LENGTH) << std::endl;
            // fflush(stdout);
            if (index >= TRAJ_LENGTH)
                index = TRAJ_LENGTH - 1;
            else if (index < 0)
                index = 0;

            //root pos
            ref_root_pos[0] = this->TrajectoryTensor(index, 0);
            ref_root_pos[1] = this->TrajectoryTensor(index, 1);
            ref_root_pos[2] = this->TrajectoryTensor(index, 2);

            //root quat
            ref_root_quat[0] = this->TrajectoryTensor(index, 3);
            ref_root_quat[1] = this->TrajectoryTensor(index, 4);
            ref_root_quat[2] = this->TrajectoryTensor(index, 5);
            ref_root_quat[3] = this->TrajectoryTensor(index, 6);

            Vec4 quat_z = z::math::quat_from_euler_xyz(Vec3({ 0.0f, 0.0f, this->TrajectoryTensor(index, 12 + 7) }));
            Vec4 quat_y = z::math::quat_from_euler_xyz(Vec3({ 0.0f, this->TrajectoryTensor(index, 14 + 7), 0.0f }));
            Vec4 quat_x = z::math::quat_from_euler_xyz(Vec3({ this->TrajectoryTensor(index, 13 + 7), 0.0f, 0.0f }));
            Vec4 yaw_quat = z::math::quat_mul(z::math::quat_mul(quat_z, quat_x), quat_y);
            ref_root_quat = z::math::quat_mul(ref_root_quat, yaw_quat);

            //joint pos and vel
            for (size_t j = 0; j < JOINT_NUMBER; j++)
            {
                ref_joint_pos[j] = this->TrajectoryTensor(index, 7 + this->JointIdMap[j]);
                ref_joint_vel[j] = this->TrajectoryTensor(index, 7 + JOINT_NUMBER + this->JointIdMap[j]);
            }
        }

        Vec6 compute_motion_anchor_ori_b(const Vec4& ref_quat, const Vec4& torso_quat)
        {
            Vec4 rot_e = z::math::quat_mul(this->init_quat, ref_quat);
            rot_e = z::math::quat_conjugate(rot_e);
            rot_e = z::math::quat_mul(rot_e, torso_quat);
            z::math::Tensor<InferencePrecision, 3, 3> rot_t = z::math::toRotationMatrix(rot_e);
            Vec6 data;
            data[0] = rot_t(0, 0);
            data[1] = rot_t(1, 0);
            data[2] = rot_t(0, 1);
            data[3] = rot_t(1, 1);
            data[4] = rot_t(0, 2);
            data[5] = rot_t(1, 2);
            return data;
        }

        Vec4 compute_torso_quat_from_base_inertial_measurement_unit_measurement()
        {
            Vec3 imu_angle;
            this->Scheduler->template GetData<"AngleValue">(imu_angle);
            MotorValVec joint_positions;
            this->Scheduler->template GetData<"CurrentMotorPosition">(joint_positions);
            Vec4 torso_quat;

            Vec4 imu_quat_e = z::math::quat_from_euler_xyz(imu_angle);
            Vec4 quat_z = z::math::quat_from_euler_xyz(Vec3({ 0.0f, 0.0f, joint_positions[this->ReverseJointIdMap[12]] }));
            Vec4 quat_y = z::math::quat_from_euler_xyz(Vec3({ 0.0f, joint_positions[this->ReverseJointIdMap[14]], 0.0f }));
            Vec4 quat_x = z::math::quat_from_euler_xyz(Vec3({ joint_positions[this->ReverseJointIdMap[13]], 0.0f, 0.0f }));
            Vec4 waist_quat_e = z::math::quat_mul(z::math::quat_mul(quat_z, quat_x), quat_y);
            torso_quat = z::math::quat_mul(imu_quat_e, waist_quat_e);

            return torso_quat;
        }

    private:
        size_t start_timestamp = 0;
        bool started = false;

        //clock; usercmd;q;dq;act;angle vel;euler xyz;
        static constexpr size_t INPUT_TENSOR_LENGTH = JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER + 3 + 6;
        //joint number
        static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;

        //trajectory tensor
        z::math::Tensor<InferencePrecision, TRAJ_LENGTH, JOINT_NUMBER * 2 + 7> TrajectoryTensor; //root{pos,quat}, joint{pos,vel}

        //input tensor
        z::math::Tensor<InferencePrecision, 1, INPUT_TENSOR_LENGTH> InputTensor;
        z::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH> InputScaleVec;

        //output tensor
        z::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;
        z::math::Vector<InferencePrecision, OUTPUT_TENSOR_LENGTH> OutputScaleVec;

        //cycle time and dt
        InferencePrecision cycle_time;
        InferencePrecision dt;

        //compute time
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;

        MotorValVec JointDefaultPos;
        Vec4 init_quat;
        bool first_run = true;
        std::array<size_t, JOINT_NUMBER> JointIdMap;
        std::array<size_t, JOINT_NUMBER> ReverseJointIdMap;
    };
};


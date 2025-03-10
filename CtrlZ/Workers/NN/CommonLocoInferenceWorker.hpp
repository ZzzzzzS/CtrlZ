/**
 * @file CommonLocoInferenceWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "AbstractInferenceWorker.hpp"
#include "onnxruntime_cxx_api.h"
#include "Utils/MathTypes.hpp"
#include "nlohmann/json.hpp"

namespace z
{
    /**
     * @brief CommonLocoInferenceWorker类型是一个通用的Locomotion推理工人类型，该类型实现了一些Locomotion推理的通用逻辑。
     * @details CommonLocoInferenceWorker类型是一个通用的Locomotion推理工人类型，该类型实现了一些Locomotion推理的通用逻辑。
     * 该类型主要从配置文件中读取一些推理的参数，比如观测量的缩放(observation scale)，动作量的缩放(action scale)，
     * 观测量的裁剪(observation clip)，动作量的裁剪(action clip)等。
     *
     * @details config.json配置文件示例：
     * {
     *   "Workers": {
     *     "NN": {
     *      "Preprocess": {
     *          "ObservationScales": {
     *            "lin_vel": 1.0,
     *            "ang_vel": 1.0,
     *            "project_gravity": 1.0,
     *            "dof_pos": 1.0,
     *            "dof_vel": 1.0
     *           },
     *          "ClipObservations": 100.0
     *       },
     *       "Postprocess": {
     *          "action_scale": [1.0, 1.0, 1.0，1.0, 1.0, 1.0], //需要和动作的维度相匹配
     *          "clip_actions": 1.0,
     *          "joint_clip_upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], //需要和动作的维度相匹配
     *          "joint_clip_lower": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0] //需要和动作的维度相匹配
     *       }
     *     }
     *     "MotorControl": {
     *         "DefaultPosition": [0, 0, 0, 0, 0, 0] //默认位置，注意这里的默认位置需要和动作的维度相匹配
     *     }
     *   }
     * }
     *
     * @tparam SchedulerType 调度器类型
     * @tparam InferencePrecision 推理精度，用户可以通过这个参数来指定推理的精度，比如可以指定为float或者double
     * @tparam JOINT_NUMBER 关节数量
     */
    template<typename SchedulerType, typename InferencePrecision, size_t JOINT_NUMBER>
    class CommonLocoInferenceWorker : public AbstractNetInferenceWorker<SchedulerType, InferencePrecision>
    {
        static_assert(std::is_arithmetic<InferencePrecision>::value, "InferencePrecision must be a arithmetic type");
    public:
        using Base = AbstractNetInferenceWorker<SchedulerType, InferencePrecision>;
        using Base::Session__;
        using Base::MemoryInfo__;
        using Base::DefaultAllocator__;
        using Base::InputNodeNames__;
        using Base::OutputNodeNames__;
        using Base::InputOrtTensors__;
        using Base::OutputOrtTensors__;
    public:
        /**
         * @brief 构造一个CommonLocoInferenceWorker类型
         *
         * @param scheduler 调度器的指针
         * @param cfg 配置文件
         */
        CommonLocoInferenceWorker(SchedulerType* scheduler, const nlohmann::json& cfg)
            :AbstractNetInferenceWorker<SchedulerType, InferencePrecision>(scheduler, cfg)
        {
            nlohmann::json PreprocessCfg = cfg["Workers"]["NN"]["Preprocess"];
            nlohmann::json PostprocessCfg = cfg["Workers"]["NN"]["Postprocess"];

            //load pre process cfg
            //load obs scale
            nlohmann::json obs_scale_cfg = PreprocessCfg["ObservationScales"];

            InferencePrecision scales_lin_vel = obs_scale_cfg["lin_vel"].get<InferencePrecision>();
            InferencePrecision scales_ang_vel = obs_scale_cfg["ang_vel"].get<InferencePrecision>();
            InferencePrecision scales_project_gravity = obs_scale_cfg["project_gravity"].get<InferencePrecision>();
            InferencePrecision scales_dof_pos = obs_scale_cfg["dof_pos"].get<InferencePrecision>();
            InferencePrecision scales_dof_vel = obs_scale_cfg["dof_vel"].get<InferencePrecision>();

            this->Scales_lin_vel = ValVec3::ones() * scales_lin_vel;
            this->Scales_ang_vel = ValVec3::ones() * scales_ang_vel;
            this->Scales_project_gravity = ValVec3::ones() * scales_project_gravity;
            this->Scales_command3 = { scales_lin_vel,scales_lin_vel ,scales_ang_vel };
            this->Scales_dof_pos = MotorValVec::ones() * scales_dof_pos;
            this->Scales_dof_vel = MotorValVec::ones() * scales_dof_vel;
            this->Scales_last_action = MotorValVec::ones();

            //load obs clip
            this->ClipObservation = PreprocessCfg["ClipObservations"].get<InferencePrecision>();

            //load post process cfg
            //load act scale
            if (PostprocessCfg["action_scale"].size() != JOINT_NUMBER)
                throw(std::runtime_error("action_scale size is not equal!"));
            for (size_t i = 0; i < JOINT_NUMBER; i++)
            {
                this->ActionScale[i] = PostprocessCfg["action_scale"][i].get<InferencePrecision>();
            }

            //load default pos
            if (cfg["Workers"]["MotorControl"]["DefaultPosition"].size() != JOINT_NUMBER)
                throw(std::runtime_error("default_pos size is not equal!"));
            for (size_t i = 0; i < JOINT_NUMBER; i++)
            {
                this->JointDefaultPos[i] = cfg["Workers"]["MotorControl"]["DefaultPosition"][i].get<InferencePrecision>();
            }

            //load act clip and joint clip
            this->ClipAction = PostprocessCfg["clip_actions"].get<InferencePrecision>();
            nlohmann::json joint_clip_upper_cfg = PostprocessCfg["joint_clip_upper"];
            nlohmann::json joint_clip_lower_cfg = PostprocessCfg["joint_clip_lower"];
            if (joint_clip_upper_cfg.size() != JOINT_NUMBER || joint_clip_lower_cfg.size() != JOINT_NUMBER)
                throw(std::runtime_error("joint_clip size is not equal!"));
            for (size_t i = 0; i < JOINT_NUMBER; i++)
            {
                this->JointClipUpper[i] = joint_clip_upper_cfg[i].get<InferencePrecision>();
                this->JointClipLower[i] = joint_clip_lower_cfg[i].get<InferencePrecision>();
            }

            this->PrintSplitLine();
            std::cout << "CommonLocoInferenceWorker" << std::endl;
            std::cout << "JOINT_NUMBER=" << JOINT_NUMBER << std::endl;
            std::cout << "Joint Default Pos=" << this->JointDefaultPos << std::endl;
            std::cout << std::endl;
            std::cout << "ClipObservation=" << this->ClipObservation << std::endl;
            std::cout << "ClipAction=" << this->ClipAction << std::endl;
            std::cout << "ClipJointUpper=" << this->JointClipUpper << std::endl;
            std::cout << "ClipJointLower=" << this->JointClipLower << std::endl;
            std::cout << std::endl;
            std::cout << "Scales_lin_vel=" << this->Scales_lin_vel << std::endl;
            std::cout << "Scales_ang_vel=" << this->Scales_ang_vel << std::endl;
            std::cout << "Scales_project_gravity=" << this->Scales_project_gravity << std::endl;
            std::cout << "Scales_command3=" << this->Scales_command3 << std::endl;
            std::cout << "Scales_dof_pos=" << this->Scales_dof_pos << std::endl;
            std::cout << "Scales_dof_vel=" << this->Scales_dof_vel << std::endl;
            std::cout << "Scales_last_action=" << this->Scales_last_action << std::endl;
            std::cout << "Scales_action=" << this->ActionScale << std::endl;
            this->PrintSplitLine();
        }

        /**
         * @brief 析构函数
         *
         */
        virtual ~CommonLocoInferenceWorker()
        {
        }

    protected:
        /// @brief 电机向量类型
        using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;

        /// @brief 三维向量类型(用于IMU， cmd等)
        using ValVec3 = math::Vector<InferencePrecision, 3>;

        /// @brief 电机默认位置
        MotorValVec JointDefaultPos;

        /// @brief 电机裁剪上限
        MotorValVec JointClipUpper;

        /// @brief 电机裁剪下限
        MotorValVec JointClipLower;

        /// @brief 动作缩放
        MotorValVec ActionScale;

        /// @brief 观测量裁剪
        InferencePrecision ClipObservation;

        /// @brief 动作裁剪
        InferencePrecision ClipAction;

        /// @brief 观测量线速度缩放
        ValVec3 Scales_lin_vel;

        /// @brief 观测量角速度缩放
        ValVec3 Scales_ang_vel;

        /// @brief 观测量重力投影缩放
        ValVec3 Scales_project_gravity;

        /// @brief 观测量命令缩放 ValVec3 Scales_command3 = { scales_lin_vel,scales_lin_vel ,scales_ang_vel }
        ValVec3 Scales_command3;

        /// @brief 观测量关节位置缩放
        MotorValVec Scales_dof_pos;

        /// @brief 观测量关节速度缩放
        MotorValVec Scales_dof_vel;

        /// @brief 观测量上一次动作缩放
        MotorValVec Scales_last_action;
    };
};

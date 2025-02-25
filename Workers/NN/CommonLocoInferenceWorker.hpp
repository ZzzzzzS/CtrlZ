#pragma once
#include "AbstractInferenceWorker.hpp"
#include "onnxruntime_cxx_api.h"
#include "Utils/MathTypes.hpp"
#include "nlohmann/json.hpp"

namespace zzs
{
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

        virtual ~CommonLocoInferenceWorker()
        {
        }

    protected:
        using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
        using ValVec3 = math::Vector<InferencePrecision, 3>;

        MotorValVec JointDefaultPos;
        MotorValVec JointClipUpper;
        MotorValVec JointClipLower;
        MotorValVec ActionScale;

        InferencePrecision ClipObservation;
        InferencePrecision ClipAction;

        ValVec3 Scales_lin_vel;
        ValVec3 Scales_ang_vel;
        ValVec3 Scales_project_gravity;
        ValVec3 Scales_command3;

        MotorValVec Scales_dof_pos;
        MotorValVec Scales_dof_vel;
        MotorValVec Scales_last_action;
    };
};

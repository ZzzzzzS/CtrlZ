#pragma once
#include "CommonLocoInferenceWorker.hpp"
#include "NetInferenceWorker.h"
#include "chrono"

namespace zzs
{
	template<typename SchedulerType, typename InferencePrecision, size_t JOINT_NUMBER>
	class PlainInferenceWorker : public CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>
	{
	public:
		using Base = CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>;
		using Base::MotorValVec;
		using Base::ValVec3;

	public:
		PlainInferenceWorker(SchedulerType* scheduler, const nlohmann::json& cfg)
			:CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>(scheduler, cfg),
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

		virtual ~PlainInferenceWorker()
		{

		}

		void PreProcess() override
		{
			this->start_time = std::chrono::steady_clock::now();

			MotorValVec CurrentMotorVel;
			this->Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel);

			MotorValVec CurrentMotorPos;
			this->Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);

			MotorValVec LastAction;
			this->Scheduler->template GetData<"NetLastAction">(LastAction);

			ValVec3 UserCmd3;
			this->Scheduler->template GetData<"NetUserCommand3">(UserCmd3);

			ValVec3 LinVel;
			this->Scheduler->template GetData<"LinearVelocityValue">(LinVel);

			ValVec3 AngVel;
			this->Scheduler->template GetData<"AngleVelocityValue">(AngVel);

			ValVec3 Ang;
			this->Scheduler->template GetData<"AngleValue">(Ang);

			ValVec3 ProjectedGravity = ComputeProjectedGravity(Ang, this->GravityVector);
			this->Scheduler->template SetData<"NetProjectedGravity">(ProjectedGravity);


			auto InputVecScaled = math::cat(
				LinVel,
				AngVel,
				ProjectedGravity,
				UserCmd3,
				CurrentMotorPos,
				CurrentMotorVel,
				LastAction
			) * this->InputScaleVec;

			this->InputTensor.Array() = InputVecScaled;
		}

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
		//base lin vel; base ang vel; proj grav; cmd3; dof pos; dof vel; last action
		static constexpr size_t INPUT_TENSOR_LENGTH = 3 + 3 + 3 + 3 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER;
		//joint number
		static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;

		//input tensor
		zzs::math::Tensor<InferencePrecision, 1, INPUT_TENSOR_LENGTH> InputTensor;
		zzs::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH> InputScaleVec;


		//output tensor
		zzs::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;
		zzs::math::Vector<InferencePrecision, OUTPUT_TENSOR_LENGTH> OutputScaleVec;

		const ValVec3 GravityVector;

		//compute time
		std::chrono::steady_clock::time_point start_time;
		std::chrono::steady_clock::time_point end_time;
	};
};


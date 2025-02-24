#pragma once

#include "CommonLocoInferenceWorker.hpp"
#include "NetInferenceWorker.h"
#include "Utils/ZenBuffer.hpp"
#include <chrono>

namespace zzs
{
	template<typename SchedulerType, typename InferencePrecision, size_t INPUT_STUCK_LENGTH, size_t JOINT_NUMBER>
	class HumanoidGymInferenceWorker : public CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>
	{
	public:
		using Base = CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>;
		using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
		using ValVec3 = math::Vector<InferencePrecision, 3>;

	public:
		HumanoidGymInferenceWorker(SchedulerType* scheduler, const nlohmann::json& cfg)
			:CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>(scheduler, cfg),
			GravityVector({ 0.0,0.0,-1.0 }),
			HistoryInputBuffer(INPUT_STUCK_LENGTH)
		{
			//read cfg
			nlohmann::json InferenceCfg = cfg["Workers"]["NN"]["Inference"];
			nlohmann::json NetworkCfg = cfg["Workers"]["NN"]["Network"];
			this->cycle_time = NetworkCfg["Cycle_time"].get<InferencePrecision>();
			this->dt = cfg["Scheduler"]["dt"].get<InferencePrecision>();

			this->PrintSplitLine();
			std::cout << "EraxInferenceWorker" << std::endl;
			std::cout << "JOINT_NUMBER=" << JOINT_NUMBER << std::endl;
			std::cout << "Cycle_time=" << this->cycle_time << std::endl;
			std::cout << "dt=" << this->dt << std::endl;
			this->PrintSplitLine();

			//concatenate all scales
			auto clock_scales = math::Vector<InferencePrecision, 2>::ones();
			this->InputScaleVec = math::cat(
				clock_scales,
				this->Scales_command3,
				this->Scales_dof_pos,
				this->Scales_dof_vel,
				this->Scales_last_action,
				this->Scales_ang_vel,
				this->Scales_project_gravity
			);
			this->OutputScaleVec = this->ActionScale;

			//warp input tensor
			this->InputOrtTensors__.push_back(this->WarpOrtTensor(InputTensor));
			this->OutputOrtTensors__.push_back(this->WarpOrtTensor(OutputTensor));
		}

		virtual ~HumanoidGymInferenceWorker()
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
			this->Scheduler->template GetData<"NetLastAction">(LastAction);

			ValVec3 UserCmd3;
			this->Scheduler->template GetData<"NetUserCommand3">(UserCmd3);

			ValVec3 LinVel;
			this->Scheduler->template GetData<"LinearVelocityValue">(LinVel);

			ValVec3 AngVel;
			this->Scheduler->template GetData<"AngleVelocityValue">(AngVel);

			ValVec3 Ang;
			this->Scheduler->template GetData<"AngleValue">(Ang);


			size_t t = this->Scheduler->getTimeStamp();
			InferencePrecision phase = this->dt * static_cast<InferencePrecision>(t) / this->cycle_time;
			InferencePrecision clock_sin = std::sin(phase * 2 * M_PI);
			InferencePrecision clock_cos = std::cos(phase * 2 * M_PI);
			zzs::math::Vector<InferencePrecision, 2> ClockVector = { clock_sin, clock_cos };
			this->Scheduler->template SetData<"NetClockVector">(ClockVector);

			auto SingleInputVecScaled = math::cat(
				ClockVector,
				UserCmd3,
				CurrentMotorPos,
				CurrentMotorVel,
				LastAction,
				AngVel,
				Ang	
			) * this->InputScaleVec;

			this->HistoryInputBuffer.push(SingleInputVecScaled);


			math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH> InputVec;
			for (size_t i = 0; i < INPUT_STUCK_LENGTH; i++)
			{
				std::copy(this->HistoryInputBuffer[i].begin(), this->HistoryInputBuffer[i].end(), InputVec.begin() + i * INPUT_TENSOR_LENGTH_UNIT);
			}

			this->InputTensor.Array() = decltype(InputVec)::clamp(InputVec, -this->ClipObservation, this->ClipObservation);
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
		//clock; usercmd;q;dq;act;angle vel;euler xyz;
		static constexpr size_t INPUT_TENSOR_LENGTH_UNIT = 2 + 3 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER + 3 + 3;
		static constexpr size_t INPUT_TENSOR_LENGTH = INPUT_TENSOR_LENGTH_UNIT * INPUT_STUCK_LENGTH;
		//joint number
		static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;

		//input tensor
		zzs::math::Tensor<InferencePrecision, 1, INPUT_TENSOR_LENGTH> InputTensor;
		zzs::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH_UNIT> InputScaleVec;
		zzs::RingBuffer<zzs::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH_UNIT>> HistoryInputBuffer;

		//output tensor
		zzs::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;
		zzs::math::Vector<InferencePrecision, OUTPUT_TENSOR_LENGTH> OutputScaleVec;

		const ValVec3 GravityVector;

		InferencePrecision cycle_time;
		InferencePrecision dt;

		const InferencePrecision M_PI = 3.14159265358979323846;

		//compute time
		std::chrono::steady_clock::time_point start_time;
		std::chrono::steady_clock::time_point end_time;
	};
};


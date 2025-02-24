#pragma once
#include "AbstractWorker.hpp"
#include "Schedulers/AbstractScheduler.hpp"
#include "Utils/MathTypes.hpp"
#include "Utils/StaticStringUtils.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

namespace zzs
{
	template<typename SchedulerType, typename MotorPrecision, size_t JointNumber>
	class MotorResetPositionWorker : public AbstractWorker<SchedulerType>
	{
		using MotorValVec = math::Vector<MotorPrecision, JointNumber>;
	public:
		MotorResetPositionWorker(SchedulerType* scheduler, const nlohmann::json& cfg)
			:AbstractWorker<SchedulerType>(scheduler),
			enabled(false)
		{
			nlohmann::json Motor_cfg = cfg["Workers"]["MotorControl"];
			if (Motor_cfg["DefaultPosition"].size() != JointNumber)
				throw(std::runtime_error("Default Position size is not equal!"));

			for (size_t i = 0; i < JointNumber; i++)
			{
				this->DefaultPosition[i] = Motor_cfg["DefaultPosition"][i].get<MotorPrecision>();
			}

			MotorPrecision dt = cfg["Scheduler"]["dt"];
			MotorPrecision Duration = cfg["Workers"]["ResetPosition"]["Duration"];
			this->DefaultResetEpoches = static_cast<size_t>(Duration / dt);

			this->PrintSplitLine();
			std::cout << "MotorResetPositionWorker" << std::endl;
			std::cout << "DefaultPosition=" << this->DefaultPosition << std::endl;
			std::cout << "reset duration=" << Duration << std::endl;
			std::cout << "DefaultResetEpoches=" << this->DefaultResetEpoches << std::endl;

			this->PrintSplitLine();
		}

		void StartReset(size_t epoches=0)
		{
			this->ResetEpoches = (epoches != 0)? epoches:this->DefaultResetEpoches;
			this->ResetCnt = this->ResetEpoches + this->Scheduler->getTimeStamp();

			this->Scheduler->template GetData<"CurrentMotorPosition">(this->TargetPosition);
			MotorValVec err = this->DefaultPosition - this->TargetPosition;
			this->PositionStep = err / static_cast<MotorPrecision>(this->ResetEpoches);
			this->enabled = true;
		}

		void StopReset()
		{
			this->enabled = false;
		}

		void TaskRun() override
		{
			if (this->enabled)
			{
				if (this->Scheduler->getTimeStamp() < this->ResetCnt)
				{
					this->TargetPosition = this->DefaultPosition - this->PositionStep * (this->ResetCnt - this->Scheduler->getTimeStamp());
					this->Scheduler->template SetData<"TargetMotorPosition">(this->TargetPosition);
				}
				else
				{
					this->Scheduler->template SetData<"TargetMotorPosition">(this->DefaultPosition);
					this->enabled = false;
				}
			}

		}

	private:
		size_t DefaultResetEpoches = 0;
		size_t ResetEpoches = 0;
		size_t ResetCnt = 0;

		std::atomic<bool> enabled;

		MotorValVec DefaultPosition;
		MotorValVec PositionStep;
		MotorValVec TargetPosition;
	};
};

#pragma once
#include "AbstractWorker.hpp"
#include "../Schedulers/AbstractScheduler.hpp"
#include "../Utils/StaticStringUtils.hpp"
#include "../Utils/ZenBuffer.hpp"
#include "../Utils/MathTypes.hpp"

#include <iostream>
#include <memory>

namespace zzs
{
	template<typename SchedulerType,typename ImuType, typename ImuPrecision>
	class ImuProcessWorker : public AbstractWorker<SchedulerType>
	{
		static_assert(std::is_arithmetic<ImuPrecision>::value, "ImuPrecision must be a arithmetic type");
		using ImuValVec = math::Vector<ImuPrecision, 3>;
	public:
		ImuProcessWorker(SchedulerType* scheduler, ImuType ImuInstance, const nlohmann::json& root_cfg= nlohmann::json())
			:AbstractWorker<SchedulerType>(scheduler),
			ImuInstance(ImuInstance)
		{
			nlohmann::json cfg = root_cfg["Workers"]["ImuProcess"];
			this->PrintSplitLine();
			std::cout << "ImuProcessWorker" << std::endl;
			
			try
			{
				std::vector<ImuValVec> AccFilterWeightVec;
				std::vector<ImuValVec> GyroFilterWeightVec;
				std::vector<ImuValVec> MagFilterWeightVec;
				nlohmann::json AccFilterWeight_cfg = cfg["AccFilterWeight"];
				nlohmann::json GyroFilterWeight_cfg = cfg["GyroFilterWeight"];
				nlohmann::json MagFilterWeight_cfg = cfg["MagFilterWeight"];

				for (auto&& val : AccFilterWeight_cfg)
				{
					ImuPrecision weight = val.get<ImuPrecision>();
					AccFilterWeightVec.emplace_back(ImuValVec::ones() * weight);
				}

				for (auto&& val : GyroFilterWeight_cfg)
				{
					ImuPrecision weight = val.get<ImuPrecision>();
					GyroFilterWeightVec.emplace_back(ImuValVec::ones() * weight);
				}

				for (auto&& val : MagFilterWeight_cfg)
				{
					ImuPrecision weight = val.get<ImuPrecision>();
					MagFilterWeightVec.emplace_back(ImuValVec::ones() * weight);
				}

				this->AccFilter = std::make_unique<WeightFilter<ImuValVec>>(AccFilterWeightVec);
				this->GyroFilter = std::make_unique<WeightFilter<ImuValVec>>(GyroFilterWeightVec);
				this->MagFilter = std::make_unique<WeightFilter<ImuValVec>>(MagFilterWeightVec);
			}
			catch (const std::exception& e)
			{
				std::cerr << e.what() << std::endl;
				std::cerr << "Failed to get filter weight from config, use default value." << std::endl;
				this->AccFilter = std::make_unique<WeightFilter<ImuValVec>>(std::vector<ImuValVec>(1, ImuValVec::ones()));
				this->GyroFilter = std::make_unique<WeightFilter<ImuValVec>>(std::vector<ImuValVec>(1, ImuValVec::ones()));
				this->MagFilter = std::make_unique<WeightFilter<ImuValVec>>(std::vector<ImuValVec>(1, ImuValVec::ones()));
			}

			this->PrintSplitLine();
		}

		~ImuProcessWorker() {}

		void TaskCycleBegin() override
		{
			ImuValVec Acc = {
			static_cast<ImuPrecision>(this->ImuInstance->GetAccX()),
			static_cast<ImuPrecision>(this->ImuInstance->GetAccY()),
			static_cast<ImuPrecision>(this->ImuInstance->GetAccZ()) 
			}; //获取Acc数据
			ImuValVec LastAcc;
			this->Scheduler->template GetData<"AccelerationRaw">(LastAcc);
			Acc = RemoveNan(Acc, LastAcc); //去除nan值，用上一次的值代替
			this->Scheduler->template SetData<"AccelerationRaw">(Acc);


			ImuValVec Gyro = {
				static_cast<ImuPrecision>(this->ImuInstance->GetGyroX()),
				static_cast<ImuPrecision>(this->ImuInstance->GetGyroY()),
				static_cast<ImuPrecision>(this->ImuInstance->GetGyroZ())
			}; //获取Gyro数据
			ImuValVec LastGyro;
			this->Scheduler->template GetData<"AngleVelocityRaw">(LastGyro);
			Gyro = RemoveNan(Gyro, LastGyro);
			this->Scheduler->template SetData<"AngleVelocityRaw">(Gyro);


			ImuValVec Mag = {
				static_cast<ImuPrecision>(this->ImuInstance->GetRoll()),
				static_cast<ImuPrecision>(this->ImuInstance->GetPitch()),
				static_cast<ImuPrecision>(this->ImuInstance->GetYaw())
			}; //获取Mag数据
			ImuValVec LastMag;
			this->Scheduler->template GetData<"AngleRaw">(LastMag);
			Mag = RemoveNan(Mag, LastMag);
			this->Scheduler->template SetData<"AngleRaw">(Mag);


			this->Scheduler->template SetData<"AccelerationValue">((*AccFilter)(Acc)); //滤波
			this->Scheduler->template SetData<"AngleVelocityValue">((*GyroFilter)(Gyro));
			this->Scheduler->template SetData<"AngleValue">((*MagFilter)(Mag));
		}

		void TaskRun() override
		{
		}

	private:

		ImuValVec RemoveNan(ImuValVec& vec, const ImuValVec& last_value)
		{
			vec.apply([&last_value](ImuPrecision& val, size_t idx) {
				val = std::isnan(val) ? last_value[idx] : val;
				});
			return vec;
		}

	private:
		ImuType ImuInstance;

		
		std::unique_ptr<WeightFilter<ImuValVec>> AccFilter;
		std::unique_ptr<WeightFilter<ImuValVec>> GyroFilter;
		std::unique_ptr<WeightFilter<ImuValVec>> MagFilter;
	};
};


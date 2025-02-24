#pragma once
#include "Workers/AbstractWorker.hpp"
#include <thread>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include <array>
#include <vector>
#include "Utils/MathTypes.hpp"
#include "Schedulers/AbstractScheduler.hpp"
#include "onnxruntime_cxx_api.h"
#include "NetInferenceWorker.h"

namespace zzs
{
	template<typename SchedulerType, typename InferencePrecision>
	class AbstractNetInferenceWorker : public AbstractWorker<SchedulerType>
	{
		static_assert(std::is_arithmetic<InferencePrecision>::value, "InferencePrecision must be a arithmetic type");
	public:
		AbstractNetInferenceWorker(SchedulerType* scheduler, const nlohmann::json& cfg)
			:AbstractWorker<SchedulerType>(scheduler, cfg),
			Session__(nullptr),
			MemoryInfo__(nullptr),
			IoBinding__(nullptr)
		{
			//读取配置文件
			nlohmann::json InferenceCfg = cfg["Workers"]["NN"]["Inference"];
			nlohmann::json NetworkCfg = cfg["Workers"]["NN"]["Network"];

			//读取推理配置
			this->WarmUpModel__ = InferenceCfg["WarmUpModel"].get<bool>();
			if (this->WarmUpModel__)
			{
				this->WarmUpCnt__ = InferenceCfg["WarmUpCount"].get<size_t>();
			}
			this->IntraNumberThreads__ = InferenceCfg["IntraNumberThreads"].get<size_t>();

			//读取网络配置
			this->ModelPath__ = NetworkCfg["ModelPath"].get<std::string>();
			this->InputNodeNames__ = NetworkCfg["InputNodeNames"].get<std::vector<std::string>>();
			this->OutputNodeNames__ = NetworkCfg["OutputNodeNames"].get<std::vector<std::string>>();

			//初始化模型
			this->SessionOptions__.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			this->SessionOptions__.SetIntraOpNumThreads(this->IntraNumberThreads__);
			this->SessionOptions__.SetInterOpNumThreads(1);
			this->SessionOptions__.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

#ifdef _WIN32
			std::wstring wstr = string_to_wstring(this->ModelPath__);
#else
			std::string wstr = this->ModelPath__;
#endif
			this->Session__ = Ort::Session(GetOrtEnv(), wstr.c_str(), this->SessionOptions__);
			this->MemoryInfo__ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

			this->PrintSplitLine();
			std::cout << "AbstractNetInferenceWorker" << std::endl;
			std::cout << "IntraNumberThreads=" << this->IntraNumberThreads__ << std::endl;
			std::cout << "WarmUpModel=" << this->WarmUpModel__ << std::endl;
			std::cout << "WarmUpCnt=" << this->WarmUpCnt__ << std::endl;
			std::cout << std::endl;
			std::cout << "ModelPath=" << this->ModelPath__ << std::endl;

			std::cout << "InputNodeNames=[";
			for (auto&& name : this->InputNodeNames__)
				std::cout << name << " ";
			std::cout << "]\n";

			std::cout << "OutputNodeNames=[";
			for (auto&& name : this->OutputNodeNames__)
				std::cout << name << " ";
			std::cout << "]\n";

			this->PrintSplitLine();
		}

		virtual ~AbstractNetInferenceWorker()
		{
		}

		void virtual PreProcess() = 0;
		void virtual PostProcess()  = 0;

		void virtual InferenceOnce()
		{
			this->Session__.Run(Ort::RunOptions{ nullptr }, this->IoBinding__);
		}

		void TaskRun() override
		{
			PreProcess();
			InferenceOnce();
			PostProcess();
		}

		void TaskCreate() override
		{
			//bind input and output tensor
			if(this->InputOrtTensors__.empty()||this->OutputOrtTensors__.empty())
				throw(std::runtime_error("InputOrtTensors or OutputOrtTensors is empty, call WarpOrtTensor to create Ort tensors before launch scheduler!"));

			if(this->InputNodeNames__.size()!=this->InputOrtTensors__.size())
				throw(std::runtime_error("InputNodeNames size is not equal to InputOrtTensors size!"));
			if (this->OutputNodeNames__.size() != this->OutputOrtTensors__.size())
				throw(std::runtime_error("OutputNodeNames size is not equal to OutputOrtTensors size!"));

			this->IoBinding__ = Ort::IoBinding(this->Session__);
			
			for (size_t i = 0; i < this->InputNodeNames__.size(); i++)
			{
				this->IoBinding__.BindInput(this->InputNodeNames__[i].c_str(), this->InputOrtTensors__[i]);
			}

			for (size_t i = 0; i < this->OutputNodeNames__.size(); i++)
			{
				this->IoBinding__.BindOutput(this->OutputNodeNames__[i].c_str(), this->OutputOrtTensors__[i]);
			}

			//warm up model
			if (this->WarmUpModel__)
			{
				for (size_t i = 0; i < this->WarmUpCnt__; i++)
				{
					InferenceOnce();
				}
			}
		}

		template<int64_t ...Dims>
		void WarpOrtTensor(math::Tensor<InferencePrecision, Dims...>& Tensor, Ort::Value& OrtTensor)
		{
			OrtTensor = Ort::Value::CreateTensor<InferencePrecision>(this->MemoryInfo__, Tensor.data__(), Tensor.size(), Tensor.shape_ptr(), Tensor.num_dims());
		}

		template<int64_t ...Dims>
		Ort::Value WarpOrtTensor(math::Tensor<InferencePrecision, Dims...>& Tensor)
		{
			return Ort::Value::CreateTensor<InferencePrecision>(this->MemoryInfo__, Tensor.data(), Tensor.size(), Tensor.shape_ptr(), Tensor.num_dims());
		}
		

	protected:
		bool WarmUpModel__ = false;
		size_t WarmUpCnt__ = 0;
		std::string ModelPath__;
		size_t IntraNumberThreads__ = 1;


		Ort::Session Session__;
		Ort::SessionOptions SessionOptions__;
		Ort::AllocatorWithDefaultOptions DefaultAllocator__;
		Ort::MemoryInfo MemoryInfo__;
		
		std::vector<std::string> InputNodeNames__;
		std::vector<std::string> OutputNodeNames__;
		std::vector<Ort::Value> InputOrtTensors__;
		std::vector<Ort::Value> OutputOrtTensors__;

		Ort::IoBinding IoBinding__;
	};
};
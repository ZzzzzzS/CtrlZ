/**
 * @file AbstractInferenceWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
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

namespace z
{
	/**
	 * @brief AbstractNetInferenceWorker类型是一切神经网络推理工人类型的基类，该类提供一些基本推理的功能，
	 * 用户可以通过继承这个类来实现自己的推理工人类型。
	 * @details AbstractNetInferenceWorker类型是一切神经网络推理工人类型的基类，该类提供一些基本推理的功能，
	 * 用户可以通过继承这个类来实现自己的推理工人类型。该类型实现了与ONNXRuntime的交互，用户可以通过配置文件来配置模型的路径，
	 * 输入节点名称，输出节点名称，线程数等参数。用户可以通过继承这个类来实现自己的推理工人类型，用户必须实现PreProcess，PostProcess，方法
	 * 分别用来实现推理前的准备工作，推理后的处理工作，将在推理前后(InferenceOnce方法前后)依次调用这两个方法。
	 *
	 * details config.json配置文件示例：
	 * {
	 * 		"Workers": {
	 * 			"NN": {
	 * 				"Inference": {
	 * 					"WarmUpModel": true, //是否需要预热模型
	 * 					"WarmUpCount": 10, //预热次数
	 * 					"IntraNumberThreads": 1 //线程数
	 * 				},
	 * 				"Network": {
	 * 					"ModelPath": "model.onnx", //模型路径
	 * 					"InputNodeNames": ["input"], //输入节点名称
	 * 					"OutputNodeNames": ["output"] //输出节点名称
	 * 				}
	 * 			}
	 * 		}
	 * }
	 *
	 * @tparam SchedulerType 调度器类型
	 * @tparam InferencePrecision 推理精度，用户可以通过这个参数来指定推理的精度，比如可以指定为float或者double
	 */
	template<typename SchedulerType, typename InferencePrecision>
	class AbstractNetInferenceWorker : public AbstractWorker<SchedulerType>
	{
		/// @brief 推理精度必须是一个算术类型
		static_assert(std::is_arithmetic<InferencePrecision>::value, "InferencePrecision must be a arithmetic type");
	public:
		/**
		 * @brief 构造一个AbstractNetInferenceWorker类型
		 *
		 * @param scheduler 调度器的指针
		 * @param cfg 配置文件
		 */
		AbstractNetInferenceWorker(SchedulerType* scheduler, const nlohmann::json& cfg)
			:AbstractWorker<SchedulerType>(scheduler, cfg),
			Session__(nullptr),
			SessionOptions__(),
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
			//this->SessionOptions__ = Ort::SessionOptions();
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

		/**
		 * @brief 析构函数
		 *
		 */
		virtual ~AbstractNetInferenceWorker()
		{
		}

		/**
		 * @brief 预处理方法，用户必须实现这个方法，用来实现推理前的准备工作,该方法在每次InferenceOnce()之前被调用
		 *
		 */
		void virtual PreProcess() = 0;

		/**
		 * @brief 后处理方法，用户必须实现这个方法，用来实现推理后的处理工作,该方法在每次InferenceOnce()之后被调用
		 *
		 */
		void virtual PostProcess() = 0;

		/**
		 * @brief 推理方法，用户可选重定义这个方法，用来实现推理的逻辑，默认实现是使用onnxruntime的推理方法
		 *
		 */
		void virtual InferenceOnce()
		{
			this->Session__.Run(Ort::RunOptions{ nullptr }, this->IoBinding__);
		}

		/**
		 * @brief 在每次任务队列循环中被调用，用来实现推理的逻辑,默认实现是依次调用PreProcess，InferenceOnce，PostProcess方法
		 *
		 */
		void TaskRun() override
		{
			PreProcess();
			InferenceOnce();
			PostProcess();
		}

		/**
		 * @brief 任务创建的方法，该方法中会初始化模型，预热模型，绑定输入输出节点等
		 *
		 */
		void TaskCreate() override
		{
			//bind input and output tensor
			if (this->InputOrtTensors__.empty() || this->OutputOrtTensors__.empty())
				throw(std::runtime_error("InputOrtTensors or OutputOrtTensors is empty, call WarpOrtTensor to create Ort tensors before launch scheduler!"));

			if (this->InputNodeNames__.size() != this->InputOrtTensors__.size())
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

		/**
		 * @brief 将z::math::Tensor类型的数据转换为Ort::Value类型的数据。
		 *
		 * @details 将z::math::Tensor类型的数据转换为Ort::Value类型的数据。
		 * **注意Ort::Value类型的数据是一个指针，因此必须保证Tensor的生命周期大于Ort::Value的生命周期，
		 * 即Tensor不能提前销毁！**
		 *
		 * @tparam Dims Tensor的维度
		 * @param Tensor 输入的Tensor
		 * @return Ort::Value 返回的Ort::Value类型的数据
		 */
		template<int64_t ...Dims>
		Ort::Value WarpOrtTensor(math::Tensor<InferencePrecision, Dims...>& Tensor)
		{
			return Ort::Value::CreateTensor<InferencePrecision>(this->MemoryInfo__, Tensor.data(), Tensor.size(), Tensor.shape_ptr(), Tensor.num_dims());
		}


	protected:
		/// @brief 是否需要预热模型
		bool WarmUpModel__ = false;

		/// @brief 预热次数
		size_t WarmUpCnt__ = 0;

		/// @brief 模型路径
		std::string ModelPath__;

		/// @brief 推理线程数，默认为1，增大线程数可能可以提升推理速度但是会增加资源消耗
		size_t IntraNumberThreads__ = 1;

		/// @brief ONNXRuntime的Session对象，用来加载模型和进行推理
		Ort::Session Session__;

		/// @brief ONNXRuntime的SessionOptions对象，用来配置Session
		Ort::SessionOptions SessionOptions__;

		/// @brief ONNXRuntime的AllocatorWithDefaultOptions对象，用来分配内存
		Ort::AllocatorWithDefaultOptions DefaultAllocator__;

		/// @brief ONNXRuntime的MemoryInfo对象，用来配置内存信息
		Ort::MemoryInfo MemoryInfo__;


		/// @brief 输入节点名称列表
		std::vector<std::string> InputNodeNames__;

		/// @brief 输出节点名称列表
		std::vector<std::string> OutputNodeNames__;

		/// @brief 输入Ort::Value列表，顺序与InputNodeNames一致
		std::vector<Ort::Value> InputOrtTensors__;

		/// @brief 输出Ort::Value列表，顺序与OutputNodeNames一致
		std::vector<Ort::Value> OutputOrtTensors__;

		/// @brief ONNXRuntime的IoBinding对象，用来绑定输入输出节点，绑定名称和数据
		Ort::IoBinding IoBinding__;
	};
};
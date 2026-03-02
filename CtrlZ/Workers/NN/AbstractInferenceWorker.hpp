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
#include <unordered_map>
#include <filesystem>
#include "Utils/MathTypes.hpp"
#include "Utils/StaticStringUtils.hpp"
#include "Schedulers/AbstractScheduler.hpp"

 // 根据宏定义选择推理后端
#ifdef USE_OPENVINO
#include <openvino/openvino.hpp>
#ifdef OPENVINO_ENABLE_GPU
#include <openvino/runtime/intel_gpu/properties.hpp>
#endif
#else
#include "onnxruntime_cxx_api.h"
#include "NetInferenceWorker.h"
#endif

namespace z
{
	/**
	 * @brief AbstractNetInferenceWorker类型是一切神经网络推理工人类型的基类，该类提供一些基本推理的功能，
	 * 用户可以通过继承这个类来实现自己的推理工人类型。
	 * @details AbstractNetInferenceWorker类型是一切神经网络推理工人类型的基类，该类提供一些基本推理的功能，
	 * 用户可以通过继承这个类来实现自己的推理工人类型。该类型实现了与ONNXRuntime/OpenVINO的交互，用户可以通过配置文件来配置模型的路径，
	 * 输入节点名称，输出节点名称，线程数等参数。用户可以通过继承这个类来实现自己的推理工人类型，用户必须实现PreProcess，PostProcess，方法
	 * 分别用来实现推理前的准备工作，推理后的处理工作，将在推理前后(InferenceOnce方法前后)依次调用这两个方法。
	 *
	 * @details config.json配置文件示例：
	 * @code {.json}
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
	 * 					"Device": "CPU", //推理设备（OpenVINO模式下有效，可选值：CPU, GPU, NPU, AUTO等）
	 * 					"InputNodeNames": ["input"], //输入节点名称
	 * 					"OutputNodeNames": ["output"] //输出节点名称
	 * 				}
	 * 			}
	 * 		}
	 * }
	 * @endcode
	 *
	 * @tparam SchedulerType 调度器类型
	 * @tparam NetName 网络名称，用户可以通过这个参数来指定网络的名称, 这在有多个网络时可以区分数据总线上的不同网络数据
	 * @tparam InferencePrecision 推理精度，用户可以通过这个参数来指定推理的精度，比如可以指定为float或者double
	 */
	template<typename SchedulerType, CTString NetName, typename InferencePrecision>
	class AbstractNetInferenceWorker : public AbstractWorker<SchedulerType>
	{
		/// @brief 推理精度必须是一个算术类型
		static_assert(std::is_arithmetic<InferencePrecision>::value, "InferencePrecision must be a arithmetic type");
	public:
#ifdef USE_OPENVINO
		/// @brief 张量类型别名 (OpenVINO)
		using TensorType = ov::Tensor;
#else
		/// @brief 张量类型别名 (ONNXRuntime)
		using TensorType = Ort::Value;
#endif

		/**
		 * @brief 构造一个AbstractNetInferenceWorker类型
		 *
		 * @param scheduler 调度器的指针
		 * @param cfg 配置文件
		 */
		AbstractNetInferenceWorker(SchedulerType::Ptr scheduler, const nlohmann::json& cfg)
			:AbstractWorker<SchedulerType>(scheduler, cfg)
		{
			//读取配置文件
			nlohmann::json InferenceCfg = cfg["Inference"];
			nlohmann::json NetworkCfg = cfg["Network"];

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

			// 读取OpenVINO设备配置（可选，默认为CPU）
			this->DeviceName__ = "CPU"; // 默认值
			if (InferenceCfg.contains("Device")) {
				this->DeviceName__ = InferenceCfg["Device"].get<std::string>();
			}

			// 将相对路径转换为绝对路径
			std::filesystem::path model_path = this->ModelPath__;
			if (!model_path.is_absolute()) {
				model_path = std::filesystem::absolute(model_path);
			}
			std::string absolute_model_path = model_path.string();

#ifdef USE_OPENVINO
			// OpenVINO 初始化
			this->Core__ = std::make_unique<ov::Core>();

			// 列出所有可用设备
			auto available_devices = this->Core__->get_available_devices();
			std::cout << "OpenVINO available devices: [";
			for (size_t i = 0; i < available_devices.size(); i++) {
				std::cout << available_devices[i];
				if (i < available_devices.size() - 1) std::cout << ", ";
			}
			std::cout << "]" << std::endl;

			// 检查请求的设备是否可用
			bool device_found = false;
			for (const auto& device : available_devices) {
				if (device == this->DeviceName__) {
					device_found = true;
					break;
				}
			}
			if (!device_found) {
				std::cerr << "Warning: Requested device '" << this->DeviceName__ 
					  << "' not found in available devices!" << std::endl;
				std::cerr << "Falling back to AUTO device selection." << std::endl;
				this->DeviceName__ = "AUTO";
			}

			// 设置线程数（仅对CPU设备有效）
			if (this->DeviceName__ == "CPU") {
				this->Core__->set_property("CPU", ov::inference_num_threads(this->IntraNumberThreads__));
			}

			// GPU 设备特定配置
			if (this->DeviceName__.find("GPU") != std::string::npos) {
				try {
					// 设置 GPU 任务优先级（如果支持）
					#ifdef OPENVINO_ENABLE_GPU
					this->Core__->set_property("GPU", ov::intel_gpu::hint::host_task_priority(ov::hint::Priority::HIGH));
					std::cout << "GPU device configuration applied with HIGH priority." << std::endl;
					#else
					std::cout << "GPU device configuration (no additional properties set)." << std::endl;
					#endif
				} catch (const std::exception& e) {
					std::cerr << "Warning: Failed to set GPU properties: " << e.what() << std::endl;
				}
			}

			// 加载模型（使用配置的设备）
			ov::CompiledModel compiled_model;
			try {
				compiled_model = this->Core__->compile_model(absolute_model_path, this->DeviceName__);
			} catch (const ov::Exception& e) {
				std::cerr << "Error: Failed to compile model for device '" << this->DeviceName__ << "'" << std::endl;
				std::cerr << "Exception: " << e.what() << std::endl;
				// 如果指定设备失败，尝试使用 AUTO
				if (this->DeviceName__ != "AUTO") {
					std::cerr << "Trying AUTO device selection..." << std::endl;
					this->DeviceName__ = "AUTO";
					compiled_model = this->Core__->compile_model(absolute_model_path, "AUTO");
				} else {
					throw;
				}
			}
			this->CompiledModel__ = std::make_unique<ov::CompiledModel>(std::move(compiled_model));

			// 创建推理请求
			ov::InferRequest infer_request = this->CompiledModel__->create_infer_request();
			this->InferRequest__ = std::make_unique<ov::InferRequest>(std::move(infer_request));

			// 构建输入节点名称到索引的映射
			auto inputs = this->CompiledModel__->inputs();
			for (size_t i = 0; i < inputs.size(); i++) {
				std::string name = inputs[i].get_any_name();
				this->InputNameToIndex__[name] = i;
			}

			// 构建输出节点名称到索引的映射
			auto outputs = this->CompiledModel__->outputs();
			for (size_t i = 0; i < outputs.size(); i++) {
				std::string name = outputs[i].get_any_name();
				this->OutputNameToIndex__[name] = i;
			}

			// 验证输入输出节点
			if (this->InputNodeNames__.size() != inputs.size()) {
				std::cerr << "Warning: InputNodeNames size (" << this->InputNodeNames__.size()
					<< ") != model inputs size (" << inputs.size() << ")" << std::endl;
			}
			if (this->OutputNodeNames__.size() != outputs.size()) {
				std::cerr << "Warning: OutputNodeNames size (" << this->OutputNodeNames__.size()
					<< ") != model outputs size (" << outputs.size() << ")" << std::endl;
			}
#else
			// ONNXRuntime 初始化
			this->SessionOptions__ = Ort::SessionOptions();
			this->SessionOptions__.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			this->SessionOptions__.SetIntraOpNumThreads(this->IntraNumberThreads__);
			this->SessionOptions__.SetInterOpNumThreads(1);
			this->SessionOptions__.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

#ifdef _WIN32
			std::wstring wstr = string_to_wstring(absolute_model_path);
#else
			std::string wstr = absolute_model_path;
#endif
			this->Session__ = Ort::Session(GetOrtEnv(), wstr.c_str(), this->SessionOptions__);
			this->MemoryInfo__ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif

			this->PrintSplitLine();
			std::cout << "AbstractNetInferenceWorker" << std::endl;
#ifdef USE_OPENVINO
			std::cout << "Backend: OpenVINO" << std::endl;
			std::cout << "Device=" << this->DeviceName__ << std::endl;
#else
			std::cout << "Backend: ONNXRuntime" << std::endl;
#endif
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
		 * @brief 推理方法，用户可选重定义这个方法，用来实现推理的逻辑，默认实现是使用onnxruntime/openvino的推理方法
		 *
		 */
		void virtual InferenceOnce()
		{
#ifdef USE_OPENVINO
			// 执行推理
			this->InferRequest__->infer();

			// 将输出数据从 InferRequest 复制到 OutputOrtTensors__
			// 通过名称查找索引获取对应的输出张量
			for (size_t i = 0; i < this->OutputNodeNames__.size(); i++) {
				const std::string& node_name = this->OutputNodeNames__[i];
				auto it = this->OutputNameToIndex__.find(node_name);
				if (it != this->OutputNameToIndex__.end()) {
					ov::Tensor src_tensor = this->InferRequest__->get_output_tensor(it->second);
					void* dst_data = this->OutputOrtTensors__[i].data();
					if (dst_data != nullptr && src_tensor.data() != nullptr) {
						std::memcpy(dst_data, src_tensor.data(), src_tensor.get_byte_size());
					}
				}
				else {
					std::cerr << "Error: Output node '" << node_name << "' not found in model!" << std::endl;
				}
			}
#else
			this->Session__.Run(Ort::RunOptions{ nullptr }, this->IoBinding__);
#endif
		}

		/**
		 * @brief 设置是否启动推理。或是否在未来的某个时间点启动推理。
		 *
		 * @param enable 是否启用推理
		 * @param future_time 在未来的某个时间点启用推理，单位为秒，默认值为0，表示立即启用
		 */
		void SetEnable(bool enable, InferencePrecision future_time = 0)
		{
			if (this->NextEnable__.load() == enable) [[unlikely]]
				return;

			this->NextEnable__.store(enable);
			this->NextActionCycleCnt__.store(this->Scheduler->getTimeStamp() + static_cast<size_t>(future_time / this->Scheduler->getSpinOnceTime()));
		}

		/**
		 * @brief 在每次任务队列循环中被调用，用来实现推理的逻辑,默认实现是依次调用PreProcess，InferenceOnce，PostProcess方法
		 *
		 */
		void TaskRun() override
		{
			if (this->NextActionCycleCnt__.load() < this->Scheduler->getTimeStamp()) [[likely]]
			{
				this->Enable__ = this->NextEnable__.load();
			}

			if (this->Enable__)
			{
				PreProcess();
				InferenceOnce();
				PostProcess();
			}

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

#ifdef USE_OPENVINO
			// OpenVINO 绑定输入张量（通过名称查找索引）
			for (size_t i = 0; i < this->InputNodeNames__.size(); i++) {
				const std::string& node_name = this->InputNodeNames__[i];
				auto it = this->InputNameToIndex__.find(node_name);
				if (it != this->InputNameToIndex__.end()) {
					try {
						this->InferRequest__->set_input_tensor(it->second, this->InputOrtTensors__[i]);
					}
					catch (const std::exception& e) {
						std::cerr << "Error: Failed to set input tensor for node '" << node_name
							<< "' at index " << it->second << ": " << e.what() << std::endl;
					}
				}
				else {
					std::cerr << "Error: Input node '" << node_name << "' not found in model!" << std::endl;
					std::cerr << "Available input nodes: ";
					for (const auto& pair : this->InputNameToIndex__) {
						std::cerr << "[" << pair.first << "=" << pair.second << "] ";
					}
					std::cerr << std::endl;
				}
			}
#else
			// ONNXRuntime 绑定输入输出张量
			this->IoBinding__ = Ort::IoBinding(this->Session__);

			for (size_t i = 0; i < this->InputNodeNames__.size(); i++)
			{
				this->IoBinding__.BindInput(this->InputNodeNames__[i].c_str(), this->InputOrtTensors__[i]);
			}

			for (size_t i = 0; i < this->OutputNodeNames__.size(); i++)
			{
				this->IoBinding__.BindOutput(this->OutputNodeNames__[i].c_str(), this->OutputOrtTensors__[i]);
			}
#endif

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
		 * @brief 获取输入张量的引用（OpenVINO 内部使用）
		 */
		std::vector<TensorType>& GetInputTensors()
		{
			return this->InputOrtTensors__;
		}

		/**
		 * @brief 获取输出张量的引用（OpenVINO 内部使用）
		 */
		std::vector<TensorType>& GetOutputTensors()
		{
			return this->OutputOrtTensors__;
		}

		/**
		 * @brief 将z::math::Tensor类型的数据转换为推理张量类型的数据。
		 *
		 * @details 将z::math::Tensor类型的数据转换为推理张量类型的数据。
		 * **注意张量类型的数据是一个指针，因此必须保证Tensor的生命周期大于张量的生命周期，
		 * 即Tensor不能提前销毁！**
		 *
		 * @tparam Dims Tensor的维度
		 * @param Tensor 输入的Tensor
		 * @return TensorType 返回的张量类型的数据
		 */
		template<int64_t ...Dims>
		TensorType WarpOrtTensor(math::Tensor<InferencePrecision, Dims...>& Tensor)
		{
#ifdef USE_OPENVINO
			// OpenVINO 张量创建
			std::vector<size_t> shape;
			(this->addShape(shape, Dims), ...);
			return ov::Tensor(this->getOvElementType<InferencePrecision>(), shape, Tensor.data());
#else
			// ONNXRuntime 张量创建
			return Ort::Value::CreateTensor<InferencePrecision>(this->MemoryInfo__, Tensor.data(), Tensor.size(), Tensor.shape_ptr(), Tensor.num_dims());
#endif
		}


	protected:
#ifdef USE_OPENVINO
		/**
		 * @brief 获取 OpenVINO 元素类型
		 */
		template<typename T>
		ov::element::Type getOvElementType() {
			if constexpr (std::is_same_v<T, float>) {
				return ov::element::f32;
			}
			else if constexpr (std::is_same_v<T, double>) {
				return ov::element::f64;
			}
			else if constexpr (std::is_same_v<T, int32_t>) {
				return ov::element::i32;
			}
			else if constexpr (std::is_same_v<T, int64_t>) {
				return ov::element::i64;
			}
			else {
				static_assert(std::is_same_v<T, float>, "Unsupported precision type for OpenVINO");
				return ov::element::f32;
			}
		}

		/**
		 * @brief 添加形状维度
		 */
		void addShape(std::vector<size_t>& shape, int64_t dim) {
			shape.push_back(static_cast<size_t>(dim));
		}
#endif

		/// @brief 是否需要预热模型
		bool WarmUpModel__ = false;

		/// @brief 预热次数
		size_t WarmUpCnt__ = 0;

		/// @brief 模型路径
		std::string ModelPath__;

		/// @brief 推理线程数，默认为1，增大线程数可能可以提升推理速度但是会增加资源消耗
		size_t IntraNumberThreads__ = 1;

		/// @brief OpenVINO推理设备名称（仅在OpenVINO模式下使用，如CPU, GPU, NPU等）
		std::string DeviceName__ = "CPU";

#ifdef USE_OPENVINO
		/// @brief OpenVINO Core对象
		std::unique_ptr<ov::Core> Core__;

		/// @brief OpenVINO 编译模型对象
		std::unique_ptr<ov::CompiledModel> CompiledModel__;

		/// @brief OpenVINO 推理请求对象
		std::unique_ptr<ov::InferRequest> InferRequest__;

		/// @brief 输入节点名称到索引的映射
		std::unordered_map<std::string, size_t> InputNameToIndex__;

		/// @brief 输出节点名称到索引的映射
		std::unordered_map<std::string, size_t> OutputNameToIndex__;
#else
		/// @brief ONNXRuntime的Session对象，用来加载模型和进行推理
		Ort::Session Session__ = Ort::Session(nullptr);

		/// @brief ONNXRuntime的SessionOptions对象，用来配置Session
		Ort::SessionOptions SessionOptions__;

		/// @brief ONNXRuntime的AllocatorWithDefaultOptions对象，用来分配内存
		Ort::AllocatorWithDefaultOptions DefaultAllocator__;

		/// @brief ONNXRuntime的MemoryInfo对象，用来配置内存信息
		Ort::MemoryInfo MemoryInfo__ = Ort::MemoryInfo(nullptr);

		/// @brief ONNXRuntime的IoBinding对象，用来绑定输入输出节点，绑定名称和数据
		Ort::IoBinding IoBinding__ = Ort::IoBinding(nullptr);
#endif

		/// @brief 输入节点名称列表
		std::vector<std::string> InputNodeNames__;

		/// @brief 输出节点名称列表
		std::vector<std::string> OutputNodeNames__;

		/// @brief 输入张量列表，顺序与InputNodeNames一致
		std::vector<TensorType> InputOrtTensors__;

		/// @brief 输出张量列表，顺序与OutputNodeNames一致
		std::vector<TensorType> OutputOrtTensors__;

	private:
		std::atomic<bool> NextEnable__ = true; //未来是否是允许的状态
		bool Enable__ = true; //当前是否是允许的状态
		std::atomic<size_t> NextActionCycleCnt__ = 0; //下一个动作周期计数器
	};
};

@mainpage

# CtrlZ

CtrlZ是一个多线程强化学习部署框架，用于简化学习类机器人运动控制算法在实际机器人上的部署，提升部署的灵活性，通用性，简化部署流程，同时利用多线程推理加速来提升实时性。

# 安装

该项目仅需用户手动安装[onnxruntime](https://onnxruntime.ai/)，其余依赖项均可实现自动安装。用户安装CtrlZ前需要从[onnxruntime官方网站](https://github.com/microsoft/onnxruntime)下载安装，并在本项目根CMakeList.txt中指定onnxruntime的根目录。
**用户也可以使用vcpkg实现自动安装，CtrlZ支持使用vcpkg清单模式安装**

```bash
#将路径换成onnxruntime的安装路径
if(WIN32)
  set(ONNXRUNTIME_ROOT "C:/ProgramFiles/lib/onnxruntime")
elseif(UNIX)
    set(ONNXRUNTIME_ROOT "/usr/local/")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()
```

> * 若要编译本项目文档还需要安装doxygen和graphviz，文档默认不编译，可在cmake变量中设置``BUILD_DOC``为``ON``来启动编译。
> * 若要编译本项目的Bitbot示例(默认不编译)还需下载一些依赖项，请在完全畅通的网络环境下编译
> * 若要编译本项目的Airbot示例(默认不编译)，可能需要先手动安装[airbot robotics](https://airbots.online)的相关依赖库

# 简介

CtrlZ是一个针对于强化学习类控制算法部署问题编写的多线程推理框架。框架实现了传感器信息获取，用户指令控制，网络推理，PD跟随，数据记录等功能。支持建立任务队列实现多线程同步计算，还提供了时间戳功能用于数据同步。

CtrlZ框架主要可以分为``Schedulers``, ``Workers``和``Utils``三个子模块。``Schedulers``模块是调度器，用于实现多线程任务调度功能，``Workers``模块下含各种具体的任务子模块，该模块用于实现下层具体的子功能。用户通过配置``Schedulers``调度器模块调度多个``Workers``模块，组合实现复杂上层任务。``Utils``模块包含了一些常用辅助函数供其余模块开发使用。

![CtrlZ System Design](/doc/CtrlZ_System_Design.svg)

## @ref z::AbstractWorker< SchedulerType > "Schedulers"

调度器类型是CtrlZ框架的核心类型之一，用于管理任务和工作线程，和工人类(Workers)负责执行每一个具体的任务不同，调度器类型主要负责管理任务和工作线程。通过在调度器类型中创建任务队列(TaskList)，并在这些任务队列中添加工作线程(Workers)，调度器类型可以实现多任务调度，每一个任务队列(TaskList)都有一个独立的线程，用于调度这个任务队列中的工作线程。在运行过程中，调度器会根据任务流水线中Worker的顺序在每个流水线循环中依次调用Worker的TaskCycleBegin, TaskRun和TaskCycleEnd方法，在任务开始和结束的时候还会调用TaskCreate和TaskDestroy方法。通过这些方法，工作线程可以在任务开始和结束的时候进行初始化和销毁工作，而在任务运行的时候进行具体的工作。此外调度器还可以为不同的任务队列设置不同的调度周期，通过设置调度周期，可以实现不同任务队列的不同调度频率，从而实现不同任务队列的不同调度速度。调度器分为主线程任务(MainThreadTask)和其他任务队列，主线程任务是一个特殊的任务队列，它是在主线程中运行的，而其他任务队列是在独立的线程中运行的。其他任务队列的频率可以通过设置调度周期来调整，可以设置为主任务队列的整数倍分频(1/n)，而主任务队列的频率是固定的，受spinOnce函数的调用频率控制。调度器类型还可以管理数据管线(DataCenter)，用于存储和获取数据，数据管线中的数据通过时间戳来进行读写数据，存储在数据中心中的数据都是线程安全的，可以在任何线程中读写数据。同时时间辍的存在保证了数据的时序性。

## @ref z::AbstractWorker< SchedulerType > "Workers"

![Workers](/doc/Worker.svg)

AbstractWorker 类型是一切工人类型的基类，在这个类中指定了一些必须要实现的基本方法，这些方法将在调度器的调度流水线下被依次调用来实现工人的工作逻辑。通过将多个Worker类型的工人注册到调度器中，形成一个TaskList，用户可以实现复杂的工作流程，在构成一个流水线的同时，保证每个工人模块的逻辑是独立的，从而实现了工作流程的模块化和可扩展性。这个类的主要目的是为了实现工人的工作逻辑，用户可以通过继承这个类来实现自己的工人类型，然后通过调度器来调度这些工人的工作流程。具体来说，调度器在任务队列创建和删除的时候会调用抽象工人类型的TaskCreate和TaskDestroy方法；而在任务队列的循环中，调度器会依次调用TaskCycleBegin，TaskRun和TaskCycleEnd方法，其中TaskRun方法是用户必须要实现的方法，在用户继承的子类中应当实现当前工作的具体逻辑。举例来说，如果这个Worker是一个用来处理电机的Worker，那么可能的实现是在TaskRun方法中获取当前电机的状态和下发电机的目标状态，在TaskCreate方法中初始化电机的通信接口，在TaskDestroy方法中关闭电机的通信接口。

### 基本Workers功能

* **@ref z::ImuProcessWorker< SchedulerType, ImuType, ImuPrecision > "ImuProcessWorker:"** 实现了IMU获取数据，异常值过滤和滤波的功能。
* **@ref z::KeyboardCommandWorker< SchedulerType > "KeyboardCommandWorker:"** 实现了获取用户键盘命令状态的功能。
* **@ref z::MotorControlWorker< SchedulerType, JointType, MotorPrecision, JointNumber >" MotorControlWorker:"** 实现了电机状态获取，PD控制，电机命令控制等功能。
* **@ref z::MotorResetPositionWorker< SchedulerType, MotorPrecision, JointNumber > "MotorResetPositionWorker:"** 实现了电机复位的功能。
* **@ref z::AsyncLoggerWorker< SchedulerType, LogPrecision, Args > "AsyncLoggerWorker:"** 实现了数据记录的功能。
* **@ref z::NetCmdWorker< SchedulerType, CmdPrecision, CmdArgs > "NetCmdWorker":** 实现了处理用户指令输入的功能。

### 网络推理Workers功能

* **@ref z::AbstractNetInferenceWorker< SchedulerType, InferencePrecision > "AbstractInferenceWorker:"** 实现了onnxruntime的封装，实现了使用onnxruntime推理的功能。
* **@ref z::CommonLocoInferenceWorker< SchedulerType, InferencePrecision, JOINT_NUMBER > "CommonLocoInferenceWorker:"** 实现了运动控制网络基本数据处理功能。
* **@ref z::HumanoidGymInferenceWorker< SchedulerType, InferencePrecision, INPUT_STUCK_LENGTH, JOINT_NUMBER > "HumanoidGymInferneceWorker:"** 实现了[HumanoidGym](https://github.com/roboterax/humanoid-gym)网络相同的推理逻辑。

## Utils

* **@ref z::DataCenter< CTS > "DataCenter:"** 用于存储工作流水线中的数据。
* **@ref z::math "mathTypes"** 定义了@ref z::math::Vector< T, N > "vector"和@ref z::math::Tensor< T, Dims > "tensor"两种数学类型，实现了一些常用数学功能。
* **@ref StaticStringUtils.hpp "StaticStringUtils:"** 用于编译期字符串处理。
* **@ref ZenBuffer.hpp "ZenBuffer:"** 实现了@ref z::RingBuffer< T >  "环形缓冲区"和@ref z::WeightFilter< T >"滤波器"。

# 示例

## BitbotSimulation

BitbotSimulation示例是使用Bitbot Mujoco结合CtrlZ，在Mujoco仿真器中控制机器人的程序，Bitbot也可用于真实机器人部署，关于更多Bitbot的信息请参阅[Bitbot网站](https://bitbot.lmy.name/)。若要编译该示例，需要在cmake中设置``DBUILD_EXAMPLE_BITBOT_SIM``变量为``ON``. *可能需要稳定的国际互联网连接来正确下载部分依赖项*.

## BitbotAirbotARM

BitbotAirbotArm示例是使用CtrlZ结合Airbot机械臂进行抓取控制的示例，若要编译该示例，需要在cmake中设置``BUILD_EXAMPLE_AIRBOT_ARM``为``ON``. *由于SDK限制，该项目仅能在Ubuntu20.04环境下运行，且需要提前安装Airbot机械臂相关[SDK](https://airbots.online/dowload)*。

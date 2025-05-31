/**
 * @file AbstractScheduler.hpp
 * @author Zishun Zhou
 * @brief AbstractScheduler 调度器类型，用于管理任务，工作线程和数据。
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "../Utils/ZObject.hpp"
#include "../Utils/StaticStringUtils.hpp"
#include "../Utils/DataCenter.hpp"
#include "../Workers/AbstractWorker.hpp"
#include <thread>
#include <vector>
#include <string>
#include <map>
#include <condition_variable>
#include <atomic>
#include <nlohmann/json.hpp>
#include <chrono>
#include <functional>
#include <memory>


namespace z
{

    /**
     * @brief AbstractScheduler 调度器类型，用于管理任务，工作线程和数据。
     * @details 调度器类型是CtrlZ框架的核心类型之一，用于管理任务和工作线程，和工人类(Workers)负责执行每一个具体的任务不同，
     * 调度器类型主要负责管理任务和工作线程。通过在调度器类型中创建任务队列(TaskList)，并在这些任务队列中添加工作线程(Workers)，
     * 调度器类型可以实现多任务调度，每一个任务队列(TaskList)都有一个独立的线程，用于调度这个任务队列中的工作线程。在运行过程中，调度器会根据
     * 任务流水线中Worker的顺序在每个流水线循环中依次调用Worker的TaskCycleBegin, TaskRun和TaskCycleEnd方法，在任务开始和结束的时候还会调用TaskCreate和TaskDestroy方法。
     * 通过这些方法，工作线程可以在任务开始和结束的时候进行初始化和销毁工作，而在任务运行的时候进行具体的工作。
     * 此外调度器还可以为不同的任务队列设置不同的调度周期，通过设置调度周期，可以实现不同任务队列的不同调度频率，从而实现不同任务队列的不同调度速度。
     * 调度器分为主线程任务(MainThreadTask)和其他任务队列，主线程任务是一个特殊的任务队列，它是在主线程中运行的，而其他任务队列是在独立的线程中运行的。其他任务队列的频率
     * 可以通过设置调度周期来调整，可以设置为主任务队列的整数倍分频(1/n)，而主任务队列的频率是固定的，受spinOnce函数的调用频率控制。
     * 调度器类型还可以管理数据管线(DataCenter)，用于存储和获取数据，数据管线中的数据通过时间戳来进行读写数据，存储在数据中心中的数据
     * 都是线程安全的，可以在任何线程中读写数据。同时时间辍的存在保证了数据的时序性。
     *
     * @tparam CTS 用于数据中心的键-类型对。其中键是一个编译期字符串常量，类型是数据的类型，通常是一个数字或者一个array。
     * 这些键-类型对用于标识数据中心中的数据，通过键可以获取对应的数据, 通过类型可以确定数据的类型。而且这些键-类型对是在编译期就确定的，
     * 保证了数据的唯一性。具体的使用方法可以参考CTSPair类，CTString类和DataCenter类。
     */
    template<CTSPair ...CTS>
    class AbstractScheduler : public ZObject
    {
        /// @brief 和调度器匹配的工人类型
        using WorkerType = AbstractWorker<AbstractScheduler<CTS...>>;
    public:
        using Ptr = std::shared_ptr<AbstractScheduler<CTS...>>; ///< 调度器类型的指针类型

        static AbstractScheduler<CTS...>::Ptr Create(const nlohmann::json& cfg = nlohmann::json())
        {
            Ptr ptr = std::make_shared<AbstractScheduler<CTS...>>(cfg);
            ptr->weak_ptr = ptr; // 将自身的shared_ptr转换为weak_ptr，方便后续使用
            return ptr;
        }

        /**
         * @brief 创建一个调度器
         *
         */
        AbstractScheduler(const nlohmann::json& cfg = nlohmann::json())
        {
            this->threadId = std::this_thread::get_id();

            if (!cfg.is_null())
            {
                this->spin_dt = cfg["dt"].get<double>();
                this->CheckFrequency__ = cfg["CheckFrequency"].get<bool>();
            }
            else
            {
                std::cout << "Scheduler config is null!" << std::endl;
            }
        }

        /**
         * @brief 销毁调度器
         * @details 销毁调度器时将销毁所有的任务队列(TaskList)但不会销毁Worker，Worker的生命周期由调度器外部管理，
         * 调度器只负责管理任务队列。销毁调度器时，调度器会停止所有的任务队列，并等待所有的任务队列线程结束，主线程任务(MainThreadTask)也会被销毁。
         */
        virtual ~AbstractScheduler()
        {
            this->CanSpin = false;
            for (auto [taskname, task] : TaskList)
            {
                task->isRunning = false;
                task->PauseLock.notify_all();
                this->SyncLock.notify_all();
                task->thread.join();
                delete task;
            }

            for (auto worker : this->MainThreadTaskBlock->workers)
            {
                worker->TaskDestroy();
            }
            for (auto worker : this->ManagedWorkers)
            {
                delete worker;
            }

            delete this->MainThreadTaskBlock;


        }

        /**
         * @brief 从调度器创建一个Worker
         *
         * @details 这个函数用于从调度器中创建一个Worker，并将其添加到调度器的ManagedWorkers列表中。
         * Worker的生命周期由调度器管理，调度器会在销毁时销毁所有的Worker。用户不需要手动销毁Worker。
         * @tparam ptr Worker类型的指针类型，通常是一个继承自AbstractWorker的类型。
         * @tparam Args Worker的构造函数参数类型，可以是任意类型，通常是一些配置参数或者数据。
         * @param args Worker的构造函数参数，可以是任意类型，通常是一些配置参数或者数据。
         * @return ptr* 返回创建的Worker指针，用户可以通过这个指针来访问Worker的成员函数和数据。
         */
        template<class ptr, typename ...Args>
        ptr* CreateWorker(Args&&... args)
        {
            if (this->weak_ptr.expired())
            {
                std::cerr << "Scheduler weak_ptr expired, cannot create worker!" << std::endl;
                return nullptr;
            }
            ptr* worker = new ptr(this->weak_ptr.lock(), std::forward<Args>(args)...);
            this->ManagedWorkers.push_back(worker);
            return worker;
        }


        /**
         * @brief 启动调度器
         * @details 启动调度器时将启动所有的任务队列(TaskList)和主线程任务(MainThreadTask)，启动任务队列时，调度器会调用任务队列中的Worker的TaskCreate方法，
         * 用于初始化Worker。启动调度器后，调度器会开始调度任务队列中的Worker，调度器会根据任务队列中的调度周期来调度任务队列中的Worker，
         * 调度周期是通过调度器的SpinOnce函数来控制的，SpinOnce函数会在主线程中调用，用于调度任务队列中的Worker，当任务队列中的Worker运行过慢时，
         * 调度器会打印一个警告信息。
         *
         * 值得注意的是创建的非主线任务默认是不使能的，需要通过EnableTaskList函数来使能任务队列，使能任务队列后，任务队列中Worker的调度周期才会生效，
         * 任务会在第一个Worker的TaskCycleBegin前被阻塞，直到任务队列被使能。然而在任务创建的时候，任务队列中的Worker会被调用一次TaskCreate方法，
         * 用于初始化Worker，不受使能状态的影响。主线程任务(MainThreadTask)是默认使能的，不受使能状态的影响。
         *
         */
        void Start()
        {
            for (auto [taskname, task] : TaskList)
            {
                task->isRunning = true;
                task->NewRun = false;
                task->thread = std::thread(&AbstractScheduler::run, this, taskname);
                task->threadId = task->thread.get_id();
            }

            for (auto worker : MainThreadTaskBlock->workers)
            {
                worker->TaskCreate();
                fflush(stdout);
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
            this->PrintWelcomeMessage();
            this->CanSpin = true;
        }

        /**
         * @brief 停止调度器，注意停止后无法再次启动调度器，必须重新创建调度器。
         *
         */
        void Stop()
        {
            this->CanSpin = false;
        }

        /**
         * @brief 进行一次调度，调度器将调度主线程任务(MainThreadTask)中的Worker，并为其他任务队列运行一个新的周期。
         * 如果任务运行过慢，调度器会打印一个警告信息。**请不要在Worker中或者其他线程中调用这个函数，这个函数应该在主线程中调用。
         * 否则会出现未定义行为！**
         *
         */
        void SpinOnce()
        {
            if (!this->CanSpin)
            {
                std::cout << "Scheduler is not started, start scheduler before call spin function!" << std::endl;
                return;
            }

            if (this->CheckFrequency__)
                this->CheckRuntimeFrequency();

            this->SyncMutex.lock();

            this->TimeStamp++;

            for (auto [taskname, task] : TaskList)
            {
                task->cnt++;
                task->cnt = task->cnt % task->div;
                if (task->cnt % task->div == 0)
                {
                    if (task->NewRun && !task->Pause)
                    {
                        std::cout << "Task " << taskname << " is running too slow!" << std::endl;
                    }
                    task->NewRun = true;
                }

            }
            this->SyncMutex.unlock();
            this->SyncLock.notify_all();

            this->run_once(MainThreadTaskBlock);
        }

        /**
         * @brief 进行连续任务调度。
         *
         */
        void Spin()
        {
            std::chrono::microseconds sleep_all_time = std::chrono::microseconds(static_cast<long long>(this->spin_dt * 1e6));
            while (this->CanSpin)
            {
                auto start_time = std::chrono::high_resolution_clock::now();
                this->SpinOnce();
                auto end_time = std::chrono::high_resolution_clock::now();
                auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                auto sleep_time = sleep_all_time - time_cost;
                if (sleep_time > std::chrono::microseconds(0))
                {
                    std::this_thread::sleep_for(sleep_time);
                }
                else
                {
                    std::cout << "SpinOnce is too slow!" << std::endl;
                }
            }
        }


        /**
         * @brief 获取时间辍。
         * @details 时间辍是CtrlZ框架中的一个重要概念，用于标识数据的时序性，时间辍是一个递增的整数，每次调用SpinOnce函数时，时间辍都会递增1，
         * 不同的TaskList中的Worker可以通过时间辍来获取数据，通过时间辍可以保证数据的时序性。
         * 时间辍与TaskList中的分频系数无关，仅与SpinOnce函数的调用频率有关。
         *
         * @return size_t 时间辍
         */
        size_t getTimeStamp()
        {
            return this->TimeStamp.load();
        }

        /**
         * @brief 获取调度器的时间步长
         * @details 调度器的时间步长是调度器是所有任务队列的基础调度间隔，也是MainTaskList的间隔。
         * 这个间隔由配置文件设置，但实际调用频率和调度器的SpinOnce函数的调用频率有关。SpinOnce会检查
         * 设置的时间步长是否和实际调用频率一致，如果不一致，会打印警告信息。
         *
         * @return double 调度器的时间步长(seconds)
         */
        double getSpinOnceTime()
        {
            return this->spin_dt;
        }

        /**
         * @brief 创建一个任务队列，在调度器启动时，Workers中的TaskCreate方法会被调用，用于初始化每个Worker。
         *
         * @param TaskName 任务名
         * @param div 分频系数，用于控制任务的调度频率，任务的调度频率是主线程任务的1/div。如果是主线程任务，div将被忽略。
         * @param MainThreadTask 是否是主线程任务。
         * @return size_t 任务ID
         */
        size_t CreateTaskList(const std::string& TaskName, size_t div, bool MainThreadTask = false)
        {
            if (TaskList.find(TaskName) != TaskList.end())
            {
                std::cout << "Task " << TaskName << " already exists!" << std::endl;
                return -1;
            }

            if (MainThreadTask && this->MainThreadTaskBlock)
            {
                std::cout << "Main thread task already exists, create task list failed." << std::endl;
                return -1;
            }

            TCB* tcb = new TCB();
            tcb->TaskName = TaskName;
            tcb->div = div;
            tcb->TaskId = TaskList.size() + 1;
            tcb->isRunning = false;
            tcb->NewRun = false;
            tcb->Pause = true;
            tcb->cnt = 0;

            if (MainThreadTask)
            {
                tcb->div = 1;
                MainThreadTaskBlock = tcb;
                MainThreadTaskName = TaskName;
            }
            else
            {
                TaskList[TaskName] = tcb;
            }

            return MainThreadTask ? 0 : tcb->TaskId;

        }

        /**
         * @brief 删除一个任务队列。
         * @details 删除一个任务队列时，调度器会停止这个任务队列，并等待当前周期结束，然后销毁任务队列。
         * 销毁时将依次调用Worker的TaskDestroy方法。注意主线程任务(MainThreadTask)不能被销毁。
         *
         * @param TaskName 待销毁的任务名
         */
        void DestroyTaskList(const std::string& TaskName)
        {
            if (TaskName == this->MainThreadTaskName)
            {
                std::cout << "Main thread task cannot be destroyed!" << std::endl;
                return;
            }

            if (TaskList.find(TaskName) == TaskList.end())
            {
                std::cout << "Task " << TaskName << " does not exist!" << std::endl;
                return;
            }

            TCB* tcb = TaskList[TaskName];
            tcb->isRunning = false;
            tcb->PauseLock.notify_all();
            this->SyncLock.notify_all();
            if (tcb->thread.joinable())
                tcb->thread.join();
            delete tcb;
            TaskList.erase(TaskName);
        }

        /**
         * @brief 设置任务列表的启用状态
         * @details 此函数用于设置任务的启用状态。如果任务被启用，调度器将根据分频设置为该任务进行调度。
         * 如果任务被禁用，任务将不会被调度，并且会一直被阻塞，直到再次启用。
         * 需要注意的是，主线程任务始终处于启用状态，不受此函数的影响。
         * @param TaskName 任务名称
         * @return true 任务已启用
         * @return false 任务未启用（可能找不到任务）
         */
        bool EnableTaskList(const std::string& TaskName)
        {
            if (TaskList.find(TaskName) == TaskList.end() && TaskName != this->MainThreadTaskName)
            {
                std::cout << "Task " << TaskName << " does not exist!" << std::endl;
                return false;
            }

            if (TaskName == this->MainThreadTaskName)
            {
                std::cout << "Main thread task cannot be enable or disabled!" << std::endl;
                return false;
            }

            TCB* tcb = TaskList[TaskName];
            tcb->PauseMutex.lock();
            tcb->Pause = false;
            tcb->PauseMutex.unlock();
            tcb->PauseLock.notify_all();
            return true;
        }

        /**
         * @brief 设置任务列表的禁用状态
         * @details 该函数的作用与EnableTaskList函数相反。
         * @param TaskName 任务名称
         * @return true 任务已禁用
         * @return false 任务未禁用（可能找不到任务）
         */
        bool DisableTaskList(const std::string& TaskName)
        {
            if (TaskList.find(TaskName) == TaskList.end() && TaskName != this->MainThreadTaskName)
            {
                std::cout << "Task " << TaskName << " does not exist!" << std::endl;
                return false;
            }

            if (TaskName == this->MainThreadTaskName)
            {
                std::cout << "Main thread task cannot be enable or disabled!" << std::endl;
                return false;
            }

            TCB* tcb = TaskList[TaskName];
            tcb->PauseMutex.lock();
            tcb->Pause = true;
            tcb->PauseMutex.unlock();
            tcb->PauseLock.notify_all();

            return true;
        }


        /**
         * @brief 向任务中添加一个工人
         *
         * @param TaskName 任务名称
         * @param worker 工人
         */
        void AddWorker(const std::string& TaskName, WorkerType* worker)
        {
            if (TaskList.find(TaskName) == TaskList.end() && TaskName != this->MainThreadTaskName)
            {
                std::cout << "Task " << TaskName << " does not exist!" << std::endl;
                return;
            }
            if (TaskName == MainThreadTaskName)
            {
                MainThreadTaskBlock->workers.push_back(worker);
                MainThreadTaskBlock->workerFuncBegins.push_back(std::bind(&WorkerType::TaskCycleBegin, worker));
                MainThreadTaskBlock->workerFuncRuns.push_back(std::bind(&WorkerType::TaskRun, worker));
                MainThreadTaskBlock->workerFuncEnds.push_back(std::bind(&WorkerType::TaskCycleEnd, worker));
            }

            else
            {
                TaskList[TaskName]->workers.push_back(worker);
                TaskList[TaskName]->workerFuncBegins.push_back(std::bind(&WorkerType::TaskCycleBegin, worker));
                TaskList[TaskName]->workerFuncRuns.push_back(std::bind(&WorkerType::TaskRun, worker));
                TaskList[TaskName]->workerFuncEnds.push_back(std::bind(&WorkerType::TaskCycleEnd, worker));
            }

        }


        /**
         * @brief 批量向任务中添加工人
         *
         * @param TaskName 任务名称
         * @param workers 工人列表，列表顺序即为任务队列中工人调度的顺序
         */
        void AddWorkers(const std::string& TaskName, std::vector<WorkerType*> workers)
        {
            if (TaskList.find(TaskName) == TaskList.end() && TaskName != this->MainThreadTaskName)
            {
                std::cout << "Task " << TaskName << " does not exist!" << std::endl;
                return;
            }
            for (auto worker : workers)
            {
                if (TaskName == MainThreadTaskName)
                {
                    MainThreadTaskBlock->workers.push_back(worker);
                    MainThreadTaskBlock->workerFuncBegins.push_back(std::bind(&WorkerType::TaskCycleBegin, worker));
                    MainThreadTaskBlock->workerFuncRuns.push_back(std::bind(&WorkerType::TaskRun, worker));
                    MainThreadTaskBlock->workerFuncEnds.push_back(std::bind(&WorkerType::TaskCycleEnd, worker));
                }
                else
                {
                    TaskList[TaskName]->workers.push_back(worker);
                    TaskList[TaskName]->workerFuncBegins.push_back(std::bind(&WorkerType::TaskCycleBegin, worker));
                    TaskList[TaskName]->workerFuncRuns.push_back(std::bind(&WorkerType::TaskRun, worker));
                    TaskList[TaskName]->workerFuncEnds.push_back(std::bind(&WorkerType::TaskCycleEnd, worker));
                }

            }
        }

        /**
         * @brief 从数据中心中获取数据，并自动更新时间戳
         *
         * @tparam CT 数据名称
         * @tparam T 数据类型
         * @param data 数据
         */
        template<CTString CT, typename T>
        void SetData(const T& data)
        {
            dataCenter.template SetData<CT, T>(this->TimeStamp.load(), data);
        }

        /**
         * @brief 从数据中心中获取数据
         *
         * @tparam CT 数据名称
         * @tparam T 数据类型
         * @param data 数据
         * @return size_t 数据的时间戳
         */
        template<CTString CT, typename T>
        size_t GetData(T& data)
        {
            return dataCenter.template GetData<CT, T>(data);
        }
    protected:

        /**
         * @brief Task Control Block
         *
         */
        struct TCB
        {
            /// @brief worker list
            std::vector<WorkerType*> workers;

            /// @brief worker functions, used for lambda functions
            std::vector<std::function<void(void)>> workerFuncBegins;
            std::vector<std::function<void(void)>> workerFuncRuns;
            std::vector<std::function<void(void)>> workerFuncEnds;

            /// @brief division control of the task
            size_t div;

            /// @brief division counter of the task
            std::atomic<size_t> cnt;

            /// @brief thread id of the task
            std::thread::id threadId;

            /// @brief task name
            std::string TaskName;

            /// @brief task id
            size_t TaskId;

            /// @brief thread of the task
            std::thread thread;

            /// @brief check if the task is running
            bool isRunning;

            /// @brief for generating a new pulse
            std::atomic<bool> NewRun;

            std::mutex PauseMutex;
            std::condition_variable PauseLock;
            std::atomic<bool> Pause;

            /**
             * @brief Construct a new TCB object
             *
             */
            TCB() : isRunning(false), div(1), Pause(true) {}
        };

    protected:
        std::atomic<bool> CanSpin = false; //是否可以进行调度

        /// @brief main thread id
        std::thread::id threadId;

        /// @brief task list
        std::map<std::string, TCB*> TaskList;

        /// @brief control block of the main thread task
        TCB* MainThreadTaskBlock = nullptr;

        /// @brief main thread task name
        std::string MainThreadTaskName;

        /// @brief time stamp
        std::atomic<size_t> TimeStamp;

        /// @brief data center for storing data with time stamp
        DataCenter<CTS...> dataCenter;

        /// @brief task list sync mutex
        std::mutex SyncMutex;

        /// @brief task list sync lock variable
        std::condition_variable SyncLock;

        /// @brief worker list, these workers are created by the scheduler and managed by the scheduler.
        std::vector<WorkerType*> ManagedWorkers;

        std::weak_ptr<AbstractScheduler<CTS...>> weak_ptr; ///< weak pointer to the scheduler, used for worker to access scheduler

        /// @brief task list spin once time
        double spin_dt = 0.001; // 1ms
        double HistorySpinDt = 0;
        std::chrono::steady_clock::time_point last_time;
        bool CheckFrequency__ = true;

    protected:
        /**
         * @brief 运行任务的一个调度周期
         * @details 在一个调度周期内，调度器将按照顺序调用任务中的工作者。工作者将按照它们被添加到任务中的顺序被调用。对于每个工作者，将依次调用TaskCycleBegin、TaskRun、TaskCycleEnd函数。
         * @param tcb 给定的任务控制块
         */
        void run_once(TCB* tcb)
        {
            //std::cout << "task:" << tcb->TaskName << " is running in cycle" << this->TimeStamp.load() << std::endl;

            for (auto& func : tcb->workerFuncBegins)
            {
                func();
            }


            for (auto& func : tcb->workerFuncRuns)
            {
                func();
            }

            for (auto& func : tcb->workerFuncEnds)
            {
                func();
            }

            tcb->NewRun = false;
        }

        /**
         * @brief 运行任务
         *
         * @param TaskName 任务名
         */
        void run(const std::string& TaskName)
        {
            TCB* tcb = TaskName == MainThreadTaskName ? MainThreadTaskBlock : TaskList[TaskName];
            tcb->cnt = 0;
            for (auto worker : tcb->workers)
            {
                worker->TaskCreate();
                fflush(stdout);
            }
            tcb->NewRun = true;
            tcb->isRunning = true;
            while (tcb->isRunning)
            {
                {
                    std::unique_lock<std::mutex> lck(tcb->PauseMutex);
                    tcb->PauseLock.wait(lck, [tcb] {return !tcb->Pause || !tcb->isRunning;});
                }

                {
                    std::unique_lock<std::mutex> lck(this->SyncMutex);
                    this->SyncLock.wait(lck, [tcb] {return tcb->NewRun || !tcb->isRunning;});
                }

                if (!tcb->isRunning)
                    break;

                run_once(tcb);
            }
            for (auto worker : tcb->workers)
            {
                worker->TaskDestroy();
            }
        }


        void CheckRuntimeFrequency()
        {
            auto t = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t - this->last_time).count();
            this->last_time = t;
            float dt = static_cast<float>(duration) / 1000000.0;
            if (this->HistorySpinDt == 0)
                this->HistorySpinDt = this->spin_dt; //first time, set to spin_dt
            else [[likely]]
                this->HistorySpinDt = this->HistorySpinDt * 0.999 + dt * 0.001;

            if (this->HistorySpinDt > this->spin_dt * 1.4 || this->HistorySpinDt < this->spin_dt * 0.6)
            {
                //set to red color
                std::cout << "\033[31m" << "Scheduler spin dt is not consistent with the set dt, please check your code!" << "\033[0m" << std::endl;
                std::cout << "Current dt: " << this->HistorySpinDt << ", set dt: " << this->spin_dt << std::endl;
            }
        }

        void PrintWelcomeMessage()
        {
            fflush(stdout);
            std::string line0 = "\033[32m============================================\033[0m";
            std::string line1 = "\033[32m||  CCCC  TTTTT  RRRRR  L           ZZZZZ ||\033[0m";
            std::string line2 = "\033[32m|| C        T    R   R  L              Z  ||\033[0m";
            std::string line3 = "\033[32m|| C        T    RRRRR  L    =====    Z   ||\033[0m";
            std::string line4 = "\033[32m|| C        T    R R    L            Z    ||\033[0m";
            std::string line5 = "\033[32m||  CCCC    T    R  RR  LLLLL       ZZZZZ ||\033[0m";
            std::string line6 = "\033[32m============================================\033[0m";
            fflush(stdout);
            std::cout << std::endl << std::endl << std::endl;
            std::cout << line0 << std::endl;
            std::cout << line1 << std::endl;
            std::cout << line2 << std::endl;
            std::cout << line3 << std::endl;
            std::cout << line4 << std::endl;
            std::cout << line5 << std::endl;
            std::cout << line6 << std::endl;
            std::cout << "\033[31mCtrl-Z is a muti-thread RL deployment framework for Robot locomotion. \033[0m" << std::endl;
            std::cout << "\033[33mVisit http://opensource.zzshub.cn/CtrlZ for detailed documentation.\033[0m" << std::endl;
            std::cout << std::endl << std::endl;
        }
    };
};
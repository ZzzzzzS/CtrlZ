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


namespace zzs
{
    /**
     * @brief AbstractScheduler class, the base class for all schedulers
     * @details the scheduler is responsible for managing the tasks and workers
     * each task has a list of workers, and the scheduler will call the workers in the task in order,
     * these tasks will run in different threads simultaneously,
     * the scheduler will also call the workers in the main thread task
     *
     * @tparam CTS
     */
    template<CTSPair ...CTS>
    class AbstractScheduler : public ZObject
    {
        /// @brief worker type
        using WorkerType = AbstractWorker<AbstractScheduler<CTS...>>;
    public:

        /**
         * @brief Construct a new Abstract Scheduler object
         *
         */
        AbstractScheduler()
        {
            this->threadId = std::this_thread::get_id();
        }

        /**
         * @brief Destroy the Abstract Scheduler object
         * @details the scheduler will destroy all tasks and workers,
         * workers in other threads will be destroyed by the thread, when the thread is destroyed,
         * the main thread task will also be destroyed
         *
         */
        virtual ~AbstractScheduler()
        {
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
            delete this->MainThreadTaskBlock;
        }

        /**
         * @brief Start the scheduler
         * @details the scheduler will start all tasks and workers
         * the main thread task will also be started, when tasks are started,
         * the TaskCreate function of the workers will be called
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
            }
        }

        /**
         * @brief Spin once of the scheduler
         * @details the scheduler will call the workers in the main thread task,
         * and generate a new pulse for other tasks, if a task is running too slow,
         * the scheduler will print a warning message
         *
         */
        void SpinOnce()
        {
            //TODO: solve the problem of calling this function in the main thread
            /*if (std::this_thread::get_id() != this->threadId)
            {
                std::cout << "This function should be called in the main thread!" << std::endl;
                return;
            }*/

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
         * @brief Get the Time Stamp object
         *
         * @return size_t time stamp
         */
        size_t getTimeStamp()
        {
            return this->TimeStamp.load();
        }

        /**
         * @brief Create a Task List
         *
         * @param TaskName task name
         * @param div division of the task, this task will run every div cycles, this will be ignored if it is a main thread task
         * @param MainThreadTask check if this task is a main thread task
         * @return size_t task id
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
         * @brief Destroy a Task List
         *
         * @param TaskName task name
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
         * @brief Set enable status of a Task List
         * @details the function is use to set the enable status of a task, if the task is enabled, the scheduler will schedule a new pulse for
         * the task according to the division, if the task is disabled, the task will not be scheduled, and will be blocked until it is enabled.AbstractScheduler
         * However, it is worth mentioning that the main thread task is always enabled, and cannot be influenced by this function.
         * @param TaskName task name
         * @return true the task is enabled
         * @return false the task is not enabled(task could not be found)
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
         * @brief set disable status of a Task List
         * @details do the opposite of the EnableTaskList function
         * @param TaskName task name
         * @return true the task is disabled
         * @return false the task is not disabled(task could not be found)
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
         * @brief Add a Worker to a Task
         *
         * @param TaskName task name
         * @param worker worker
         */
        void AddWorker(const std::string& TaskName, WorkerType* worker)
        {
            if (TaskList.find(TaskName) == TaskList.end() && TaskName != this->MainThreadTaskName)
            {
                std::cout << "Task " << TaskName << " does not exist!" << std::endl;
                return;
            }
            if (TaskName == MainThreadTaskName)
                MainThreadTaskBlock->workers.push_back(worker);
            else
                TaskList[TaskName]->workers.push_back(worker);
        }


        /**
         * @brief Add a list of Workers to a Task
         *
         * @param TaskName task name
         * @param workers worker list
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
                    MainThreadTaskBlock->workers.push_back(worker);
                else
                    TaskList[TaskName]->workers.push_back(worker);
            }
        }

        /**
         * @brief Set the Data object
         *
         * @tparam CT
         * @tparam T
         * @param data
         */
        template<CTString CT, typename T>
        void SetData(const T& data)
        {
            dataCenter.template SetData<CT, T>(this->TimeStamp.load(), data);
        }

        /**
         * @brief Get the Data object
         *
         * @tparam CT
         * @tparam T
         * @param data
         * @return size_t time stamp
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

    protected:
        /**
         * @brief run one pulse of the task
         * @details during one pulse, the scheduler will call the workers in the task in order
         * the workers will be called in the order they are added to the task,
         * for each worker, the TaskCycleBegin, TaskRun, TaskCycleEnd functions will be called in order
         * @param tcb given task control block
         */
        void run_once(TCB* tcb)
        {
            //std::cout << "task:" << tcb->TaskName << " is running in cycle" << this->TimeStamp.load() << std::endl;
            for (auto worker : tcb->workers)
            {
                worker->TaskCycleBegin();
            }

            for (auto worker : tcb->workers)
            {
                worker->TaskRun();
            }

            for (auto worker : tcb->workers)
            {
                worker->TaskCycleEnd();
            }

            tcb->NewRun = false;
        }

        /**
         * @brief run the tasks
         *
         * @param TaskName
         */
        void run(const std::string& TaskName)
        {
            TCB* tcb = TaskName == MainThreadTaskName ? MainThreadTaskBlock : TaskList[TaskName];
            tcb->cnt = 0;
            for (auto worker : tcb->workers)
            {
                worker->TaskCreate();
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
    };
};
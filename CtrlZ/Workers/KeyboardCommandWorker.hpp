/**
 * @file KeyboardCommandWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <stdio.h>
#ifdef _WIN32
#include <conio.h>
#endif
#include <functional>
#include "AbstractWorker.hpp"
#include "Schedulers/AbstractScheduler.hpp"

 //check linux
#ifdef __linux__
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// Function to check if a key has been pressed
int _kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

// Function to get a character without waiting for Enter key
int _getch(void) {
    struct termios oldt, newt;
    int ch;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}
#endif

namespace z
{
    /**
     * @brief KeyboardCommandWorker 类型是一个键盘命令工人类型，用户可以通过这个工人类型来实现键盘命令的功能。
     * @details KeyboardCommandWorker 类型是一个键盘命令工人类型，用户可以通过这个工人类型来实现键盘命令的功能。
     * 该类型会在TaskRun方法中检测键盘输入，并根据用户注册的回调函数来执行相应的操作。
     * 用户可以通过注册回调函数来实现键盘输入的功能，比如用户可以通过注册回调函数来实现键盘输入的控制逻辑。
     * 该类型适用于一些需要通过键盘输入来控制的场景，比如调试阶段的控制逻辑。
     *
     * @tparam SchedulerType 调度器类型
     */
    template<typename SchedulerType>
    class KeyboardCommandWorker : public AbstractWorker<SchedulerType>
    {
    public:
        /// @brief 键盘输入回调函数类型，函数签名为void(SchedulerType::Ptr)，包含一个调度器的指针
        using KeyCallbackType = std::function<void(SchedulerType::Ptr)>;
    public:
        /**
         * @brief 构造一个键盘命令工人类型
         *
         * @param scheduler 调度器的指针
         * @param cfg 配置文件
         */
        KeyboardCommandWorker(SchedulerType::Ptr scheduler, const nlohmann::json& cfg = nlohmann::json())
            :AbstractWorker<SchedulerType>(scheduler, cfg)
        {
            for (size_t i = 0; i < 256; i++)
            {
                KeyCallbackArray[i] = nullptr;
            }
        }

        /**
         * @brief 注册一个键盘输入回调函数
         *
         * @param key 绑定的按键
         * @param callback 回调函数，函数签名为void(SchedulerType::Ptr)
         */
        void RegisterKeyCallback(char key, KeyCallbackType callback)
        {
            KeyCallbackArray[key] = callback;
        }

        /**
         * @brief 处理键盘输入的工作逻辑
         *
         */
        void TaskRun() override
        {
            if (_kbhit())
            {
                char key = _getch();
                if (KeyCallbackArray[key] != nullptr)
                {
                    KeyCallbackArray[key](this->Scheduler);
                }
            }
        }

    private:
        /// @brief 键盘输入回调函数数组
        std::array<KeyCallbackType, 256> KeyCallbackArray;
    };
};

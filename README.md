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

TBC...

# 示例

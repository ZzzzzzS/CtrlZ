# Data Structure
# Data Structure
To leverage the 3D reconstruction tasks, we designed a series of data types. The data types are divided into three categories: object types, workflow types, and configuration types. The object types are the basic data types used to store data (e.g. camera pose, maps) in the reconstruction framework. Each object type is ending with "Object" (e.g. CameraObject) The workflow types are the basic process types that perform sub-tasks. The configuration types are used for load/unload settings and managing the pipeline in the reconstruction framework. The data types are shown in the figure below. Details of each data structure can be found in its corresponding pages.

## Object Types
| Class Name | Class Description |
| :--- | :--- |
|DataFlowObject|Data Foundation Type|
|CameraObject|Camera type to save camera information like intrinsic parameters |
|MapPointObject|Map point type to save map point information like 3D coordinates, descriptors, and observations|
|FrameObject|Frame type to save frame information like image, feature points, and pose|
|EdgeObject|Edge type to save edge information between different FrameObject or MapPointObject|
|GlobalMapObject|Map type to save global map information|
|DenseMapObject| map type to save dense point cloud information|


## Workflow Types
| Class Name | Class Description |
| :--- | :--- |
|WorkFlowObject|Workflow Foundation type|
|Photographer|Photographer type to control the CameraObject types for providing image input|
|FeatureExtractor|FeatureExtractor type to extract feature points and descriptors from FrameObject types|
|FeatureMatcher|FeatureMatcher type to match feature points between FrameObject types|
|PoseReconstructor|PoseReconstructor type to reconstruct camera poses and MapPointObject from FrameObject types|
|PoseOptimizer|PoseOptimizer type to optimize camera poses and MapPointObject types|
|DenseReconstructor|DenseReconstructor type to reconstruct dense point cloud from FrameObject types|
|DenseOptimizer|DenseOptimizer type to optimize dense point cloud|
|DenseFilter|DenseFilter type to filter dense point cloud|

## Configuration & Other Types
| Class Name | Class Description |
| :--- | :--- |
|MVSObject|MVSObject type is the root class of this reconstruction framework|
|ConfigurationObject|ConfigurationObject type to load/unload settings and managing the pipeline in the reconstruction framework|
|JsonSaver|JsonSaver type to read/write data into json file|

# Inheritance Diagram

![](./class_m_v_s_object__inherit__graph.png)

<center> Overview of all classes' inheritance relationships /<center>


![](class_work_flow_object__inherit__graph.png)

<center> Overview of workflow classes' inheritance relationships </center>

This is the overview of workflow classes' inheritance relationships. The workflow classes are the basic process types that perform sub-tasks, to better use the signal-slot architecture, the workflow classes are inherited from the [QObject](https://doc.qt.io/qt-5/qobject.html) class as well.



![](./bodybg.png)
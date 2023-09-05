# SHINAL-DL 框架项目说明
## 介绍
SHINAL-DL是一个基于分离式、层次化的通用神经网络算法的深度学习框架(general **S**eparate interface and **HI**erarchical structure of **N**eural network **A**lgorithm **L**ibrary of **DL** Framework，*SHINAL-DL*)(以下简称“框架”)。框架对包括底层自动微分、求导、对象自适应创建释放、矩阵运算、优化计算，以及上层的神经网络算法等内容进行抽象、封装，再通过派生和模板化技术，设计出整个神经网络运算的模块化、层次化结构，实现灵活通用的深度学习框架。
## 环境要求

- C++20
- Eigen:3.4.0
## 使用说明

文件夹描述
- "framework"文件夹包括"header"和"src"两个文件夹，分别包含框架实现的头文件以及"*.cpp"源文件
- "examples"文件夹包括两个基于框架实现的案例源文件，分别为MNIST手写体数字识别以及基于PINNS求解Burgers方程的案例实现。
- "dataset"文件夹包括项目中两个案例实现的数据集，其中MNIST手写体数据集因文件体积过大未上传，开发者可自行前往官网下载数据集，数据集说明见"dataset/MNIST_dataset/README.md"文件

运行说明
1. 开发者下载项目到本地并配置好环境后，根据自己的数据集路径运行"examples"文件夹下的两个案例文件。本项目测试是基于VS2019，解决方案平台为x64。
2. 若开发者想要运行基于PINNS求解Burgers方程的案例，需要配置读取"*.mat"文件的dll路径。本项目采用的是MatlabR2018a版本的dll工具。

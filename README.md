# Yolov8 Instance Segmentation Demo

# Project Description
This project is a demo of instance segmentation using Yolov8 and ONNX Runtime. It is written in C++ and utilizes the OpenCV library for image processing. With this demo, you can learn how to load a model, perform image preprocessing, run inference, post-process the results, and generate instance segmentation images.  
vx:littleeyu

# Features
Load Yolov8 model for instance segmentation inference
Image preprocessing, including resizing and normalization
Draw bounding boxes and instance segmentation masks on images
Support for multiple classes and threshold settings
High-performance inference using ONNX Runtime
# Installation and Usage
To install and use the project, follow these steps:

Clone the project to your local machine:

```bash
git clone https://github.com/JiayuXu0/YOLOV8-Seg-ONNX.git
```
Navigate to the project directory:

```bash
cd YOLOV8-Seg-ONNX
```
Build the project:

```bash
mkdir build
cd build
cmake ..
make
```
Run the demo:

```bash
./bin/MaskONNXDemo
```
# Dependencies
This project depends on the following software and libraries:

OpenCV 2.x or higher
ONNX Runtime C++ API

--------------


# 项目名称

Yolov8 实例分割 Demo  
vx:littleeyu

# 项目描述

这个项目是一个使用 Yolov8 和 ONNX Runtime 进行实例分割推理的示例。它使用 C++ 编写，并利用 OpenCV 库进行图像处理。通过该示例，您可以了解如何加载模型、进行图像预处理、运行推理、后处理结果并生成实例分割图像。

# 功能特性

加载 Yolov8 模型进行实例分割推理图像预处理，包括调整大小和归一化在图像上绘制边界框和实例分割结果支持多类别和阈值设置使用 ONNX Runtime 进行高性能推理

# 安装和使用

请按照以下步骤安装和使用该项目：

克隆项目到本地计算机：
```bash
git clone https://github.com/JiayuXu0/YOLOV8-Seg-ONNX.git
```
进入项目目录：
```bash
cd YOLOV8-Seg-ONNX
```
编译项目：

```bash
mkdir build
cd build
cmake ..
make
```
运行示例：

```bash
./bin/ MaskONNXDemo
```

# 依赖

该项目依赖以下软件和库：

OpenCV 2.x 或更高版本  
ONNX Runtime C++ API



cmake_minimum_required(VERSION 3.10)
project(MaskONNXDemo)

set(CMAKE_CXX_STANDARD 11)


# 设置OpenCV的安装路径
set(OpenCV_DIR /data/anaconda3/lib)
find_package(OpenCV REQUIRED)
link_directories(/data/Code/onnxruntime/build/Linux/RelWithDebInfo)
link_directories(/data/Code/opencv/opencv-4.5.5/build/lib)
include_directories(/data/Code/onnxruntime/include/)
include_directories(/data/Code/onnxruntime/include/onnxruntime/core/session)
include_directories(/data/anaconda3/include/opencv4)
# 添加可执行文件
add_executable(MaskONNXDemo inference.cpp)


# 链接TensorRT和OpenCV库
target_link_libraries(MaskONNXDemo PRIVATE ${OpenCV_LIBRARIES})
target_link_libraries(MaskONNXDemo PRIVATE ${CUDA_LIBRARIES})

# 设置库路径
target_link_libraries(MaskONNXDemo PRIVATE libnvinfer.so)
target_link_libraries(MaskONNXDemo PRIVATE libonnxruntime.so)
target_link_libraries(MaskONNXDemo PRIVATE libonnxruntime_providers_shared.so)
target_link_libraries(MaskONNXDemo PRIVATE libcustom_op_library.so )
target_link_libraries(MaskONNXDemo PRIVATE libcustom_op_invalid_library.so)
# 设置生成的可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)



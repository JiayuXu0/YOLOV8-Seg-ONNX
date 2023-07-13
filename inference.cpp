#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>

using namespace cv;

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
    float mask[32];
};

cv::Mat sigmoid(const cv::Mat& inputMat)
{
    cv::Mat outputMat;
    cv::exp(-inputMat, outputMat);
    cv::divide(1, 1 + outputMat, outputMat);
    return outputMat;
}

int main(int argc, char* argv[])
{
    // Load the model and image
    std::string modelFilepath{"yolov8n-seg.onnx"};
    std::string imageFilepath{"demo.png"};
    int num_classes = 80;
    std::vector<cv::Mat> images;
    auto img = imread(imageFilepath, cv::IMREAD_COLOR);
    images.push_back(img);
    images.push_back(img);

    // Set thresholds
    float modelScoreThreshold = 0.5;
    float modelNMSThreshold = 0.5;

    // Initialize onnxruntime
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session.GetInputCount();
    std::cout << "Input nodes: " << numInputNodes << std::endl;
    const auto ptr = session.GetInputNameAllocated(0, allocator);
    const char* inputName = &*ptr;
    std::cout << "Input name: " << inputName << std::endl;

    size_t numOutputNodes = session.GetOutputCount();
    std::cout << "Output nodes: " << numOutputNodes << std::endl;
    const auto outputptr = session.GetOutputNameAllocated(0, allocator);
    const char* outputName = &*outputptr;
    std::cout << "Output name: " << outputName << std::endl;
    const auto outputptr2 = session.GetOutputNameAllocated(1, allocator);
    const char* outputName2 = &*outputptr2;
    std::cout << "Output name: " << outputName2 << std::endl;
    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName, outputName2};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    Ort::TypeInfo outputTypeInfo2 = session.GetOutputTypeInfo(1);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    auto outputTensorInfo2 = outputTypeInfo2.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::vector<int64_t> outputDims2 = outputTensorInfo2.GetShape();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::vector<Ort::Value> input_tensors;

    // Image preprocessing and push to input tensor
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat blob;
    blob = cv::dnn::blobFromImages(images, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    std::cout << "Output name: " << outputName2 << std::endl;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, (float*)blob.data, inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3], inputDims.data(),
        inputDims.size()));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Preprocessing time: " << duration.count() << " milliseconds" << std::endl;

    // Output tensor preprocessing
    std::cout << outputDims[2] << std::endl;
    std::vector<float> outputTensorValues(outputDims[0] * outputDims[1] * outputDims[2]);
    std::vector<Ort::Value> output_tensors;
    output_tensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputDims[0] * outputDims[1] * outputDims[2], outputDims.data(), outputDims.size()));
    std::vector<float> outputTensorValues2(outputDims2[0] * outputDims2[1] * outputDims2[2] * outputDims2[3]);
    output_tensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues2.data(), outputDims2[0] * outputDims2[1] * outputDims2[2] * outputDims2[3], outputDims2.data(), outputDims2.size()));

    // Inference
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
        input_tensors.data(), 1 /*Number of inputs*/, outputNames.data(),
        output_tensors.data(), 2 /*Number of outputs*/);

    start = std::chrono::high_resolution_clock::now();

    // Get inference results
    Ort::Value& tensor = output_tensors[0];
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    float* data = tensor.GetTensorMutableData<float>();
    cv::Mat mat(shape[1], shape[2], CV_32F, data);
    int rows = shape[2];
    int dimensions = shape[1];
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::transpose(mat, mat);
    data = (float*)mat.data;
    std::vector<int> num_index;

    // Get output box coordinates
    for (int i = 0; i < rows; ++i)
    {
        float* classes_scores = data + 4;

        cv::Mat scores(1, num_classes, CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > modelScoreThreshold)
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w));
            int top = int((y - 0.5 * h));

            int width = int(w);
            int height = int(h);
            num_index.push_back(i);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += dimensions;
    }

    // Non-maximum suppression
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
    cv::Mat resultimage(640, 640, CV_8UC1, cv::Scalar(0));
    std::vector<Detection> detections{};

    // Get the final results
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[i];
        result.confidence = confidences[idx];

        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(111, 222, 111);
        // TODO
        result.className = "1";
        result.box = boxes[idx];
        std::copy((float*)mat.data + dimensions * num_index[nms_result[i]] + 4 + num_classes, (float*)mat.data + dimensions * num_index[nms_result[i]] + 4 + num_classes + 32, result.mask);
        detections.push_back(result);
    }

    Ort::Value& tensor2 = output_tensors[1];
    auto shape2 = tensor2.GetTensorTypeAndShapeInfo().GetShape();

    float* data2 = tensor2.GetTensorMutableData<float>();
    float* maskptr = tensor2.GetTensorMutableData<float>();
    cv::Mat maskMat(shape2[1], shape2[2] * shape2[3], CV_32F, maskptr);

    // Mask post-processing
    for (auto detection : detections) {
        cv::Mat tmpMat(1, 32, CV_32F, detection.mask);
        cv::Mat tmpResult;
        cv::gemm(tmpMat, maskMat, 1.0, cv::Mat(), 0, tmpResult);
        cv::Mat reshapedImage = tmpResult.reshape(1, 160);
        cv::Mat sigmoidMat = sigmoid(reshapedImage);  // Calculate Sigmoid
        cv::Mat mat(160, 160, CV_8UC1, cv::Scalar(0));

        cv::Rect rfor4(int(detection.box.x / 4), int(detection.box.y / 4), int(detection.box.width / 4), int(detection.box.height / 4));
        cv::Mat roi = sigmoidMat(rfor4);
        cv::Mat crop_mask;
        cv::resize(roi, crop_mask,
            cv::Size(detection.box.width, detection.box.height),
            cv::INTER_CUBIC);
        cv::Size kernelSize(3, 3);  // Blur kernel size
        cv::Mat blurredImage;
        cv::blur(crop_mask, blurredImage, kernelSize);
        // Create a grayscale image with size 640x640
        cv::Mat thresholdMat;
        cv::threshold(blurredImage, thresholdMat, 0.5, 255, cv::THRESH_BINARY);
        thresholdMat.convertTo(thresholdMat, CV_8UC1);

        cv::Mat roi2 = resultimage(cv::Rect(int(detection.box.x), int(detection.box.y), int(detection.box.width), int(detection.box.height)));
        std::cout << cv::Rect(int(detection.box.x), int(detection.box.y), int(detection.box.width), int(detection.box.height)) << std::endl;
        thresholdMat.copyTo(roi2);
    }
    end = std::chrono::high_resolution_clock::now();

    // Calculate the time difference
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Output execution time
    std::cout << "Postprocessing time: " << duration.count() << " milliseconds" << std::endl;
    cv::imwrite("mask.png", resultimage);
}

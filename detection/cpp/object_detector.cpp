/**
 * Copyright (c) 2020 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#include <iostream>
#include <chrono>
#include <vector>

#include <tensorflow/lite/delegates/gpu/delegate.h>

#include "object_detector.h"

ObjectDetector::ObjectDetector(const float score_threshold)
    : score_threshold_(score_threshold)
{
}

ObjectDetector::~ObjectDetector()
{
    // Clean up delegate.
    if (delegate_)
    {
        std::cout << "Cleanup GPU delegate." << std::endl;
        TfLiteGpuDelegateV2Delete(delegate_);
    }
}

bool ObjectDetector::BuildInterpreter(
    const std::string& model_path,
    const unsigned int num_of_threads)
{
    auto is_edgetpu = false;
    auto result = false;

    // Split model name and check edge tpu model.
    if (model_path.find("edgetpu") != std::string::npos)
    {
        is_edgetpu = true;
    }

    // Load Model
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (model_ == nullptr)
    {
        std::cerr << "Fail to build FlatBufferModel from file: " << model_path << std::endl;
        return result;
    }

    if (is_edgetpu)
    {
        result = BuildEdgeTpuInterpreterInternal(model_path, num_of_threads);
    }
    else
    {
        result = BuildInterpreterInternal(num_of_threads);
    }

    return result;
}

bool ObjectDetector::BuildInterpreterInternal(
    const unsigned int num_of_threads)
{
    std::cout << "Build TF-Lite Interpreter." << std::endl;
    
    // Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder interpreter_builder(*model_, resolver);

    // Prepare GPU delegate.
    delegate_ = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
    interpreter_builder.AddDelegate(delegate_);

    if (interpreter_builder(&interpreter_) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
        return false;
    }

    // Set Thread option.
    interpreter_->SetNumThreads(num_of_threads);

    // Bind given context with interpreter.
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return false;
    }

    // Get input tensor size.
    const auto& dimensions = interpreter_->tensor(interpreter_->inputs()[0])->dims;
    input_height_ = dimensions->data[1];
    input_width_ = dimensions->data[2];
    input_channels_ = dimensions->data[3];
    input_type_ = interpreter_->tensor(interpreter_->inputs()[0])->type;

    std::cout << "width: " << input_width_ << ", height: " << input_height_ << ", channel: " << input_channels_ << std::endl;

    // Get output tensor
    output_locations_ = interpreter_->tensor(interpreter_->outputs()[0]);
    output_classes_ = interpreter_->tensor(interpreter_->outputs()[1]);
    output_scores_ = interpreter_->tensor(interpreter_->outputs()[2]);
    num_detections_ = interpreter_->tensor(interpreter_->outputs()[3]);

    return true;
}

bool ObjectDetector::BuildEdgeTpuInterpreterInternal(
    std::string model_path,
    const unsigned int num_of_threads)
{
#ifdef ENABEL_EDGETPU_DELEGATE
    std::cout << "Build EdgeTpu Interpreter." << model_path << std::endl;

    //  Create the EdgeTpuContext.
    edgetpu_context_ = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    if (edgetpu_context_ == nullptr)
    {
        std::cerr << "Fail create edge tpu context." << std::endl;
        return false;
    }

    // Build interpreter
    resolver_ = new tflite::ops::builtin::BuiltinOpResolver();
    resolver_->AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    if (tflite::InterpreterBuilder(*model_, *resolver_)(&interpreter_) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
        return false;
    }

    // Bind given context with interpreter.
    interpreter_->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context_.get());

    // Set Thread option.
    interpreter_->SetNumThreads(1);

    // Bind given context with interpreter.
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return false;
    }

    std::cout << "Success AllocateTensors" << std::endl;
    // Get input tensor size.
    const auto& dimensions = interpreter_->tensor(interpreter_->inputs()[0])->dims;
    input_height_ = dimensions->data[1];
    input_width_ = dimensions->data[2];
    input_channels_ = dimensions->data[3];
    input_type_ = interpreter_->tensor(interpreter_->inputs()[0])->type;
 
    input_tensor_shape.resize(dimensions->size);
    for (auto i = 0; i < dimensions->size; i++)
    {
        input_tensor_shape[i] = dimensions->data[i];
        input_array_size *= input_tensor_shape[i];
    }

    std::ostringstream input_string_stream;
    std::copy(input_tensor_shape.begin(), input_tensor_shape.end(), std::ostream_iterator<int>(input_string_stream, " "));

    std::cout << "input shape: " << input_string_stream.str() << std::endl;
    std::cout << "input array size: " << input_array_size << std::endl;

    // Get output tensor
    output_locations_ = interpreter_->tensor(interpreter_->outputs()[0]);
    output_classes_ = interpreter_->tensor(interpreter_->outputs()[1]);
    output_scores_ = interpreter_->tensor(interpreter_->outputs()[2]);
    num_detections_ = interpreter_->tensor(interpreter_->outputs()[3]);

    return true;
#else
    return false;
#endif
}

std::unique_ptr<std::vector<BoundingBox>> ObjectDetector::RunInference(
    const cv::Mat& input,
    std::chrono::duration<double, std::milli>& time_span)
{
    const auto& start_time = std::chrono::steady_clock::now();

    std::vector<float> output_data;
    cv::Mat convert_mat;

    if (input_type_ == kTfLiteFloat32)
    {
        input.convertTo(convert_mat, CV_32FC3);
        // convert_mat /= 255.0;
        convert_mat -= 127.5;
        convert_mat /= 127.5;
        float* input_ptr = interpreter_->typed_input_tensor<float_t>(0);
        std::memcpy(input_ptr, convert_mat.data, convert_mat.total() * convert_mat.elemSize());
    }

    interpreter_->Invoke();

    const float* locations = GetTensorData(*output_locations_);
    const float* classes = GetTensorData(*output_classes_);
    const float* scores = GetTensorData(*output_scores_);
    const int num_detections = (int)*GetTensorData(*num_detections_);
    
    auto results = std::make_unique<std::vector<BoundingBox>>();

    for (auto i = 0; i < num_detections; i++)
    {
        if (scores[i] >= score_threshold_)
        {
            auto bounding_box = std::make_unique<BoundingBox>();
            auto y0 = locations[4 * i + 0];
            auto x0 = locations[4 * i + 1];
            auto y1 = locations[4 * i + 2];
            auto x1 = locations[4 * i + 3];

            
            bounding_box->class_id = (int)classes[i];
            bounding_box->scores = scores[i];
            bounding_box->x = x0;
            bounding_box->y = y0;
            bounding_box->width = x1 - x0;
            bounding_box->height = y1 - y0;
            bounding_box->center_x = bounding_box->x + (bounding_box->width / 2.0f);
            bounding_box->center_y = bounding_box->y + (bounding_box->height / 2.0f);

#if DEBUG
            std::cout << "class_id: " << bounding_box->class_id << std::endl;
            std::cout << "scores  : " << bounding_box->scores << std::endl;
            std::cout << "x       : " << bounding_box->x << std::endl;
            std::cout << "y       : " << bounding_box->y << std::endl;
            std::cout << "width   : " << bounding_box->width << std::endl;
            std::cout << "height  : " << bounding_box->height << std::endl;
            std::cout << "center  : " << bounding_box->center_x << ", " << bounding_box->center_y << std::endl;
            std::cout << "y       : " << bounding_box->y << std::endl;
#endif
            results->emplace_back(std::move(*bounding_box));
        }
    }

    time_span =
        std::chrono::steady_clock::now() - start_time;

    return results;
}

const int ObjectDetector::Width() const
{
    return input_width_;
}

const int ObjectDetector::Height() const
{
    return input_height_;
}

const int ObjectDetector::Channels() const
{
    return input_channels_;
}

const TfLiteType ObjectDetector::InputType() const
{
    return input_type_;
}

float* ObjectDetector::GetTensorData(TfLiteTensor& tensor, const int index)
{
    float* result = nullptr;
    auto nelems = 1;
    for (auto i = 1; i < tensor.dims->size; i++)
    {
        nelems *= tensor.dims->data[i];
    }

    switch (tensor.type)
    {
    case kTfLiteFloat32:
        result = tensor.data.f + nelems * index;
        break;
        std::cerr << "Unmatch tensor type." << std::endl;
    default:
        break;
    }
    return result;
}

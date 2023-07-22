/**
 * Copyright (c) 2021 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#include <iostream>
#include <chrono>
#include <vector>

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include "centernet_detector.h"

CenterNetDetector::CenterNetDetector(const float score_threshold)
    : score_threshold_(score_threshold)
{
}

bool CenterNetDetector::BuildInterpreter(
    const std::string& model_path,
    const unsigned int num_of_threads)
{
    auto result = false;

    // Load Model
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (model_ == nullptr)
    {
        std::cerr << "Fail to build FlatBufferModel from file: " << model_path << std::endl;
        return result;
    }

    result = BuildInterpreterInternal(num_of_threads);

    return result;
}

bool CenterNetDetector::BuildInterpreterInternal(
    const unsigned int num_of_threads)
{
    std::cout << "Build TF-Lite Interpreter." << std::endl;
    
    // Build interpreter
    if (tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_, num_of_threads) != kTfLiteOk)
    {
        std::cerr << "Failed to build interpreter." << std::endl;
        return false;
    }
#if 0
    // Specify the number of threads in the xnnpack option.
    auto xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_options.num_threads = num_of_threads;
    xnnpack_delegate_ = TfLiteXNNPackDelegateCreate(&xnnpack_options);

    if (interpreter_->ModifyGraphWithDelegate(xnnpack_delegate_) != kTfLiteOk)
    {
        std::cerr << "Failed to ModifyGraphWithDelegate." << std::endl;
        return false;
    }
#endif
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
    output_keypoints_ = interpreter_->tensor(interpreter_->outputs()[4]);
    output_keypoints_scores_ = interpreter_->tensor(interpreter_->outputs()[5]);

    return true;
}


std::unique_ptr<std::vector<BoundingBox>> CenterNetDetector::RunInference(
    const unsigned char* const input,
    const size_t in_size,
    std::chrono::duration<double, std::milli>& time_span)
{
    const auto& start_time = std::chrono::steady_clock::now();

    std::vector<float> output_data;

    float* input_ptr = interpreter_->typed_input_tensor<float>(0);
    std::memcpy(input_ptr, input, in_size);
    
    interpreter_->Invoke();

    const float* locations = GetTensorData(*output_locations_);
    const float* classes = GetTensorData(*output_classes_);
    const float* scores = GetTensorData(*output_scores_);
    const int num_detections = (int)*GetTensorData(*num_detections_);
    const float* keypoints = GetTensorData(*output_keypoints_);
    const float* keypoints_scores = GetTensorData(*output_keypoints_scores_);
    
    auto results = std::make_unique<std::vector<BoundingBox>>();

#if DEBUG
    std::cout << "num_detections: " << num_detections <<std::endl;
#endif
    auto index1 = 0;
    auto index2 = 1;
    for (auto i = 0; i < num_detections; i++)
    {
        // std::cout << "score: " << scores[i] <<std::endl;
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
#endif

            for (auto j = 0; j < NUM_KEYPOINTS; j++)
            {
                auto keypoint = std::make_unique<Keypoint>();

                // keypoint->x = keypoints[i * NUM_KEYPOINTS * 2 + j * 2 + 0];
                // keypoint->x = keypoints[i * NUM_KEYPOINTS * 2 + j * 2 + 1];
                keypoint->y = keypoints[index1];
                keypoint->x = keypoints[index2];
                keypoint->scores = keypoints_scores[i * j];
#if DEBUG
                std::cout << "  kepoint: " << j << ", x: " << keypoint->x << ", y: " << keypoint->y << ", scores : " << keypoint->scores << std::endl;
#endif
                bounding_box->keypoints.emplace_back(std::move(*keypoint));
                index1 += 2;
                index2 += 2;
            }

            results->emplace_back(std::move(*bounding_box));
        }
    }

    time_span =
        std::chrono::steady_clock::now() - start_time;

    return results;
}

const int CenterNetDetector::Width() const
{
    return input_width_;
}

const int CenterNetDetector::Height() const
{
    return input_height_;
}

const int CenterNetDetector::Channels() const
{
    return input_channels_;
}

float* CenterNetDetector::GetTensorData(TfLiteTensor& tensor, const int index)
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

auto CenterNetDetector::TensorSize(const TfLiteTensor& tensor)
{
    auto result = 1;

    for (auto i = 0; i < tensor.dims->size; i++)
    {
        result *= tensor.dims->data[i];
	}

    return result;
}


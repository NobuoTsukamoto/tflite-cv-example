/**
 * Copyright (c) 2020 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#ifndef CENTERNET_DETECTOR_H_
#define CENTERNET_DETECTOR_H_

#include <chrono>
#include <memory>
#include <string>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

class Keypoint
{
public:
    float x = 0.0f;
    float y = 0.0f;
    float scores = 0.0f;
};

class BoundingBox
{
public:
    int class_id = 0;
    float scores = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float center_x = 0.0f;
    float center_y = 0.0f;
    std::vector<Keypoint> keypoints;
};

class CenterNetDetector
{
public:
    CenterNetDetector(const float score_threshold);

    bool BuildInterpreter(
        const std::string& model_path,
        const unsigned int num_of_threads = 1);

    std::unique_ptr<std::vector<BoundingBox>> RunInference(
        const unsigned char* const input,
        const size_t in_size,
        std::chrono::duration<double, std::milli>& time_span);

    const int Width() const;
    const int Height() const;
    const int Channels() const;

private:
    static constexpr int NUM_KEYPOINTS = 17;
    
    std::unique_ptr<tflite::FlatBufferModel> model_;
    tflite::ops::builtin::BuiltinOpResolver resolver_;
    std::unique_ptr<tflite::Interpreter> interpreter_;

    TfLiteTensor* output_locations_ = nullptr;
    TfLiteTensor* output_classes_ = nullptr;
    TfLiteTensor* output_scores_ = nullptr;
    TfLiteTensor* num_detections_ = nullptr;
    TfLiteTensor* output_keypoints_ = nullptr;
    TfLiteTensor* output_keypoints_scores_ = nullptr;

    float score_threshold_ = 0.5f;

    int input_width_ = 0;
    int input_height_ = 0;
    int input_channels_ = 0;

    std::vector<int> input_tensor_shape;
    size_t input_array_size = 1;

    bool BuildInterpreterInternal(const unsigned int num_of_threads);

    bool BuildEdgeTpuInterpreterInternal(std::string model_path, const unsigned int num_of_threads);

    float* GetTensorData(TfLiteTensor& tensor, const int index = 0);

    auto TensorSize(const TfLiteTensor& tensor);
};

#endif /* CENTERNET_DETECTOR_H_ */

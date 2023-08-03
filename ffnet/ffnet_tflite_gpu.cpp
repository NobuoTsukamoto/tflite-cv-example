/**
 * Copyright (c) 2023 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#include <iostream>
#include <memory>
#include <string>
#include <stdio.h>
#include <sstream>
#include <chrono>

#include "opencv2/opencv.hpp"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"
#include <tensorflow/lite/delegates/gpu/delegate.h>

#include "utils.h"

std::vector<float> RunInference(const cv::Mat &input,
                                tflite::Interpreter *interpreter,
                                std::chrono::duration<double, std::milli> &time_span)
{
    const auto &start_time = std::chrono::steady_clock::now();

    std::vector<float> output_data;
    cv::Mat convert_mat;
    input.convertTo(convert_mat, CV_32FC3);
    convert_mat -= 127.5;
    convert_mat /= 127.5;
    float *input_ptr = interpreter->typed_input_tensor<float_t>(0);
    std::memcpy(input_ptr, convert_mat.data, convert_mat.total() * convert_mat.elemSize());

    interpreter->Invoke();

    const auto &output_indices = interpreter->outputs();
    const int num_outputs = output_indices.size();
    int out_idx = 0;

    for (int i = 0; i < num_outputs; ++i)
    {
        const auto *out_tensor = interpreter->tensor(output_indices[i]);
        assert(out_tensor != nullptr);
        if (out_tensor->type == kTfLiteInt64)
        {
            const int num_values = out_tensor->bytes / sizeof(int64_t);
            output_data.resize(out_idx + num_values);
            const int64_t *output = interpreter->typed_output_tensor<int64_t>(i);
            for (int j = 0; j < num_values; ++j)
            {
                output_data[out_idx++] = output[j];
            }
        }
        else
        {
            std::cerr << "Tensor " << out_tensor->name
                      << " dose not deeplab  utput type: " << out_tensor->type
                      << std::endl;
        }
    }
    time_span =
        std::chrono::steady_clock::now() - start_time;
    return output_data;
}

int main(int argc, char *argv[])
{
    // Args parser.
    const cv::String keys =
        "{help h ? |    | show help command.}"
        "{m model  |    | path to deeplab tf-lite model flie.}"
        "{v video  |    | video file path.}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    auto model_path = parser.get<std::string>("model");
    auto input_video_path = parser.get<std::string>("video");

    std::cout << "Model  : " << model_path << std::endl;
    std::cout << "Input  : " << input_video_path << std::endl;

    // Window setting
    auto window_name = "FFNet TensorFlow Lite GPU Delegate Demo.";
    cv::namedWindow(window_name,
                    cv::WINDOW_GUI_NORMAL | cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
    cv::moveWindow(window_name, 100, 100);

    // Initialize Colormap.
    auto color_map = CreateCityscapesLabelColormap();

    // Set up InterpreterBuilder.
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (model == nullptr)
    {
        std::cerr << "Fail to build FlatBufferModel from file: " << model_path << std::endl;
        return 0;
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> model_interpreter;
    tflite::InterpreterBuilder interpreter_builder(*model, resolver);

    // Prepare GPU delegate.
    auto *delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
    interpreter_builder.AddDelegate(delegate);

    if (interpreter_builder(&model_interpreter) != kTfLiteOk)
    {
        std::cerr << "Failed to build interpreter." << std::endl;
        return false;
    }

    // Bind given context with interpreter.
    if (model_interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return false;
    }

    // Get input tensor size.
    const auto &dimensions = model_interpreter->tensor(model_interpreter->inputs()[0])->dims;
    auto input_array_size = 1;
    std::vector<int> input_tensor_shape;

    input_tensor_shape.resize(dimensions->size);
    for (auto i = 0; i < dimensions->size; i++)
    {
        input_tensor_shape[i] = dimensions->data[i];
        input_array_size *= input_tensor_shape[i];
    }
    auto input_height = input_tensor_shape[1];
    auto input_width = input_tensor_shape[2];

    std::ostringstream input_string_stream;
    std::copy(input_tensor_shape.begin(), input_tensor_shape.end(), std::ostream_iterator<int>(input_string_stream, " "));

    std::cout << "Input shape: " << input_string_stream.str() << std::endl;
    std::cout << "Input array size: " << input_array_size << std::endl;

    // videocapture setting.
    cv::VideoCapture cap(input_video_path, cv::CAP_FFMPEG);

    std::cout << "Start capture."
              << " isOpened: " << std::boolalpha << cap.isOpened() << std::endl;

    // video capture.
    std::vector<double> inference_fps;
    std::vector<double> fps;
    while (cap.isOpened())
    {
        const auto &start_time = std::chrono::steady_clock::now();

        std::chrono::duration<double, std::milli> inference_time_span;
        cv::Mat frame, input_im, output_im, output_im2;

        cap >> frame;

        // Create input data.
	input_im = frame.clone();
        cv::resize(input_im, input_im, cv::Size(input_width, input_height));
        cv::cvtColor(input_im, input_im, cv::COLOR_BGR2RGB);

        // Run inference.
        const auto &result = RunInference(input_im, model_interpreter.get(), inference_time_span);

        // Create segmantation map.
        cv::Mat seg_im(cv::Size(input_width, input_height), CV_8UC3);
        LabelToColorMap(result, *color_map.get(), seg_im);

        // output tensor size => camera resolution
        cv::resize(frame, frame, cv::Size(seg_im.cols, seg_im.rows));
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        seg_im = (frame / 2) + (seg_im / 2);

        std::chrono::duration<double, std::milli> time_span = std::chrono::steady_clock::now() - start_time;

        // Calc fps and display string.
        std::ostringstream fps_string;

        fps.emplace_back(1000.0 / time_span.count());
        fps_string << std::fixed << std::setprecision(2) << time_span.count() << "ms, FPS: " << fps.back();
        if (fps.size() > 100)
        {
            fps.erase(fps.begin());
            fps_string << " (AVG: " << CalcAverage(fps) << ")";
        }
        DrawCaption(seg_im, cv::Point(10, 30), fps_string.str());

        fps_string.str("");
        inference_fps.emplace_back(1000.0 / inference_time_span.count());

        fps_string << std::fixed << std::setprecision(2) << inference_time_span.count() << "ms, Inference FPS: " << inference_fps.back();
        if (inference_fps.size() > 100)
        {
            inference_fps.erase(inference_fps.begin());
            fps_string << " (AVG: " << CalcAverage(inference_fps) << ")";
        }
        DrawCaption(seg_im, cv::Point(10, 60), fps_string.str());

        // Display image.
        cv::imshow(window_name, seg_im);
        auto key = cv::waitKey(10) & 0xff;
        if (key == 'q')
        {
            break;
        }
    }
    std::cout << "finished capture." << std::endl;

    // Clean up.
    TfLiteGpuDelegateV2Delete(delegate);
    cv::destroyAllWindows();

    return 0;
}

/**
 * Copyright (c) 2019 Nobuo Tsukamoto
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

#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

std::vector<float> RunInference(const std::vector<uint8_t>& input_data,
                                tflite::Interpreter* interpreter,
                                std::chrono::duration<double, std::milli>& time_span)
{
  const auto& start_time = std::chrono::steady_clock::now();

  std::vector<float> output_data;
  uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data.data(), input_data.size());

  interpreter->Invoke();

  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int out_idx = 0;
  for (int i = 0; i < num_outputs; ++i)
  {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8)
    {
      const int num_values = out_tensor->bytes;
      output_data.resize(out_idx + num_values);
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      for (int j = 0; j < num_values; ++j)
      {
        output_data[out_idx++] = (output[j] - out_tensor->params.zero_point) *
                                 out_tensor->params.scale;
      }
    }
    else if (out_tensor->type == kTfLiteFloat32)
    {
      const int num_values = out_tensor->bytes / sizeof(float);
      output_data.resize(out_idx + num_values);
      const float* output = interpreter->typed_output_tensor<float>(i);
      for (int j = 0; j < num_values; ++j)
      {
        output_data[out_idx++] = output[j];
      }
    }
    else if (out_tensor->type == kTfLiteInt64)
    {
      const int num_values = out_tensor->bytes / sizeof(int64_t);
      output_data.resize(out_idx + num_values);
      const int64_t* output = interpreter->typed_output_tensor<int64_t>(i);
      for (int j = 0; j < num_values; ++j) {
        output_data[out_idx++] = output[j];
      }
    }
    else
    {
      std::cerr << "Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
   time_span =
    std::chrono::steady_clock::now() - start_time;
  return output_data;
}

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context,
    const unsigned int num_of_threads = 1)
{
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }
  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(num_of_threads);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  return interpreter;
}

int main(int argc, char *argv[])
{
  // Args parser.
  const cv::String keys =
        "{help h ?     |    | show help command.}"
        "{@input       |    | path to input image file.}"
        "{m model      |    | path to deeplab tf-lite model flie.}"
        "{n thread     |1   | num of thread to set tf-lite interpreter.}"
        "{i iterations |20  | number of inference iterations. }"
        "{d detail     |    | output log for each inference. }";

  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.has("help"))
  {
    parser.printMessage();
    return 0;
  }

  auto image_path = parser.get<std::string>(0);
  auto model_path = parser.get<std::string>("model");
  auto num_of_threads = parser.get<unsigned int>("thread");
  auto num_of_iterations = parser.get<unsigned int>("iterations");
  auto is_show_detail = false;
  if (parser.has("detail"))
  {
    is_show_detail = true;
  }

  std::cout << "image path: " << image_path << std::endl;
  std::cout << "model path: " << model_path << std::endl;
  std::cout << "threads: " << num_of_threads << std::endl;
  std::cout << "iterations: " << num_of_iterations << std::endl;
  std::cout << "detail: " << std::boolalpha << is_show_detail << std::endl;

  // Load model and create the EdgeTpuContext.
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (model == nullptr)
  {
    std::cerr << "Fail to build FlatBufferModel from file: " << model_path << std::endl;
    return 0;
  }
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  if (model == nullptr)
  {
    std::cerr << "Fail create edge tpu context." << std::endl;
    return 0;
  }

  std::unique_ptr<tflite::Interpreter> model_interpreter =
    BuildEdgeTpuInterpreter(*model, edgetpu_context.get(), num_of_threads);

  // Get input tensor size.
  const auto& dimensions = model_interpreter->tensor(model_interpreter->inputs()[0])->dims;
  auto input_array_size = 1;
  std::vector<int> input_tensor_shape;

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

  // read image.
  auto im = cv::imread(image_path);

  // input image resolution  => input_im tensor size (1, 513, 513, 3)
  cv::Mat input_im;

  cv::resize(im, input_im, cv::Size(input_tensor_shape[1], input_tensor_shape[2]));
  cv::cvtColor(input_im, input_im, cv::COLOR_BGR2RGB);
  std::vector<uint8_t> input_data(input_im.data,
                                  input_im.data + (input_im.cols * input_im.rows * input_im.elemSize()));

  // Run inference.
  std::vector<std::chrono::duration<double, std::milli>> inference_times;
  
  for (auto i = 0; i < num_of_iterations; i++)
  {
    std::chrono::duration<double, std::milli> inference_time_span;

    const auto& result = RunInference(input_data, model_interpreter.get(), inference_time_span);
    inference_times.emplace_back(inference_time_span);
    
    if (is_show_detail)
    {
      std::cout << std::setw(4) << std::setfill('0') << i << ": " <<
      std::fixed << std::setprecision(3) << inference_time_span.count() << std::endl;
    }
  }

  auto time_per_inference = 0.0, sum = 0.0;
  for (auto& i : inference_times)
  {
    sum += i.count();
  }
  time_per_inference = sum / num_of_iterations;

  std::cout << std::fixed << std::setprecision(3) << time_per_inference << " ms (iterations = " << num_of_iterations << ")" << std::endl;
  return 0;
}

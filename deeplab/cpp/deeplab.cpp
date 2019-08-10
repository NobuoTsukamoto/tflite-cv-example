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

#include "utils.h"

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
        "{help h ? |    | show help command}"
        "{m model  |    | path to deeplab tf-lite model flie.}"
        "{n thread |1   | num of thread to set tf-lite interpreter.}"
        "{W width  |640 | camera resolution width.}"
        "{H height |480 | camera resolution height.}"
        "{i info   |    | display Inference fps.}";

  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.has("help"))
  {
    parser.printMessage();
    return 0;
  }

  auto model_path = parser.get<std::string>("model");
  auto width = parser.get<int>("width");
  auto height = parser.get<int>("height");
  auto num_of_threads = parser.get<unsigned int>("thread");
  auto is_display_inference_fps = false;
  if (parser.has("info"))
  {
    is_display_inference_fps = true;
  }

  std::cout << "model path: " << model_path << std::endl;
  std::cout << "width: " << width << std::endl;
  std::cout << "height: " << height << std::endl;
  std::cout << "threads: " << num_of_threads << std::endl;
  std::cout << "info: " << std::boolalpha << is_display_inference_fps << std::endl;

  // Window setting
  auto window_name = "Edge TPU Segmantation cpp demo.";

  cv::namedWindow(window_name,
      cv::WINDOW_GUI_NORMAL | cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
  cv::moveWindow(window_name, 100, 100);

  // Initialize Colormap.
  auto color_map = CreatePascalLabelColormap();

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

  // videocapture setting.
  std::ostringstream gst_string_stream;
  /*
  gst_string_stream << "v4l2src device=/dev/video0 ! video/x-raw, "
    "width=" << width << ", height=" << height << ", format=NV12 ! videoscale "
    "! video/x-raw, width=" << width << ", height=" << height <<
    "! videoconvert  ! appsink";
  */
  gst_string_stream << "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
    "width=" << width << ", height=" << height << ", format=(string)NV12, "
    "framerate=(fraction)30/1 ! nvvidconv flip-method=2 !  video/x-raw, "
    "width=(int)" << width << ", height=(int)" << height << ", format=(string)BGRx ! "
    "videoconvert ! appsink";
  cv::VideoCapture cap(gst_string_stream.str());

  // video capture.
  std::vector<double> inference_fps;
  std::vector<double> fps;
  auto is_display_label_im = false;

  std::cout << "Start capture." << std::endl;
  while(cap.isOpened())
  {
    const auto& start_time = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> inference_time_span;
    cv::Mat frame, input_im, output_im, output_im2;

    cap >> frame;

    // Create input data.
    // camera resolution  => input_im tensor size (1, 513, 513, 3)
    cv::resize(frame, input_im, cv::Size(input_tensor_shape[1], input_tensor_shape[2]));
    cv::cvtColor(input_im, input_im, cv::COLOR_BGR2RGB);
    std::vector<uint8_t> input_data(input_im.data,
                                    input_im.data + (input_im.cols * input_im.rows * input_im.elemSize()));

    // Run inference.
    const auto& result = RunInference(input_data, model_interpreter.get(), inference_time_span);

    // Create segmantation map.
    cv::Mat seg_im(cv::Size(input_tensor_shape[1], input_tensor_shape[2]), CV_8UC3);
    LabelToColorMap(result, *color_map.get(), seg_im);

    // output tensor size => camera resolution
    cv::resize(seg_im, seg_im, cv::Size(frame.cols, frame.rows));
    if (!is_display_label_im)
    {
      seg_im = (frame / 2) + (seg_im / 2);
    }

    std::chrono::duration<double, std::milli> time_span = std::chrono::steady_clock::now() - start_time;

    // Calc fps and display string.
    std::ostringstream fps_string;

    fps.emplace_back(1000.0 / time_span.count());
    fps_string << std::fixed << std::setprecision(2) <<
      time_span.count() << "ms, FPS: " << fps.back();
    if (fps.size() > 100)
    {
      fps.erase(fps.begin());
      fps_string << " (AVG: " << CalcAverage(fps) << ")";
    }
    DrawCaption(seg_im, cv::Point(10, 30), fps_string.str());

    if (is_display_inference_fps)
    {
      fps_string.str("");
      inference_fps.emplace_back(1000.0 / inference_time_span.count());

      fps_string << std::fixed << std::setprecision(2) <<
        inference_time_span.count() << "ms, Inference FPS: " << inference_fps.back();
      if (inference_fps.size() > 100)
      {
        inference_fps.erase(inference_fps.begin());
        fps_string << " (AVG: " << CalcAverage(inference_fps) << ")"; 
      }
      DrawCaption(seg_im, cv::Point(10, 60), fps_string.str());
    }

    // Display image.
    cv::imshow(window_name, seg_im);
    auto key = cv::waitKey(10) & 0xff;
    if (key == 'q')
    {
      break;
    }
    else if (key == ' ')
    {
      is_display_label_im = !is_display_label_im;
    }
  }

  cv::destroyAllWindows();

  return 0;
}

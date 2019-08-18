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
#include "deeplab_engine.h"

int main(int argc, char *argv[])
{
  // Args parser.
  const cv::String keys =
        "{help h ? |    | show help command.}"
        "{m model  |    | path to deeplab tf-lite model flie.}"
        "{l label  |15  | index of the target label for motion analysis. (default: Person)} "
        "{n thread |1   | num of thread to set tf-lite interpreter.}"
        "{W width  |640 | camera resolution width.}"
        "{H height |480 | camera resolution height.}"
        "{i info   |    | display Inference fps.}"
        "{s src    |nano| videocapture source. "
        "nano: jetson nano camera, pi: raspberry pi picamera. other: video file path}"
        "{c count  |10  | number of mask image histories.}"
        "{S skip   |0   | number of skipped frames.}";

  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.has("help"))
  {
    parser.printMessage();
    return 0;
  }

  auto model_path = parser.get<std::string>("model");
  auto label_index = parser.get<int>("label");
  auto width = parser.get<unsigned int>("width");
  auto height = parser.get<unsigned int>("height");
  auto num_of_threads = parser.get<unsigned int>("thread");
  auto is_display_inference_fps = false;
  auto video_src = parser.get<std::string>("src");
  auto mask_count = parser.get<unsigned int>("count");
  auto skip_num = parser.get<unsigned int>("skip");
  if (parser.has("info"))
  {
    is_display_inference_fps = true;
  }

  std::cout << "model path: " << model_path << std::endl;
  std::cout << "label: " << label_index << std::endl;
  std::cout << "width: " << width << std::endl;
  std::cout << "height: " << height << std::endl;
  std::cout << "threads: " << num_of_threads << std::endl;
  std::cout << "info: " << std::boolalpha << is_display_inference_fps << std::endl;
  std::cout << "src: " << video_src << std::endl;
  std::cout << "count: " << mask_count << std::endl;
  std::cout << "skip: " << skip_num << std::endl;

  // Window setting
  auto window_name = "Edge TPU Segmantation motion cpp demo.";

  cv::namedWindow(window_name,
      cv::WINDOW_GUI_NORMAL | cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
  cv::moveWindow(window_name, 100, 100);

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
  if (video_src == "nano")
  {
    gst_string_stream << "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
      "width=" << width << ", height=" << height << ", format=(string)NV12, "
      "framerate=(fraction)30/1 ! nvvidconv flip-method=2 !  video/x-raw, "
      "width=(int)" << width << ", height=(int)" << height << ", format=(string)BGRx ! "
      "videoconvert ! appsink";
  }
  else if (video_src == "pi")
  {
    gst_string_stream << "4l2src device=/dev/video0 ! video/x-raw, "
      "width=" << width << ", height=" << height << ", format=(string)NV12 "
      // "framerate=(fraction)30/1 !videoscale ! video/x-raw, "
      " ! videoscale ! video/x-raw, "
      // "width=(int)" << width << ", height=(int)" << height << ", format=(string)BGRx ! "
      "width=(int)" << width << ", height=(int)" << height << " ! "
      "videoconvert ! appsink";
  }
  else
  {
    gst_string_stream << video_src;
  }
  cv::VideoCapture cap(gst_string_stream.str());

  // video capture.
  std::vector<double> inference_fps;
  std::vector<double> fps;
  std::vector<cv::Mat> label_mask_array;
  std::vector<cv::Mat> mask_im_array;
  auto frame_count = 0ul;

  std::cout << "Start capture." << " isOpened: " << std::boolalpha << cap.isOpened() << std::endl;
  while(cap.isOpened())
  {
    const auto& start_time = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> inference_time_span;
    cv::Mat frame, input_im;

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
    cv::Mat output_im = frame.clone();
    cv::cvtColor(output_im, output_im, cv::COLOR_BGR2BGRA);
    cv::Mat label_mask = cv::Mat::zeros(cv::Size(input_tensor_shape[1], input_tensor_shape[2]), CV_8UC1);

    LabelMaskImage(result, label_index, input_im, label_mask);

    // output tensor size => camera resolution
    cv::resize(label_mask, label_mask, cv::Size(frame.cols, frame.rows));

    if ((frame_count % skip_num) == 0)
    {
      label_mask_array.emplace_back(label_mask.clone());
      if (label_mask_array.size() > mask_count)
      {
        label_mask_array.erase(label_mask_array.begin());
      }
      mask_im_array.emplace_back(output_im.clone());
      if (mask_im_array.size() > mask_count)
      {
        mask_im_array.erase(mask_im_array.begin());
      }
    }

    cv::Mat temp = output_im.clone();
    for (size_t i = 0; i < label_mask_array.size(); i++)
    {
      mask_im_array[i].copyTo(output_im, label_mask_array[i]);
      // output_im.copyTo(mask_im_array[i], label_mask_array[i]);
    }
    temp.copyTo(output_im, label_mask);

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
    DrawCaption(output_im, cv::Point(10, 30), fps_string.str());

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
      DrawCaption(output_im, cv::Point(10, 60), fps_string.str());
    }

    // Display image.
    cv::imshow(window_name, output_im);
    auto key = cv::waitKey(10) & 0xff;
    if (key == 'q')
    {
      break;
    }
    frame_count++;
  }
  std::cout << "finished capture." << std::endl;

  cv::destroyAllWindows();

  return 0;
}

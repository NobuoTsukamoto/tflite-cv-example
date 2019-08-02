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

#include "opencv4/opencv2/opencv.hpp"

int main(int argc, char *argv[])
{
  // Args
  const cv::String keys = "{help h ?|false|show help command}"
        "{m model||path to deeplab tf-lite model flie.}"
        "{w width|640|camera resolution width.}"
        "{h height|480|camera resolution height.}";

  cv::CommandLineParser paser(argc, argv, keys);
  if (parser.has("help"))
  {
    parser.printMessage();
    return 0;
  }

  auto model_path = parser.get<std::string>("model");
  auto width = parser.get<int>("width");
  auto height = parser.get<int>("height")

  std::cout << "model path :" << model_path << std::endl;
  std::cout << "width :" << width << std::endl;
  std::cout << "height :" << heigth << std::endl;

  // Window setting
  auto window_name = "Edge TPU Segmantation demo."
  cv::namedWindow(window_name, WINDOW_GUI_NORMAL | WINDOW_AUTOSIZE | WINDOW_KEEPRATIO);
  cv::moveWindow(window_name, 100, 100);

  cv::destroyAllWindows()
  
  return 0;
}
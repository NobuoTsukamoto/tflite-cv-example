/**
 * Copyright (c) 2019 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#include <iostream>
#include <memory>
#include <string>

#include "utils.h"

std::unique_ptr<std::vector<cv::Scalar>> CreatePascalLabelColormap()
{
  std::unique_ptr<std::vector<cv::Scalar>> color_map(new std::vector<cv::Scalar>);
  unsigned char ind[256] = {0};

  // initialize
  for (auto i = 0; i < 256; i++)
  {
    color_map->emplace_back(cv::Scalar(0, 0, 0));
    ind[i] = i;
  }

  // create color map
  for (auto shift = 7; shift >= 0; shift--)
  {
    for (auto& e : *color_map.get())
    {
      auto index {&e - &*color_map.get()->begin()};

      for (auto channel = 0; channel < 3; channel++)
      {
        unsigned char color = (unsigned char)e[channel];
        color |= ((ind[index] >> channel) & 1) << shift;
        e[channel] = (double)color;
      }

    }

    for (auto& i : ind)
    {
      i >>= 3;
    }
  }

  return color_map;
}

void DrawCaption(cv::Mat& im, const cv::Point& point, const std::string& caption)
{
  cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1);
  cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
}

double CalcAverage(const std::vector<double>& array)
{
  auto avg = 0.0;
  for (const auto& e : array)
  {
    avg += e;
  }
  return (avg / array.size());
}

void LabelToColorMap(const std::vector<float>& result,
                     const std::vector<cv::Scalar>& color_map,
                     cv::Mat& seg_im)
{
  for (int y = 0; y < seg_im.rows; y++)
  {
    cv::Vec3b *src = &seg_im.at<cv::Vec3b>(y, 0);
    for (int x = 0; x < seg_im.cols; x++)
    {
      auto label = (int)result[(seg_im.rows * y) + x];
      auto color = color_map.at(label);
      (*src)[0] = color[0];
      (*src)[1] = color[1];
      (*src)[2] = color[2];
      src++;
    }
  }
  cv::cvtColor(seg_im, seg_im, cv::COLOR_RGB2BGR);
}

void LabelMaskImage(const std::vector<float>& result,
                const int input_label,
                const cv::Mat& input_im,
                cv::Mat& mask_im)
{
  for (int y = 0; y < mask_im.rows; y++)
  {
    auto *src = &mask_im.at<unsigned char>(y, 0);
    for (int x = 0; x < mask_im.cols; x++)
    {
      auto label = (int)result[(mask_im.rows * y) + x];
      if (label == input_label)
      {
        *src = 255;
      }
      src++;
    }
  }
}

void RandamMaskImage(const std::vector<float>& result,
                     const int input_label,
                     const cv::Mat& input_im,
                     cv::RNG& rng,
                     cv::Mat& randam_im,
                     cv::Mat& mask_im)
{
  for (int y = 0; y < mask_im.rows; y++)
  {
    auto *mask = &mask_im.at<unsigned char>(y, 0);
    auto *randam = &randam_im.at<cv::Vec4b>(y, 0);
    for (int x = 0; x < mask_im.cols; x++)
    {
      auto label = (int)result[(mask_im.rows * y) + x];
      if (label == input_label)
      {
        *mask = 255;
        (*randam)[0] = rng.uniform(0, 255);
        (*randam)[1] = rng.uniform(0, 255);
        (*randam)[2] = rng.uniform(0, 255);
      }
      mask++;
      randam++;
    }
  }
}

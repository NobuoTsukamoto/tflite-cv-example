/**
 * Copyright (c) 2019 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#ifndef UTILS_H_
#define UTILS_H_

#include "opencv2/opencv.hpp"

extern std::unique_ptr<std::vector<cv::Scalar>> CreatePascalLabelColormap();

extern void DrawCaption(cv::Mat& im, const cv::Point& point, const std::string& caption);

extern double CalcAverage(const std::vector<double>& array);

extern void LabelToColorMap(const std::vector<float>& result,
                     const std::vector<cv::Scalar>& color_map,
                     cv::Mat& seg_im);

extern void LabelMaskImage(const std::vector<float>& result,
                const int input_label,
                const cv::Mat& input_im,
                cv::Mat& mask_im);

extern void RandamMaskImage(const std::vector<float>& result,
                     const int input_label,
                     const cv::Mat& input_im,
                     cv::RNG& rng,
                     cv::Mat& randam_im,
                     cv::Mat& mask_im);
                     
#endif /* UTILS_H_ */
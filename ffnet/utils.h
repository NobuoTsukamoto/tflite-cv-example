/**
 * Copyright (c) 2023 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#ifndef UTILS_H_
#define UTILS_H_

#include "opencv2/opencv.hpp"

extern std::unique_ptr<std::vector<cv::Scalar>> CreateCityscapesLabelColormap();

extern void DrawCaption(cv::Mat& im, const cv::Point& point, const std::string& caption);

extern double CalcAverage(const std::vector<double>& array);

extern void LabelToColorMap(const std::vector<float>& result,
                     const std::vector<cv::Scalar>& color_map,
                     cv::Mat& seg_im);

#endif /* UTILS_H_ */
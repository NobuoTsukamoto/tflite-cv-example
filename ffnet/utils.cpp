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

std::unique_ptr<std::vector<cv::Scalar>> CreateCityscapesLabelColormap()
{
    std::unique_ptr<std::vector<cv::Scalar>> color_map(new std::vector<cv::Scalar>);

    int cityscapes_color_map[][3] = {
    	{128,  64, 128},
        {232,  35, 244},
        { 70,  70,  70},
        {156, 102, 102},
        {153, 153, 190},
        {153, 153, 153},
        { 30, 170, 250},
        {  0, 220, 220},
        { 35, 142, 107},
        {152, 251, 152},
        {180, 130,  70},
        { 60,  20, 220},
        {  0,   0, 255},
        {142,   0,   0},
        { 70,   0,   0},
        {100,  60,   0},
        {100,  80,   0}, 
        {230,   0,   0},
        { 32,  11, 119},
        {  0,   0,   0},

    };

    // initialize
    for (auto i = 0; i < 19; i++)
    {
        color_map->emplace_back(cv::Scalar(cityscapes_color_map[i][0], cityscapes_color_map[i][1], cityscapes_color_map[i][2]));
    }

    return color_map;
}


void DrawCaption(cv::Mat &im, const cv::Point &point, const std::string &caption)
{
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1);
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
}

double CalcAverage(const std::vector<double> &array)
{
    auto avg = 0.0;
    for (const auto &e : array)
    {
        avg += e;
    }
    return (avg / array.size());
}

void LabelToColorMap(const std::vector<float> &result,
                     const std::vector<cv::Scalar> &color_map,
                     cv::Mat &seg_im)
{
    for (auto y = 0; y < seg_im.rows; y++)
    {
        for (auto x = 0; x < seg_im.cols; x++)
        {
            cv::Vec3b *src = &seg_im.at<cv::Vec3b>(y, x);
            auto label = (int)result[(y * seg_im.cols) + x];
            auto color = color_map.at(label);
            (*src)[0] = color[0];
            (*src)[1] = color[1];
            (*src)[2] = color[2];
        }
    }
    cv::cvtColor(seg_im, seg_im, cv::COLOR_RGB2BGR);
}


/**
 * Copyright (c) 2020 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#include <fstream>
#include <iostream>
#include <numeric>

#include <opencv2/opencv.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "centernet_detector.h"

const cv::String kKeys =
    "{help h usage ? |    | show help command.}"
    "{@input         |    | path to centernet keypoint tf-lite model file.}"
    "{n thread       |2   | num of thread to set tf-lite interpreter.}"
    "{s score        |0.5 | score threshold.}"
    "{l label        |    | path to label file.}"
    "{W width        |640 | camera resolution width.}"
    "{H height       |480 | camera resolution height.}"
    "{v videopath    |    | file path of videofile.}"
    ;

const cv::String kWindowName = "CenterNet on-device with TensorFlow Lite.";
const cv::Scalar kWhiteColor = cv::Scalar(246, 250, 250);
const cv::Scalar kBuleColor = cv::Scalar(255, 209, 0);

std::unique_ptr<std::map<long, std::string>> ReadLabelFile(const std::string& label_path)
{
    auto labels = std::make_unique<std::map<long, std::string>>();

    std::ifstream ifs(label_path);
    if (ifs.is_open())
    {
        std::string label = "";
        while (std::getline(ifs, label))
        {
            std::vector<std::string> result;

            boost::algorithm::split(result, label, boost::is_any_of(" ")); // Split by space.
            if (result.size() < 2)
            {
                std::cout << "Expect 2-D input label (" << result.size() << ")." << std::endl;
                continue;
            }
             
            auto label_string = result[2];
            for (size_t i = 3; i < result.size(); i++)
            {
                label_string += " " + result[i];
            }
            auto id = std::stol(result[0]);
            //std::cout << "id: " << id << ", name: " << label_string << ", " << result.size() << std::endl;
            labels->insert(std::make_pair(id, label_string));
        }
    }
    else
    {
        std::cout << "Label file not found. : " << label_path << std::endl;
    }
    return labels;
}

void DrawCaption(
    cv::Mat& im,
    const cv::Point& point,
    const std::string& caption)
{
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
}

void DrawCircle(
    cv::Mat& im,
    const cv::Point& point)
{
    cv::circle(im, point, 7, kWhiteColor, -1);
    cv::circle(im, point, 2, kBuleColor, 2);
}

int main(int argc, char* argv[]) try
{
    // Argument parsing
    cv::String model_path;
    cv::CommandLineParser parser(argc, argv, kKeys);
    if (parser.has("h"))
    {
        parser.printMessage();
        return 0;
    }
    auto num_of_threads = parser.get<unsigned int>("thread");
    auto score_threshold = parser.get<float>("score");
    auto label_path = parser.get<cv::String>("label");
    auto camera_width = parser.get<int>("width");
    auto camera_height = parser.get<int>("height");
    auto video_path = parser.get<cv::String>("videopath");
    if (parser.has("@input"))
    {
        model_path = parser.get<cv::String>("@input");
    }
    else
    {
        std::cout << "No model file path." << std::endl;
        return 0;
    }

    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }

    std::cout << "model path      : " << model_path << std::endl;
    std::cout << "label path      : " << label_path << std::endl;
    std::cout << "threads         : " << num_of_threads << std::endl;
    std::cout << "score threshold : " << score_threshold << std::endl;
    std::cout << "width           : " << camera_width << std::endl;
    std::cout << "height          : " << camera_height << std::endl;
    std::cout << "video path      : " << video_path << std::endl;

    // Create keypoint edges
    typedef std::tuple<int, int> edge;
    std::vector<edge> keypoint_edges{
        std::make_tuple(0, 1),
        std::make_tuple(0, 2),
        std::make_tuple(1, 3),
        std::make_tuple(2, 4),
        std::make_tuple(0, 5),
        std::make_tuple(0, 6),
        std::make_tuple(5, 7),
        std::make_tuple(7, 9),
        std::make_tuple(6, 8),
        std::make_tuple(8, 10),
        std::make_tuple(5, 6),
        std::make_tuple(5, 11),
        std::make_tuple(6, 12),
        std::make_tuple(11, 12),
        std::make_tuple(11, 13),
        std::make_tuple(13, 15),
        std::make_tuple(12, 14),
        std::make_tuple(14, 16)
        };

    // Create CenterNet detector
    auto detector = std::make_unique<CenterNetDetector>(score_threshold);
    detector->BuildInterpreter(model_path, num_of_threads);
    auto width = detector->Width();
    auto height = detector->Height();

    // Load label file
    auto labels = ReadLabelFile(label_path);

    // Get model name
    boost::filesystem::path path(model_path);
    auto model_name = path.filename();

    // Window setting
    cv::namedWindow(kWindowName,
        cv::WINDOW_GUI_NORMAL | cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
    cv::moveWindow(kWindowName, 100, 100);

    // Videocapture setting.
    cv::VideoCapture cap;
    if (video_path.empty())
    {
        cap.open(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, (double)camera_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, (double)camera_height);
    }
    else
    {
        cap.open(video_path);
    }
    auto cap_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    auto cap_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "Start capture." << " isOpened: " << std::boolalpha << cap.isOpened() << std::endl;
    std::cout << "VideoCapture Width: " << cap_width << ", Height: " << cap_height << std::endl;

    std::list<double> inference_times;

    while(cap.isOpened())
    {
        const auto& start_time = std::chrono::steady_clock::now();
    
        cv::Mat frame, resized_im, input_im;

        cap >> frame;

        // Create input data.
        // camera resolution  => resize => bgr2rgb => input_im
        cv::resize(frame, resized_im, cv::Size(width, height));
        cv::cvtColor(resized_im, resized_im, cv::COLOR_BGR2RGB);
        resized_im.convertTo(input_im, CV_32FC3);
        std::vector<float> input_data(input_im.data, input_im.data + (input_im.cols * input_im.rows * input_im.elemSize()));

        // Run inference.
        std::chrono::duration<double, std::milli> inference_time_span;

        const auto& result = detector->RunInference(input_im.data, input_im.total() * input_im.elemSize(), inference_time_span);
        for (const auto& object : *result)
        {
            auto x = int(object.x * cap_width);
            auto y = int(object.y * cap_height);
            auto w = int(object.width * cap_width);
            auto h = int(object.height * cap_height);

            // Draw bounding box
            cv::rectangle(frame, cv::Rect(x, y, w, h), kBuleColor, 2);

            // Draw Caption
            std::ostringstream caption;

            auto it = labels->find(object.class_id);
            if (it != std::end(*labels))
            {
                caption << it->second;
            }
            else
            {
                caption << "ID: " << std::to_string(object.class_id);
            }
            caption << "(" << std::fixed << std::setprecision(2) << object.scores << ")";
            DrawCaption(frame, cv::Point(x, y), caption.str());
            
            // Draw keypoint
            for (const auto & keypoint : object.keypoints)
            {
                if (keypoint.scores >= score_threshold)
                {
                    auto keypoint_x = int(keypoint.x * cap_width);
                    auto keypoint_y = int(keypoint.y * cap_height);
                    DrawCircle(frame, cv::Point(keypoint_x, keypoint_y));
                }
            }

            for (const auto & keypoint_edge : keypoint_edges)
            {
                auto start_point = std::get<0>(keypoint_edge);
                auto end_point = std::get<1>(keypoint_edge);
                auto keypoint_length = object.keypoints.size();
                
                if (start_point < 0 || start_point >= keypoint_length ||
                    end_point < 0 || end_point >= keypoint_length)
                {
                    std::cout << "continue: " << start_point << ", " << end_point << ", " << keypoint_length << std::endl;
                    continue;
                }

                if (object.keypoints[start_point].scores >= score_threshold &&
                    object.keypoints[end_point].scores >= score_threshold)
                {
                    auto start_x = int(object.keypoints[start_point].x * cap_width);
                    auto start_y = int(object.keypoints[start_point].y * cap_height);
                    auto end_x = int(object.keypoints[end_point].x * cap_width);
                    auto end_y = int(object.keypoints[end_point].y * cap_height);
                    cv::line(frame, cv::Point(start_x, start_y), cv::Point(end_x, end_y), kBuleColor, 5);
                }
            }
        }
        // Display model name
        DrawCaption(frame, cv::Point(10, 20), model_name.stem().generic_string());

        // Calc fps and draw fps and inference time.
        std::chrono::duration<double, std::milli> time_span = std::chrono::steady_clock::now() - start_time;
        std::ostringstream time_caption;

        time_caption << std::fixed << std::setprecision(2) << inference_time_span.count() << "ms";

        inference_times.emplace_back(inference_time_span.count());
        if (inference_times.size() > 100)
        {
            inference_times.pop_front();
            double average = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / inference_times.size();

            time_caption << " (avg: " << average << "ms)";
        }
        time_caption << ", " << 1000.0 / time_span.count() << "FPS";
        DrawCaption(frame, cv::Point(10, 60), time_caption.str());

        cv::imshow(kWindowName, frame);
        // Handle the keyboard before moving to the next frame
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q')
        {
            break;  // Escape
        }

    }
    return EXIT_SUCCESS;

}
catch (const cv::Exception& e)
{
    std::cerr << "OpenCV error calling :\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

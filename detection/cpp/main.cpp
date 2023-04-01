/**
 * Copyright (c) 2020 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#include <fstream>
#include <iostream>
#include <map>

#include <opencv2/opencv.hpp>

#include <boost/algorithm/string.hpp>

#include "object_detector.h"

const cv::String kKeys =
    "{help h usage ? |    | show help command.}"
    "{n thread       |2   | num of thread to set tf-lite interpreter.}"
    "{s score        |0.5 | score threshold.}"
    "{l label        |.   | path to label file.}"
    "{@input         |    | path to tf-lite model file.}"
    ;

const cv::String kWindowName = "Object detection example.";
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


    // Create Object detector
    auto detector = std::make_unique<ObjectDetector>(score_threshold);

    detector->BuildInterpreter(model_path, num_of_threads);
    auto width = detector->Width();
    auto height = detector->Height();
    std::cout << "model input :" << height << ", " << width << std::endl;

    // Load label file
    auto labels = ReadLabelFile(label_path);

    // Window setting
    cv::namedWindow(kWindowName,
        cv::WINDOW_GUI_NORMAL | cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
    cv::moveWindow(kWindowName, 100, 100);

    // Videocapture setting.
    cv::VideoCapture cap(4);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    // auto cap_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    // auto cap_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto cap_width = 640;
    auto cap_height = 480;

    std::cout << "Start capture (" << cap_width << "," << cap_height << ")" << " isOpened: " << std::boolalpha << cap.isOpened() << std::endl;

    while(cap.isOpened())
    {
        const auto& start_time = std::chrono::steady_clock::now();
    
        cv::Mat frame, resized_im, input_im;

        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        // Create input data.
        // camera resolution  => resize => bgr2rgb => input_im
        cv::resize(frame, resized_im, cv::Size(width, height));
        cv::cvtColor(resized_im, resized_im, cv::COLOR_BGR2RGB);
        resized_im.convertTo(input_im, CV_32FC3);
        // std::vector<float> input_data(input_im.data, input_im.data + (input_im.cols * input_im.rows * input_im.elemSize()));

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
        }

        // Calc fps and draw fps and inference time.
        std::chrono::duration<double, std::milli> time_span = std::chrono::steady_clock::now() - start_time;
        std::ostringstream time_caption;

        time_caption << std::fixed << std::setprecision(2) << inference_time_span.count() << " ms, " << 1000.0 / time_span.count() << "FPS";
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

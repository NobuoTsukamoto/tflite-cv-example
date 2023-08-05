/**
 * Copyright (c) 2020 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <numeric>

#include <opencv2/opencv.hpp>

#include <boost/algorithm/string.hpp>

#include "object_detector.h"

const cv::String kKeys =
    "{help h usage ? |    | show help command.}"
    "{n thread       |2   | num of thread to set tf-lite interpreter.}"
    "{s score        |0.5 | score threshold.}"
    "{l label        |.   | path to label file.}"
    "{o output       |    | file path of output videofile.}"
    "{v videopath    |    | file path of videofile.}"
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
    auto video_path = parser.get<cv::String>("videopath");
    auto output_path = parser.get<cv::String>("output");
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
    std::cout << "video path      : " << video_path << std::endl;
    std::cout << "output path     : " << output_path << std::endl;

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
    cv::VideoCapture cap;
     if (video_path.empty())
    {
        cap.open(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, (double)640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, (double)480);
    }
    else
    {
        cap.open(video_path);
    }
    auto cap_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    auto cap_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto cap_fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Start capture."
              << " isOpened: " << std::boolalpha << cap.isOpened() << std::endl;
    std::cout << "VideoCapture Width: " << cap_width << ", Height: " << cap_height << ", FPS: " << cap_fps << std::endl;

    // Videowriter setting.
    cv::VideoWriter writer;
    if (!output_path.empty())
    {
        auto fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
        writer.open(output_path, fourcc, cap_fps, cv::Size(cap_width, cap_height), true);
    }

    std::list<double> inference_times;
    
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
        cv::cvtColor(resized_im, input_im, cv::COLOR_BGR2RGB);

        // Run inference.
        std::chrono::duration<double, std::milli> inference_time_span;

        const auto& result = detector->RunInference(input_im, inference_time_span);

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

        // Output file.
        if (writer.isOpened())
        {
            writer.write(frame);
        }

        cv::imshow(kWindowName, frame);
        // Handle the keyboard before moving to the next frame
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q')
        {
            break;  // Escape
        }

    }

    // Clean up.
    writer.release();
    cv::destroyAllWindows();

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

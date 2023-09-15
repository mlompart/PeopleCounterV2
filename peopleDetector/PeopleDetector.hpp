#pragma once

#include "Detection.hpp"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace peopleDetector {

    class PeopleDetector {
    public:
        PeopleDetector(const std::string &modelPath, const std::string &classListPath);

        Detections detect(const cv::Mat &frame);

        uint64_t getPerfProfile(std::vector<double> layersTimes);


    private:
        std::vector<cv::Mat> preProcess(const cv::Mat &inputImage);

        Detections postProcess(const cv::Mat &inputImage, std::vector<cv::Mat> &outputs);
        Detections postProcessYolo8(const cv::Mat &inputImage, std::vector<cv::Mat> &outputs);
        cv::dnn::Net net;
        std::vector<std::string> classList;

    };

} // peopleDetector

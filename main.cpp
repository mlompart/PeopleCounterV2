#include "peopleDetector/PeopleDetector.hpp"
#include "peopleTracker/MultipleObjectTracker.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>


int main() {
    // Load video.
    cv::VideoCapture videoInput;

    videoInput.open("rtsp://admin@192.168.1.108:554", cv::CAP_FFMPEG);

    if (!videoInput.isOpened()) {
        std::cout << " Video is not open!!!\n";
        return 1;
    }

    cv::Mat frame{};
    peopleTracker::MultipleObjectTracker tracker{};
    peopleDetector::PeopleDetector detector("../model/yolov5x.onnx", "../model/coco.names.txt");

    while (true) {
        if (not videoInput.read(frame)) {
            std::cout << "Capture read error" << std::endl;
            return -1;
        }
        cv::resize(frame, frame, {640, 640});

        Detections detections = detector.detect(frame);
        tracker.trackDetections(frame, detections);

        char esc = cv::waitKey(5);
        if (esc == 27) break;
    }
    cv::destroyAllWindows();
    return 0;
}
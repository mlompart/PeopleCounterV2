#include "peopleDetector/PeopleDetector.hpp"
#include "peopleTracker/MultipleObjectTracker.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>


int main() {
    cv::VideoCapture videoInput;

    videoInput.open("../samples/multi-person-part2.mp4", cv::CAP_FFMPEG);

    if (!videoInput.isOpened()) {
        std::cout << " Video is not open!!!\n";
        return 1;
    }

    cv::Mat frame{};
    peopleTracker::MultipleObjectTracker tracker{"../peopleReId/config/config.yaml"};
    peopleDetector::PeopleDetector detector("../models/yolov5s.onnx", "../models/coco.names.txt");

    while (true) {
        if (not videoInput.read(frame)) {
            std::cout << "Capture read error" << std::endl;
            return -1;
        }
        cv::resize(frame, frame, {640, 640});

        Detections detections = detector.detect(frame);
        tracker.trackDetections(frame, detections);
        cv::imshow("PeopleDetector", frame);
        char esc = cv::waitKey(20);
        if (esc == 27) break;
    }
    cv::destroyAllWindows();
    return 0;
}
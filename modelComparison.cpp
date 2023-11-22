#include "peopleDetector/PeopleDetector.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <future>
#include <fstream>

static bool shouldTerminate{false};
[[noreturn]] void saveDetections(std::atomic<std::uint32_t>& totalDetections, std::vector<uint32_t>& detectionsHistory) {
    while (not shouldTerminate) {
        detectionsHistory.push_back(static_cast<uint32_t>(totalDetections));
        std::cout << "Aktualnie detekcji jest: "<< totalDetections << " dla wątku:" << std::this_thread::get_id() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main() {
    // Load video.
    cv::VideoCapture videoInput;

    videoInput.open("/home/mlompart/Filmy/fragmenty_testowe/multi-person-part1.mp4");

    if (!videoInput.isOpened()) {
        std::cout << " Video is not open!!!\n";
        return 1;
    }

    cv::Mat frame{};
    peopleDetector::PeopleDetector model1("../models/yolov5sBeforeTraining.onnx", "../models/coco.names.txt");
    peopleDetector::PeopleDetector model2("../models/yolov5sAfterTraining.onnx", "../models/coco.names.txt");

    std::atomic<uint32_t> totalDetections1(0), totalDetections2(0);
    std::vector<uint32_t> detectionsHistory1, detectionsHistory2;

    // Uruchom wątek zapisujący detekcje
    std::future<void> savingThread1 = std::async(std::launch::async, saveDetections, std::ref(totalDetections1), std::ref(detectionsHistory1));
    std::future<void> savingThread2 = std::async(std::launch::async, saveDetections, std::ref(totalDetections2), std::ref(detectionsHistory2));

    while (true) {
        if (not videoInput.read(frame)) {
            std::cout << "Capture read error" << std::endl;
            break;
        }
        cv::resize(frame, frame, {640, 640});
        Detections detections1 = model1.detect(frame);
        totalDetections1 += detections1.size();
        for (auto &detection: detections1) { cv::rectangle(frame, detection.getBoundingBox(), {255, 0, 0}, 3); };
        Detections detections2 = model2.detect(frame);
        totalDetections2 += detections2.size();
        for (auto &detection: detections2) { cv::rectangle(frame, detection.getBoundingBox(), {0, 0, 255}, 3); };
        cv::imshow("modelComparison", frame);
        char esc = cv::waitKey(5);
        if (esc == 27) break;
    }
    shouldTerminate = true;
    savingThread1.wait();
    savingThread2.wait();
    cv::destroyAllWindows();
    std::ofstream outputFile1("../beforeTraining.txt");
    std::ofstream outputFile2("../afterTraining.txt");

    if (outputFile1.is_open() and outputFile2.is_open()) {
        for (const auto& value : detectionsHistory1) {
            outputFile1 << value << "\n";
        }
        for (const auto& value : detectionsHistory2) {
            outputFile2 << value << "\n";
        }
        outputFile1.close();
        outputFile2.close();

        std::cout << "Dane zostały pomyślnie zapisane do pliku." << std::endl;
    } else {
        std::cerr << "Błąd podczas otwierania pliku do zapisu." << std::endl;
    }
    return 0;
}
#include "PeopleDetector.hpp"
#include "Constants.hpp"
#include <fstream>
#include <iostream>


namespace peopleDetector {
    PeopleDetector::PeopleDetector(const std::string &modelPath, const std::string &classListPath) {

        std::ifstream ifs(classListPath);
        std::string line;
        while (std::getline(ifs, line)) { classList.push_back(line); }

        net = cv::dnn::readNet(modelPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }

    uint64_t PeopleDetector::getPerfProfile(std::vector<double> layersTimes) { return net.getPerfProfile(layersTimes); }

    Detections PeopleDetector::detect(const cv::Mat &frame) {
        auto output = preProcess(frame);
        return postProcess(frame, output);
    }

    std::vector<cv::Mat> PeopleDetector::preProcess(const cv::Mat &inputImage) {
        // Convert to blob.
        cv::Mat blob;
        cv::dnn::blobFromImage(inputImage, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

        net.setInput(blob);

        // Forward propagate.
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        return outputs;
    }

    Detections PeopleDetector::postProcess(const cv::Mat &inputImage, std::vector<cv::Mat> &outputs) {

        // Initialize vectors to hold respective outputs while unwrapping     detections.
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        // Resizing factor.
        float x_factor = inputImage.cols / INPUT_WIDTH;
        float y_factor = inputImage.rows / INPUT_HEIGHT;
        auto *data = reinterpret_cast<float *>(outputs[0].data);
        const int dimensions = 85;
        // 25200 for default size 640.
        const int rows = 25200;
        // Iterate through 25200 detections.
        for (int i = 0; i < rows; ++i) {
            float confidence = data[4];
            // Discard bad detections and continue.
            if (confidence >= CONFIDENCE_THRESHOLD) {
                float *classes_scores = data + 5;
                // Create a 1x85 Mat and store class scores of 80 classes.
                cv::Mat scores(1, classList.size(), CV_32FC1, classes_scores);
                // Perform minMaxLoc and acquire the index of best class  score.
                cv::Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                // Continue if the class score is above the threshold.
                if (max_class_score > SCORE_THRESHOLD) {
                    // Store class ID and confidence in the pre-defined respective vectors.
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
                    // Center.
                    float cx = data[0];
                    float cy = data[1];
                    // Box dimension.
                    float w = data[2];
                    float h = data[3];
                    // Bounding box coordinates.
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    // Store good detections in the boxes vector.
                    boxes.emplace_back(left, top, width, height);
                }
            }
            // Jump to the next row.
            data += 85;
        }
        // Perform Non-Maximum Suppression and draw predictions.
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
        Detections detects;
        for (int i = 0; i < indices.size(); i++) {

            int idx = indices[i];
            if (classList[class_ids[idx]] == "person") {
                Detection detection(boxes[idx], confidences[idx], classList[class_ids[idx]]);
                std::cout << "PeopleDetector::postProcess detect: " << detection.getClassName() << " confidence: " << confidences[idx]
                          << std::endl;
                detects.push_back(detection);
            }
        }
        return detects;
    }
}// namespace peopleDetector

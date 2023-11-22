#include "MultipleObjectTracker.hpp"
#include "Constants.hpp"
#include <functional>
#include <iostream>
#include <opencv2/core.hpp>

namespace peopleTracker {
    MultipleObjectTracker::MultipleObjectTracker(const std::string &configPath) {
        YAML::Node root = YAML::LoadFile(configPath);
        peopleReId = std::make_unique<peopleReId::Fastreid>(root["fastreid"]);
        peopleReId->LoadEngine();
        if (peopleReId) {
            std::cout << "FastReId model loaded successfully!" << std::endl;
        } else {
            std::cout << "FastReId model not loaded" << std::endl;
        }
    }

    void MultipleObjectTracker::trackDetections(cv::Mat &frame, const Detections &detections) {

        if (not detections.empty()) {
            for (auto &detection: detections) {

                std::cout << "MultipleObjectTracker::trackDetectionsDetection Check detection, Detection area: " << detection.getArea()
                          << std::endl;
                if (detection.getArea() < MIN_DETECTION_AREA) { continue; }
                auto trackedObject = std::find_if(trackedObjects.begin(), trackedObjects.end(), [this, detection](auto &trackedObject) {
                    return /*isIntersectionValid(detection.getBoundingBox(), trackedObject->getTrackedWindow())*/
                            not trackedObject.second->isReleased() and not trackedObject.second->isInside() and
                            isDistanceValid(detection.getBoundingBox(), trackedObject.second->getTrackedWindow(),
                                            trackedObject.second->getId());
                });
                if (trackedObject not_eq trackedObjects.end()) {
                    if (detection.getArea() < MAX_DETECTION_AREA || detection.getArea() > MIN_DETECTION_AREA) {
                        trackedObject->second->setTrackedWindow(detection.getBoundingBox());
                    }
                } else {
                    if (not isObjectAtFrameEdge(frame, detection) and
                        (detection.getArea() < MAX_DETECTION_AREA || detection.getArea() > MIN_DETECTION_AREA)) {
                        initNewTrackedObject(frame, detection);
                    }
                }
            }
        }
        updateTrackedObjects(frame);
        drawObjectInsideInfo(frame);
    }

    void MultipleObjectTracker::updateTrackedObjects(cv::Mat &frame) {
        if (not trackedObjects.empty()) {
            for (auto &trackedObject: this->trackedObjects) {

                switch (trackedObject.second->getStatus()) {
                    case StatusT::RELEASED:
                    case StatusT::INSIDE:
                        continue;
                    case StatusT::ENTERING:
                    case StatusT::EXITING:
                        trackedObject.second->update(frame);
                }
            }
            cleanUpReleasedObjects();
        }
    }

    bool MultipleObjectTracker::isDistanceValid(const cv::Rect newBox, const cv::Rect oldBox, const uint32_t id) {
        cv::Point oldCenter(oldBox.x + (oldBox.width / 2), oldBox.y + (oldBox.height / 2));
        cv::Point newCenter(newBox.x + (newBox.width / 2), newBox.y + (newBox.height / 2));

        double distance = std::sqrt(std::pow(newCenter.x - oldCenter.x, 2) + std::pow(newCenter.y - oldCenter.y, 2));
        bool isValid = distance <= DISTANCE_THRESHOLD;
        if (isValid) {
            std::cout << "MultipleObjectTracker::isDistanceValid Distance: " << distance << " is valid for id: " << id << std::endl;
        } else {
            std::cout << "MultipleObjectTracker::isDistanceValid Distance = " << distance << " is invalid for id: " << id << std::endl;
        }
        return isValid;
    }

    bool MultipleObjectTracker::isIntersectionValid(cv::Rect newBox, cv::Rect oldBox) {
        float validIntersection = newBox.area() * IOU_RATIO;
        uint intersection = (newBox & oldBox).area();

        printf("MultipleObjectTracker::isIntersectionValid Valid intersection is %f, object has %d\n", validIntersection, intersection);

        return intersection > static_cast<int>(validIntersection);
    }

    void MultipleObjectTracker::initNewTrackedObject(cv::Mat &frame, const Detection &detection) {
        printf("MultipleObjectTracker::initNewTrackedObject Init new objects to track\n");
        const auto &roi = detection.getBoundingBox();
        if (0 <= detection.getBoundingBox().x && 0 <= roi.width && roi.x + roi.width <= frame.cols && 0 <= roi.y && 0 <= roi.height &&
            roi.y + roi.height <= frame.rows) {

            auto status = recognizeDirection(detection.getBoundingBox());

            if (status == StatusT::EXITING and isAnyObjectInside()) {

                auto feature = calculateDetectionFeature(frame, detection);
                auto key = recognizeExitingObjectByFeature(feature);
                auto id = trackedObjects[key]->getId();
                trackedObjects.erase(key);
                trackedObjects[key] = std::make_unique<ObjectTracker>(frame, detection.getBoundingBox(), id, status, feature);
            } else {
                auto feature = calculateDetectionFeature(frame, detection);
                this->trackedObjects[nextId] =
                        std::make_unique<ObjectTracker>(frame, detection.getBoundingBox(), nextId++, status, feature);
            }
        }
    }

    peopleReId::FeatureRes MultipleObjectTracker::calculateDetectionFeature(const cv::Mat &frame, const Detection &detection) {
        cv::Mat objectImage = frame(detection.getBoundingBox());
        return peopleReId->inferenceDetections(objectImage);
    }
    StatusT MultipleObjectTracker::recognizeDirection(const cv::Rect &trackedWin) {
        StatusT status{};
        if (trackedWin.x + trackedWin.width / 2 > static_cast<uint>(INPUT_WIDTH / 2)) {
            status = StatusT::ENTERING;
        } else {
            status = StatusT::EXITING;
        }
        return status;
    }
    bool MultipleObjectTracker::isObjectAtFrameEdge(cv::Mat &mat, const Detection &detection) {
        return detection.left() < 5 or detection.left() + detection.width() > mat.cols - 5;
    }

    void MultipleObjectTracker::cleanUpReleasedObjects() {
        for (auto it = trackedObjects.begin(); it != trackedObjects.end();) {
            if (it->second->isReleased()) {
                it = trackedObjects.erase(it);
            } else {
                ++it;
            }
        }
    }

    cv::Mat MultipleObjectTracker::calculateDetectionHistogram(cv::Mat &frame, const cv::Rect rect) {
        auto roi = frame(rect);
        cv::Mat hsv_roi;
        cv::Mat mask;
        cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
        cv::inRange(hsv_roi, cv::Scalar(0, 60, 32), cv::Scalar(180, 255, 255), mask);
        float range[2] = {0, 180};
        const float *range_[] = {range};
        int histSize[1];
        int channels[1];
        histSize[0] = {180};
        channels[0] = {0};
        cv::Mat hist;
        cv::calcHist(&hsv_roi, 1, channels, mask, hist, 1, histSize, range_);
        cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);
        return hist;
    }

    uint32_t MultipleObjectTracker::recognizeExitingObjectByFeature(const peopleReId::FeatureRes &detFeature) {
        std::unordered_map<uint32_t, float> similarities;
        for (auto &object: trackedObjects) {
            if (object.second->getStatus() == StatusT::INSIDE) {
                similarities[object.first] = peopleReId->computeSimilarity(detFeature, object.second->getFeature());
            }
        }
        if (not similarities.empty()) std::cout << "Similarities: ";
        for (auto &similarity: similarities) { std::cout << similarity.second << " "; }
        std::cout << std::endl;
        auto maxSimilarityPair = std::max_element(
                similarities.begin(), similarities.end(),
                [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b) { return a.second > b.second; });

        if (maxSimilarityPair != similarities.end()) {
            return maxSimilarityPair->first;
        } else {
        }
        return 0;
    }

    uint32_t MultipleObjectTracker::recognizeExitingObjectByDetectHist(const cv::Mat &detectHist) {
        int comparisonMethod = cv::HISTCMP_CHISQR;
        std::unordered_map<uint32_t, double> similarities;
        for (auto &object: trackedObjects) {
            if (object.second->getStatus() == StatusT::INSIDE) {
                similarities[object.first] = cv::compareHist(detectHist, object.second->getHistogram(), comparisonMethod);
            }
        }
        if (not similarities.empty()) std::cout << "Similarities: ";
        for (auto &similarity: similarities) { std::cout << similarity.second << " "; }
        std::cout << std::endl;
        auto maxSimilarityPair = std::min_element(
                similarities.begin(), similarities.end(),
                [](const std::pair<uint32_t, double> &a, const std::pair<uint32_t, double> &b) { return a.second < b.second; });

        if (maxSimilarityPair != similarities.end()) {
            uint32_t bestIndex = maxSimilarityPair->first;
            double bestSimilarity = maxSimilarityPair->second;
            return maxSimilarityPair->first;
        } else {
        }
        return 255;
    }
    bool MultipleObjectTracker::isAnyObjectInside() {
        return std::any_of(trackedObjects.begin(), trackedObjects.end(),
                           [](auto &object) { return object.second->getStatus() == StatusT::INSIDE; });
    }
    void MultipleObjectTracker::drawObjectInsideInfo(cv::Mat &frame) {
        // Display the label at the top of the bounding box.
        int baseLine;
        auto amountOfObjects = getAmountOfInsideObject();
        std::string plularOrSingular = (amountOfObjects == 1) ? " czlowiek." : " ludzi.";
        std::string label{"W budynku jest obecnie: " + std::to_string(amountOfObjects) + plularOrSingular};
        cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
        auto top = cv::max(20, label_size.height);
        // Top left corner.
        cv::Point tlc = cv::Point(0, 0);
        // Bottom right corner.
        cv::Point brc = cv::Point(frame.rows, top + label_size.height + 2);
        // Draw white rectangle.
        rectangle(frame, tlc, brc, BLACK_COLOUR, cv::FILLED);
        // Put the label on the black rectangle.
        putText(frame, label, cv::Point(0, top + label_size.height), FONT_FACE, 0.8, WHITE_COLOUR, 1);
        putText(frame, {"Numer klatki: " + std::to_string(frameNr)}, cv::Point(0, 600), FONT_FACE, FONT_SCALE, {244, 5, 5}, 1, 2);
        frameNr++;
    }

    uint32_t MultipleObjectTracker::getAmountOfInsideObject() const {
        return std::count_if(trackedObjects.begin(), trackedObjects.end(),
                             [](auto &trackedObject) { return trackedObject.second->getStatus() == StatusT::INSIDE; });
    }

}// namespace peopleTracker
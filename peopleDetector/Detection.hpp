#pragma once
#include <opencv2/core.hpp>
#include <utility>

class Detection
{
public:
    Detection(cv::Rect boundingBox, float confidence, std::string className) : box{boundingBox}, confidence{confidence}, className{std::move(className)}{};
    [[nodiscard]] inline cv::Rect getBoundingBox() const { return box;}
    [[nodiscard]] inline uint32_t left() const { return box.x;}
    [[nodiscard]] inline uint32_t top() const { return box.y;}
    [[nodiscard]] inline uint32_t width() const { return box.width;}
    [[nodiscard]] inline uint32_t height() const { return box.height;}
    [[nodiscard]] inline uint32_t getArea() const { return box.area();}
    [[nodiscard]] inline std::string getClassName() const { return className;}
    [[nodiscard]] inline float getConfidence() const { return confidence;}

private:
    cv::Rect box;
    float confidence;
    std::string className;
};
using Detections = std::vector<Detection>;

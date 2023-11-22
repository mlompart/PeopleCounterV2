#pragma once
#include <cstdint>
#include "opencv2/core.hpp"
// Text parameters.
extern const float FONT_SCALE;
extern const int FONT_FACE;
extern const int THICKNESS;

// Colors.
extern cv::Scalar BLACK_COLOUR;
extern cv::Scalar BLUE_COLOUR;
extern cv::Scalar YELLOW_COLOUR;
extern cv::Scalar RED_COLOUR;
extern cv::Scalar WHITE_COLOUR;

// Track Params
constexpr float IOU_RATIO = 0.10;
constexpr uint16_t DISTANCE_THRESHOLD = 80;
constexpr uint16_t AMOUNT_REPEAT = 155;
constexpr uint16_t MIN_DETECTION_AREA = 35000;
constexpr uint32_t MAX_DETECTION_AREA = 250000;

// Detect Params
constexpr uint INPUT_WIDTH = 640;
constexpr uint INPUT_HEIGHT = 640;
constexpr float SCORE_THRESHOLD = 0.5;
constexpr float NMS_THRESHOLD = 0.5;
constexpr float CONFIDENCE_THRESHOLD = 0.5;
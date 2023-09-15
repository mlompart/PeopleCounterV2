#include "ObjectTracker.hpp"
#include "Constants.hpp"
#include <iostream>

namespace peopleTracker {
    ObjectTracker::ObjectTracker(cv::Mat &frame, cv::Rect trackedWin, const uint32_t objectId, const StatusT status)
        : status(status), id(objectId), trackedWindow(trackedWin), amountOfRepeat(0) {
        roi = frame(trackedWindow);

        calcHist(frame);
        std::cout << "ObjectTracker::ObjectTracker Start tracking object id = " << this->id << std::endl;
    }

    void ObjectTracker::update(cv::Mat frame) {
        calcHist(frame);
        cv::Mat hsv, dst;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        const float *range_[] = {range};

        std::cout << "ObjectTracker::update updating object id: " << id << std::endl;
        cv::calcBackProject(&hsv, 1, channels, roi_hist, dst, range_);
        // apply meanshift to get the new location
        cv::meanShift(dst, trackedWindow, term_crit);
//        cv::CamShift(dst, trackedWindow, term_crit);
        drawBoundingBoxesAndLabels(frame);
        checkBoundaries(frame);
    }

    void ObjectTracker::release() {
        this->status = StatusT::RELEASED;
        std::cout << "ObjectTracker::release() releasing object id = " << this->id << std::endl;
    }

    StatusT ObjectTracker::getStatus() { return this->status; }

    void ObjectTracker::setStatus(const StatusT newStatus) { this->status = newStatus; }

    void ObjectTracker::drawObjectLabel(const cv::Mat &input_image, const std::string &label, int left, int top) {
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
        top = cv::max(top, label_size.height);
        cv::Point tlc = cv::Point(left, top);
        cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
        rectangle(input_image, tlc, brc, BLACK_COLOUR, cv::FILLED);
        putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW_COLOUR, THICKNESS);
    }
    void ObjectTracker::checkBoundaries(const cv::Mat &frame) {
        switch (this->status) {
            case StatusT::RELEASED:
            case StatusT::INSIDE:
                break;
            case StatusT::ENTERING:
                if (trackedWindow.x == 0) {
                    std::cout << "ObjectTracker::checkBoundaries() trackWindow == 0 for id: " << this->getId() << std::endl;
                    if (this->amountOfRepeat < AMOUNT_REPEAT - 25) { this->amountOfRepeat = AMOUNT_REPEAT - 25; }
                }
                if (++amountOfRepeat >= AMOUNT_REPEAT) {
                    std::cout << "ObjectTracker::checkBoundaries() amountRepeat: " << amountOfRepeat << " for id: " << getId() << std::endl;
                    setStatus(StatusT::INSIDE);
                }
                break;
            case StatusT::EXITING:
                if (trackedWindow.x + trackedWindow.width >= frame.cols - 1) {
                    std::cout << "ObjectTracker::checkBoundaries() trackWindow == " << INPUT_WIDTH << " for id: " << this->getId()
                              << std::endl;
                    if (this->amountOfRepeat < AMOUNT_REPEAT - 25) { this->amountOfRepeat = AMOUNT_REPEAT - 25; }
                }
                if (++amountOfRepeat >= AMOUNT_REPEAT) {
                    std::cout << "ObjectTracker::checkBoundaries() amountRepeat: " << amountOfRepeat << " for id: " << getId() << std::endl;
                    release();
                }
                break;
        }
    }
    void ObjectTracker::drawBoundingBoxesAndLabels(const cv::Mat &frame) {
        cv::rectangle(frame, trackedWindow, BLUE_COLOUR, THICKNESS);
        drawObjectLabel(frame, "ID=" + std::to_string(id), trackedWindow.x, trackedWindow.y);
    }
    void ObjectTracker::calcHist(const cv::Mat& frame)
    {
        cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
        cv::inRange(hsv_roi, cv::Scalar(0, 60, 32), cv::Scalar(180, 255, 255), mask);

        const float *range_[] = {range};
        histSize[0] = {180};
        channels[0] = {0};
        cv::calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range_);
        cv::normalize(roi_hist, roi_hist, 0, 255, cv::NORM_MINMAX);
        term_crit = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1);
    }
}// namespace peopleTracker
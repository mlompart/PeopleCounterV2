#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include "../peopleReId/fast-reid.hpp"

namespace peopleTracker {
    enum class StatusT { RELEASED, ENTERING, INSIDE, EXITING };

    class ObjectTracker {
    public:
        ObjectTracker(cv::Mat &frame, cv::Rect trackedWindow, uint32_t objectId, StatusT status, peopleReId::FeatureRes  featureRes);

        ~ObjectTracker() = default;

        void update(cv::Mat frame);
        [[nodiscard]] inline cv::Rect getTrackedWindow() const { return this->trackedWindow; }
        inline void setTrackedWindow(const cv::Rect newWindow) { this->trackedWindow = newWindow; }
        StatusT getStatus();
        [[nodiscard]] cv::Mat getHistogram() const { return roi_hist; };
        [[nodiscard]] peopleReId::FeatureRes getFeature() const { return feature; };
        [[nodiscard]] inline uint32_t getId() const { return id; };
        void setStatus(StatusT newStatus);
        void release();
        inline bool isReleased() { return this->getStatus() == StatusT::RELEASED; }
        inline bool isInside() { return this->getStatus() == StatusT::INSIDE; }
        void calcHist(const cv::Mat& frame);

    private:
        void drawBoundingBoxesAndLabels(const cv::Mat &frame);
        static void drawObjectLabel(const cv::Mat &input_image, const std::string &label, int left, int top);
        void checkBoundaries(const cv::Mat &frame);

        cv::Mat roi;
        cv::Mat hsv_roi;
        cv::Mat mask;
        cv::Rect trackedWindow;
        cv::Mat roi_hist;
        cv::TermCriteria term_crit;
        uint16_t amountOfRepeat;
        const uint32_t id;
        peopleReId::FeatureRes feature;

        float range[2] = {0, 180};

        int histSize[1];
        int channels[1];

        StatusT status;
    };
}// namespace peopleTracker
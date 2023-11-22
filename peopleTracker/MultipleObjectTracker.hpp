#include "../peopleDetector/Detection.hpp"
#include "../peopleReId/fast-reid.hpp"
#include "ObjectTracker.hpp"

namespace peopleTracker {
    class MultipleObjectTracker {
    public:
        explicit MultipleObjectTracker(const std::string& config_path);
        void trackDetections(cv::Mat &frame, const Detections &detections);

    private:
        void cleanUpReleasedObjects();
        void updateTrackedObjects(cv::Mat &frame);
        void drawObjectInsideInfo(cv::Mat &frame);
        static bool isIntersectionValid(cv::Rect newBox, cv::Rect oldBox);
        static bool isDistanceValid(cv::Rect newBox, cv::Rect oldBox, uint32_t id);
        void initNewTrackedObject(cv::Mat &frame, const Detection &detection);
        static bool isObjectAtFrameEdge(cv::Mat &mat, const Detection &detection);
        static StatusT recognizeDirection(const cv::Rect& trackedWin) ;
        static cv::Mat calculateDetectionHistogram(cv::Mat& frame, cv::Rect rect);
        uint32_t recognizeExitingObjectByDetectHist(const cv::Mat& detectHist);
        uint32_t recognizeExitingObjectByFeature(const peopleReId::FeatureRes &detFeature);
        uint32_t getAmountOfInsideObject() const;
        bool isAnyObjectInside();
        peopleReId::FeatureRes calculateDetectionFeature(const cv::Mat& frame, const Detection& detection);

        std::unordered_map<uint32_t, std::unique_ptr<ObjectTracker>> trackedObjects;
        uint32_t nextId{};
        uint32_t frameNr{};
        std::unique_ptr<peopleReId::Fastreid> peopleReId{nullptr};
    };

}// namespace peopleTracker
#pragma once
#include "../peopleDetector/Detection.hpp"
#include "model.hpp"

namespace peopleReId {

    struct FeatureRes {
        std::vector<float> feature;
    };

    class Feature : public Model {
    public:
        explicit Feature(const YAML::Node &config);
        std::vector<FeatureRes> inferenceImages(std::vector<cv::Mat> &vec_img);
        void inferenceFolder(const std::string &folder_name) override;
        void computeSimilarity(const std::vector<FeatureRes> &results_a, const std::vector<FeatureRes> &results_b);
        [[nodiscard]] float computeSimilarity(const FeatureRes &first, const FeatureRes &second) const;
        FeatureRes inferenceDetections(cv::Mat &detection);

    protected:
        std::vector<FeatureRes> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output);
        cv::Mat feature2Mat(const std::vector<FeatureRes> &vec_results);
    };
}// namespace peopleReId

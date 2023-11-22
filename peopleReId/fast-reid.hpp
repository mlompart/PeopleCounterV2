#pragma once

#include "feature.hpp"
namespace peopleReId {

    class Fastreid : public Feature {
    public:
        explicit Fastreid(const YAML::Node &config);
    };
}

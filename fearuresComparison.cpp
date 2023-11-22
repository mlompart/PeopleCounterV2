#include "peopleReId/fast-reid.hpp"

int main()
{

    YAML::Node root = YAML::LoadFile("../peopleReId/config/config.yaml");
    peopleReId::Feature fastReId(root["fastreid"]);
    fastReId.LoadEngine();
    fastReId.inferenceFolder("../samples/reid");
    return 0;
}
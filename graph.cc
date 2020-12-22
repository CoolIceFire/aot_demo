
#include "graph.h"
#include <vector>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

extern "C" int Run(const std::vector<std::vector<std::vector<float>>>& input,
                   std::vector<float>& output) {
    Eigen::ThreadPool tp(std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());
    Graph graph;
    graph.set_thread_pool(&device);

    std::vector<int> dim = {2048};
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i].size() != Graph::kNumArgs) {
            std::cerr << "TFFeaValues size is wrong, expected " << Graph::kNumArgs / 2 << " but get" << input[i].size() << std::endl;
            return -1;
        }
        if ((int)input[i][0].size() != dim[0]) {
            std::cerr << "embedding size is wrong, expected " << dim[0]<< " but get" << input[i][0].size() << std::endl;
            return -1;
        }
        std::copy(input[i][0].data(), input[i][0].data() + dim[0], graph.arg_feed_inputs_layer_data() + i * dim[0]);
    }

    auto ok = graph.Run();
    if (!ok) {
        return -1;
    }

    output.clear();
    output.assign(input.size(), 0.0);
    std::copy(graph.result0_data(), graph.result0_data() + input.size(), output.data());

    return 0;
}

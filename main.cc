// Created By liguang
// Date: 2020/12/22
// ----------------------
//
// Created by 李光 on 2020/12/22.
//
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <boost/algorithm/string.hpp>
#include <dlfcn.h>

using namespace std::chrono;

typedef int (*RUN_FUNC)(const std::vector<std::vector<std::vector<float>>>&,
                        std::vector<float>&);


int main() {
    int INPUT_DIM = 2048;
    int BATCHSIZE = 100;

    void* handle = dlopen("./libmodel.so", RTLD_LAZY);
    if (handle == NULL) {
        std::cerr << "dlopen error." << std::endl;
        return -1;
    }
    RUN_FUNC run_func = reinterpret_cast<RUN_FUNC> (dlsym(handle, "Run"));
    if (run_func == NULL) {
        std::cerr << "get Run error." << std::endl;
        return -1;
    }

    std::vector<std::vector<std::vector<float>>> inputs_tensor;
    std::vector<float> outputs;
    std::vector<std::vector<float>> tmp;

    std::vector<float> vec;
    for (int i = 0; i < INPUT_DIM; ++i) {
        vec.push_back(0.1);
    }
    tmp.push_back(vec);
    inputs_tensor.push_back(tmp);

    for (int i = 0; i < BATCHSIZE-1; ++i) {
        inputs_tensor.push_back(inputs_tensor[0]);
    }

    std::cout << "batch_size = " << inputs_tensor.size() << std::endl;
    std::cout << "input_tensor.dim = " << inputs_tensor[0][0].size() << std::endl;
    auto start = system_clock::now();
    run_func(inputs_tensor, outputs);
    auto end   = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    for (auto w : outputs) {
        std::cout <<  w << std::endl;
    }

    return 0;
}

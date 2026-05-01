#include "nanocompile_model.hpp"
#include <iostream>

int main() {

    alignas(64) float input[8] = { -0.428571429f, -0.285714286f, -0.142857143f, 0.0f, 0.142857143f, 0.285714286f, 0.428571429f, -0.428571429f };
    alignas(64) float output[4] = {0};
    nanocompile::inference(input, output);
    std::cout << "NanoCompile output:";
    for (float v : output) std::cout << " " << v;
    std::cout << "\n";
    return 0;
}

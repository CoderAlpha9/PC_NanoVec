#include "hnsw.hpp"
#include <iostream>
#include <vector>

int main() {
    size_t dim = 128; 
    size_t num_elements = 1000;

    hnsw::HNSW index(dim, 16, 200);

    std::vector<std::vector<float>> dataset(num_elements, std::vector<float>(dim));
    for (size_t i = 0; i < num_elements; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            dataset[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
        index.add_point(dataset[i].data(), i);
    }

    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
        query[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    size_t k = 5; 
    size_t ef_search = 50; 
    
    auto results = index.search_knn(query.data(), k, ef_search);

    std::cout << "Top " << k << " nearest neighbors:\n";
    for (const auto& res : results) {
        std::cout << "Node ID: " << res.first << " | Distance: " << res.second << "\n";
    }

    return 0;
}

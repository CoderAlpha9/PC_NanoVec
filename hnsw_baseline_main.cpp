#include "hnsw.hpp"
#include <iostream>
#include <vector>

int main() {
    size_t dim = 128; // Example dimension
    size_t num_elements = 1000;

    // Initialize the graph (Dimension, M, ef_construction)
    hnsw::HNSW index(dim, 16, 200);

    // Create some synthetic float data
    std::vector<std::vector<float>> dataset(num_elements, std::vector<float>(dim));
    for (size_t i = 0; i < num_elements; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            dataset[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
        // Insert into graph
        index.add_point(dataset[i].data(), i);
    }

    // Define a query vector
    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
        query[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Traverse / Search
    size_t k = 5; // Top 5 nearest neighbors
    size_t ef_search = 50; // Search scope
    
    auto results = index.search_knn(query.data(), k, ef_search);

    std::cout << "Top " << k << " nearest neighbors (Cosine Distance):\n";
    for (const auto& res : results) {
        std::cout << "Node ID: " << res.first << " | Distance: " << res.second << "\n";
    }

    return 0;
}
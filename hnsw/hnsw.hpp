#pragma once

#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <unordered_set>
#include <immintrin.h>
#include <algorithm>

namespace hnsw {

inline float cosine_distance_avx2(const float* a, const float* b, size_t dim) {
    __m256 sum_dot = _mm256_setzero_ps();
    __m256 sum_sq_a = _mm256_setzero_ps();
    __m256 sum_sq_b = _mm256_setzero_ps();

    size_t i = 0;
    // unroll by 8 for avx
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        sum_dot = _mm256_add_ps(sum_dot, _mm256_mul_ps(va, vb));
        sum_sq_a = _mm256_add_ps(sum_sq_a, _mm256_mul_ps(va, va));
        sum_sq_b = _mm256_add_ps(sum_sq_b, _mm256_mul_ps(vb, vb));
    }

    alignas(32) float dot_arr[8];
    alignas(32) float sq_a_arr[8];
    alignas(32) float sq_b_arr[8];
    
    _mm256_store_ps(dot_arr, sum_dot);
    _mm256_store_ps(sq_a_arr, sum_sq_a);
    _mm256_store_ps(sq_b_arr, sum_sq_b);

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int j = 0; j < 8; ++j) {
        dot += dot_arr[j];
        norm_a += sq_a_arr[j];
        norm_b += sq_b_arr[j];
    }

    // clean up remaining dimensions
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;

    float similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    similarity = std::max(-1.0f, std::min(1.0f, similarity)); 
    
    return 1.0f - similarity;
}

using NodeId = int;

struct Node {
    NodeId id;
    const float* data;
    std::vector<std::vector<NodeId>> neighbors; 
};

struct DistId {
    float dist;
    NodeId id;
    bool operator<(const DistId& other) const { return dist < other.dist; }
    bool operator>(const DistId& other) const { return dist > other.dist; }
};

class HNSW {
private:
    size_t dim_;
    size_t M_;
    size_t M0_;
    size_t ef_construction_;
    float mult_;
    
    std::vector<Node> nodes_;
    NodeId enter_point_;
    int max_level_;

    std::mt19937 rng_;
    std::uniform_real_distribution<double> level_dist_;

    float distance(const float* a, const float* b) const {
        return cosine_distance_avx2(a, b, dim_);
    }

    int get_random_level() {
        double r = level_dist_(rng_);
        return static_cast<int>(-std::log(r) * mult_);
    }

    std::priority_queue<DistId, std::vector<DistId>, std::less<DistId>> 
    search_layer(const float* query, NodeId ep, int ef, int level) {
        std::unordered_set<NodeId> visited;
        std::priority_queue<DistId, std::vector<DistId>, std::greater<DistId>> candidates;
        std::priority_queue<DistId, std::vector<DistId>, std::less<DistId>> top_results;

        float d = distance(query, nodes_[ep].data);
        candidates.push({d, ep});
        top_results.push({d, ep});
        visited.insert(ep);

        while (!candidates.empty()) {
            DistId c = candidates.top();
            candidates.pop();

            DistId f = top_results.top();
            if (c.dist > f.dist) break;

            for (NodeId neighbor : nodes_[c.id].neighbors[level]) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    f = top_results.top();
                    float dist_to_query = distance(query, nodes_[neighbor].data);
                    
                    if (dist_to_query < f.dist || top_results.size() < (size_t)ef) {
                        candidates.push({dist_to_query, neighbor});
                        top_results.push({dist_to_query, neighbor});
                        
                        if (top_results.size() > (size_t)ef) {
                            top_results.pop();
                        }
                    }
                }
            }
        }
        return top_results;
    }

    std::vector<NodeId> select_neighbors(std::priority_queue<DistId, std::vector<DistId>, std::less<DistId>>& candidates, size_t m) {
        std::vector<NodeId> selected;
        while (!candidates.empty()) {
            selected.push_back(candidates.top().id);
            candidates.pop();
        }
        std::reverse(selected.begin(), selected.end());
        if (selected.size() > m) {
            selected.resize(m);
        }
        return selected;
    }

public:
    HNSW(size_t dim, size_t M = 16, size_t ef_construction = 200) 
        : dim_(dim), M_(M), M0_(M * 2), ef_construction_(ef_construction), 
          enter_point_(-1), max_level_(-1), rng_(42), level_dist_(0.0, 1.0) {
        mult_ = 1.0f / std::log(1.0f * M_);
    }

    void add_point(const float* data, NodeId id) {
        int level = get_random_level();
        Node new_node{id, data, std::vector<std::vector<NodeId>>(level + 1)};
        
        if (id >= nodes_.size()) {
            nodes_.resize(id + 1);
        }
        nodes_[id] = new_node;

        if (enter_point_ == -1) {
            enter_point_ = id;
            max_level_ = level;
            return;
        }

        NodeId curr_ep = enter_point_;
        float min_dist = distance(data, nodes_[curr_ep].data);

        // find best entry point by dropping down from the top level
        for (int l = max_level_; l > level; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (NodeId neighbor : nodes_[curr_ep].neighbors[l]) {
                    float d = distance(data, nodes_[neighbor].data);
                    if (d < min_dist) {
                        min_dist = d;
                        curr_ep = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // build connections in target levels
        for (int l = std::min(level, max_level_); l >= 0; --l) {
            auto top_candidates = search_layer(data, curr_ep, ef_construction_, l);
            auto neighbors = select_neighbors(top_candidates, l == 0 ? M0_ : M_);
            
            for (NodeId n : neighbors) {
                nodes_[id].neighbors[l].push_back(n);
                nodes_[n].neighbors[l].push_back(id);
                
                size_t max_conn = (l == 0 ? M0_ : M_);
                if (nodes_[n].neighbors[l].size() > max_conn) {
                    std::priority_queue<DistId, std::vector<DistId>, std::less<DistId>> n_candidates;
                    for (NodeId nn : nodes_[n].neighbors[l]) {
                        n_candidates.push({distance(nodes_[n].data, nodes_[nn].data), nn});
                    }
                    nodes_[n].neighbors[l] = select_neighbors(n_candidates, max_conn);
                }
            }
            curr_ep = top_candidates.top().id; 
        }

        if (level > max_level_) {
            max_level_ = level;
            enter_point_ = id;
        }
    }

    std::vector<std::pair<NodeId, float>> search_knn(const float* query, size_t k, size_t ef_search = 50) {
        if (enter_point_ == -1) return {};

        ef_search = std::max(ef_search, k);
        NodeId curr_ep = enter_point_;
        float min_dist = distance(query, nodes_[curr_ep].data);

        for (int l = max_level_; l > 0; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (NodeId neighbor : nodes_[curr_ep].neighbors[l]) {
                    float d = distance(query, nodes_[neighbor].data);
                    if (d < min_dist) {
                        min_dist = d;
                        curr_ep = neighbor;
                        changed = true;
                    }
                }
            }
        }

        auto top_candidates = search_layer(query, curr_ep, ef_search, 0);

        std::vector<std::pair<NodeId, float>> results;
        while (!top_candidates.empty()) {
            results.push_back({top_candidates.top().id, top_candidates.top().dist});
            top_candidates.pop();
        }
        std::reverse(results.begin(), results.end()); 

        if (results.size() > k) {
            results.resize(k);
        }
        return results;
    }
};

} // namespace hnsw

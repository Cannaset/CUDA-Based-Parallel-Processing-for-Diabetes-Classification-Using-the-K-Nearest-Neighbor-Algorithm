#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t error = call;                                                         \
        if (error != cudaSuccess) {                                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "       \
                      << cudaGetErrorString(error) << std::endl;                          \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

// Kernel CUDA: setiap thread menghitung satu jarak antara satu data uji
// dan satu data latih. Indeks global thread dibuat dari blockIdx, blockDim,
// dan threadIdx agar seluruh pasangan test-train dapat diproses paralel.
__global__ void computeDistancesKernel(
    const float* X_train,
    const float* X_test,
    float* distances,
    int n_train,
    int n_test,
    int n_features
) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n_train * n_test;

    if (global_thread_id >= total_pairs) {
        return;
    }

    int test_idx = global_thread_id / n_train;
    int train_idx = global_thread_id % n_train;

    float sum = 0.0f;
    for (int feature = 0; feature < n_features; feature++) {
        float diff = X_test[test_idx * n_features + feature] -
                     X_train[train_idx * n_features + feature];
        sum += diff * diff;
    }

    distances[test_idx * n_train + train_idx] = sqrtf(sum);
}

std::vector<std::vector<float>> readFeatureCSV(const std::string& path, int& rows, int& cols) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    rows = static_cast<int>(data.size());
    cols = rows > 0 ? static_cast<int>(data[0].size()) : 0;
    return data;
}

std::vector<int> readLabelCSV(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::vector<int> labels;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        labels.push_back(std::stoi(line));
    }

    return labels;
}

std::vector<float> flatten(const std::vector<std::vector<float>>& matrix) {
    std::vector<float> flat;
    if (matrix.empty()) {
        return flat;
    }

    flat.reserve(matrix.size() * matrix[0].size());
    for (const auto& row : matrix) {
        for (float value : row) {
            flat.push_back(value);
        }
    }
    return flat;
}

int voteKNearest(const float* distances, const std::vector<int>& y_train, int n_train, int test_idx, int k) {
    std::vector<std::pair<float, int>> dist_label;
    dist_label.reserve(n_train);

    for (int train_idx = 0; train_idx < n_train; train_idx++) {
        float distance = distances[test_idx * n_train + train_idx];
        dist_label.push_back({distance, y_train[train_idx]});
    }

    std::partial_sort(
        dist_label.begin(),
        dist_label.begin() + k,
        dist_label.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        }
    );

    int count_0 = 0;
    int count_1 = 0;
    for (int i = 0; i < k; i++) {
        if (dist_label[i].second == 1) {
            count_1++;
        } else {
            count_0++;
        }
    }

    return count_1 >= count_0 ? 1 : 0;
}

void writePredictions(const std::string& path, const std::vector<int>& predictions) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot write file: " + path);
    }

    file << "prediction\n";
    for (int prediction : predictions) {
        file << prediction << "\n";
    }
}

int main() {
    const std::string x_train_path = "data/processed/X_train.csv";
    const std::string x_test_path = "data/processed/X_test.csv";
    const std::string y_train_path = "data/processed/y_train.csv";
    const std::string y_test_path = "data/processed/y_test.csv";
    const std::string output_path = "outputs/results/gpu_predictions.csv";
    const int k = 5;

    auto total_start = std::chrono::high_resolution_clock::now();

    int n_train = 0;
    int n_features_train = 0;
    int n_test = 0;
    int n_features_test = 0;

    std::vector<std::vector<float>> X_train_matrix = readFeatureCSV(x_train_path, n_train, n_features_train);
    std::vector<std::vector<float>> X_test_matrix = readFeatureCSV(x_test_path, n_test, n_features_test);
    std::vector<int> y_train = readLabelCSV(y_train_path);
    std::vector<int> y_test = readLabelCSV(y_test_path);

    if (n_train == 0 || n_test == 0) {
        throw std::runtime_error("Training or testing data is empty.");
    }

    if (n_features_train != n_features_test) {
        throw std::runtime_error("Feature count mismatch between train and test data.");
    }

    if (static_cast<int>(y_train.size()) != n_train) {
        throw std::runtime_error("y_train length does not match X_train rows.");
    }

    if (static_cast<int>(y_test.size()) != n_test) {
        throw std::runtime_error("y_test length does not match X_test rows.");
    }

    if (n_train < k) {
        throw std::runtime_error("Training samples must be at least k.");
    }

    int n_features = n_features_train;
    std::vector<float> X_train = flatten(X_train_matrix);
    std::vector<float> X_test = flatten(X_test_matrix);
    std::vector<float> distances(static_cast<size_t>(n_train) * n_test);

    float* d_X_train = nullptr;
    float* d_X_test = nullptr;
    float* d_distances = nullptr;

    size_t train_bytes = X_train.size() * sizeof(float);
    size_t test_bytes = X_test.size() * sizeof(float);
    size_t distance_bytes = distances.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_X_train, train_bytes));
    CUDA_CHECK(cudaMalloc(&d_X_test, test_bytes));
    CUDA_CHECK(cudaMalloc(&d_distances, distance_bytes));

    CUDA_CHECK(cudaMemcpy(d_X_train, X_train.data(), train_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X_test, X_test.data(), test_bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int total_pairs = n_train * n_test;
    int grid_size = (total_pairs + block_size - 1) / block_size;
    int total_launched_threads = grid_size * block_size;

    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    CUDA_CHECK(cudaEventRecord(kernel_start));
    computeDistancesKernel<<<grid_size, block_size>>>(
        d_X_train,
        d_X_test,
        d_distances,
        n_train,
        n_test,
        n_features
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float kernel_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, kernel_start, kernel_stop));

    CUDA_CHECK(cudaMemcpy(distances.data(), d_distances, distance_bytes, cudaMemcpyDeviceToHost));

    std::vector<int> predictions;
    predictions.reserve(n_test);
    for (int test_idx = 0; test_idx < n_test; test_idx++) {
        predictions.push_back(voteKNearest(distances.data(), y_train, n_train, test_idx, k));
    }

    writePredictions(output_path, predictions);

    CUDA_CHECK(cudaFree(d_X_train));
    CUDA_CHECK(cudaFree(d_X_test));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));

    auto total_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_stop - total_start;

    std::cout << "GPU kernel time (seconds): " << kernel_time_ms / 1000.0f << std::endl;
    std::cout << "GPU total time (seconds): " << total_elapsed.count() << std::endl;
    std::cout << "Training samples: " << n_train << std::endl;
    std::cout << "Testing samples: " << n_test << std::endl;
    std::cout << "Features: " << n_features << std::endl;
    std::cout << "Block size: " << block_size << std::endl;
    std::cout << "Grid size: " << grid_size << std::endl;
    std::cout << "Total launched threads: " << total_launched_threads << std::endl;
    std::cout << "Predictions saved to: " << output_path << std::endl;

    return 0;
}

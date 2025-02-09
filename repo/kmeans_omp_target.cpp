#include "kmeans_utils.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <limits>
#include <random>
#include <numeric>
#include <map>
#include <algorithm>
#include <omp.h>

// Function to transform a matrix into a one-dimensional vector
std::vector<float> flatten(const std::vector<std::vector<float>>& matrix) {
    std::vector<float> flat;
    for (const auto& row : matrix) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

// Algorithm K-means with OpenMP Offloading
std::vector<int> classify_images(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& centroids) {
    std::vector<int> labels(data.size());

    std::vector<float> data_flat = flatten(data);          // Transform the data into a one-dimensional vector
    std::vector<float> centroids_flat = flatten(centroids); // Transform the centroids into a one-dimensional vector

    // Convert std::vector to regular arrays
    float* data_flat_array = new float[data_flat.size()];
    float* centroids_flat_array = new float[centroids_flat.size()];
    int* labels_array = new int[labels.size()];

    // Copying data into regular arrays
    std::copy(data_flat.begin(), data_flat.end(), data_flat_array);
    std::copy(centroids_flat.begin(), centroids_flat.end(), centroids_flat_array);
    std::copy(labels.begin(), labels.end(), labels_array);

    #pragma omp target teams distribute parallel for \
    map(to: data_flat_array[0:data_flat.size()], centroids_flat_array[0:centroids_flat.size()]) \
    map(from: labels_array[0:labels.size()])
    for (size_t i = 0; i < data.size(); ++i) {
        float min_distance = std::numeric_limits<float>::max();
        int closest_centroid = -1;
        for (size_t j = 0; j < centroids.size(); ++j) {
            float dist = euclidean_distance(data[i], centroids[j]);
            if (dist < min_distance) {
                min_distance = dist;
                closest_centroid = j;
            }
        }
        labels_array[i] = closest_centroid;
    }

    // Copying the result back to std::vector
    std::copy(labels_array, labels_array + labels.size(), labels.begin());

    // Freeing up memory
    delete[] data_flat_array;
    delete[] centroids_flat_array;
    delete[] labels_array;

    return labels;
}

// Function for calculating accuracy
double calculate_accuracy(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
    int correct = 0;

    int* true_labels_ptr = const_cast<int*>(true_labels.data());
    int* predicted_labels_ptr = const_cast<int*>(predicted_labels.data());

    #pragma omp target teams distribute parallel for reduction(+:correct) \
    map(to: true_labels_ptr[0:true_labels.size()]) \
    map(to: predicted_labels_ptr[0:predicted_labels.size()])
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == predicted_labels[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / true_labels.size();
}

int main() {
    // Data loading
    std::vector<std::vector<float>> train_data = readIDXImages("/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/data/mnist/emnist-balanced-train-images-idx3-ubyte", 60000, 784);
    std::vector<int> train_labels = readIDXLabels("/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/data/mnist/emnist-balanced-train-labels-idx1-ubyte", 60000);

    int k = 10;
    std::vector<std::vector<float>> centroids = initialize_centroids_kmeans_pp(train_data, k);
    std::vector<int> labels(train_data.size(), -1);

    double start_time = omp_get_wtime();

    bool centroids_changed = true;
    while (centroids_changed) {
        centroids_changed = false;
        labels = classify_images(train_data, centroids); 

	std::vector<int> cluster_sizes(k, 0);  
       	for (size_t i = 0; i < labels.size(); ++i) {
	       	cluster_sizes[labels[i]]++;
	}

	std::cout << "Cluster sizes: ";
	for (int size : cluster_sizes) {
		std::cout << size << " ";
	}
	std::cout << std::endl;

        std::vector<float> train_data_flat = flatten(train_data);          // Transform the data into a one-dimensional vector
        std::vector<float> centroids_flat = flatten(centroids);            // Transform the centroids into a one-dimensional vector

        // Converting std::vector to regular arrays
        float* train_data_flat_array = new float[train_data_flat.size()];
        float* centroids_flat_array = new float[centroids_flat.size()];
        int* labels_array = new int[labels.size()];

        // Copying data into regular arrays
        std::copy(train_data_flat.begin(), train_data_flat.end(), train_data_flat_array);
        std::copy(centroids_flat.begin(), centroids_flat.end(), centroids_flat_array);
        std::copy(labels.begin(), labels.end(), labels_array);

        #pragma omp target teams distribute parallel for \
        map(to: train_data_flat_array[0:train_data_flat.size()], labels_array[0:labels.size()]) \
        map(tofrom: centroids_flat_array[0:centroids_flat.size()])
        for (int j = 0; j < k; ++j) {
            std::vector<std::vector<float>> cluster_points;
            for (size_t i = 0; i < train_data.size(); ++i) {
                if (labels_array[i] == j) {
                    cluster_points.push_back(train_data[i]);
                }
            }

            std::vector<float> new_centroid(784, 0.0);
            for (const auto& point : cluster_points) {
                for (size_t d = 0; d < 784; ++d) {
                    new_centroid[d] += point[d];
                }
            }
            for (size_t d = 0; d < 784; ++d) {
                new_centroid[d] /= cluster_points.size();
            }
            centroids[j] = new_centroid;
        }

        // Freeing up memory
        delete[] train_data_flat_array;
        delete[] centroids_flat_array;
        delete[] labels_array;
    }

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    std::cout << "Execution time: " << elapsed_time << " seconds" << std::endl;

    std::vector<std::vector<float>> test_data = readIDXImages("/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/data/mnist/emnist-balanced-test-images-idx3-ubyte", 10000, 784);
    std::vector<int> test_labels = readIDXLabels("/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/data/mnist/emnist-balanced-test-labels-idx1-ubyte", 10000);
    double test_accuracy = calculate_accuracy(test_labels, classify_images(test_data, centroids));
    std::cout << "Test accuracy: " << test_accuracy << std::endl;


    // Accuracy evaluation
    double train_accuracy = calculate_accuracy(labels, train_labels);
    std::cout << "Training accuracy: " << train_accuracy << std::endl;
    return 0;
}


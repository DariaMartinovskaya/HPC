#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <numeric>
#include <limits>
#include <algorithm>
// kmeans_utils.cpp
#include "kmeans_utils.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

std::vector<std::vector<float>> readIDXImages(const std::string& filename, int num_images, int image_size) {
    std::cout << "Reading images from: " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    file.seekg(16);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(image_size));
    for (int i = 0; i < num_images; ++i) {
        if (i % 1000 == 0) {
            std::cout << "Processing image " << i + 1 << "/" << num_images << std::endl;
        }
        for (int j = 0; j < image_size; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0; // Normalize pixel values to [0, 1]
        }
    }
    file.close();
    std::cout << "Finished reading " << num_images << " images." << std::endl;
    return images;
}

std::vector<int> readIDXLabels(const std::string& filename, int num_labels) {
    std::cout << "Reading labels from: " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    file.seekg(8);
    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }
    file.close();
    std::cout << "Finished reading " << num_labels << " labels." << std::endl;
    return labels;
}

// Function to calculate Euclidean distance between two points
float euclidean_distance(const std::vector<float>& p1, const std::vector<float>& p2) {
    float distance = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        distance += std::pow(p1[i] - p2[i], 2);
    }
    return std::sqrt(distance);
}

// Function to initialize random centroids
std::vector<std::vector<float>> initialize_centroids_kmeans_pp(const std::vector<std::vector<float>>& data, int k) {
    std::cout << "Initializing centroids using K-means++" << std::endl;
    std::vector<std::vector<float>> centroids;
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(0, data.size() - 1);

    //Initializing first random centroid randomly
    centroids.push_back(data[distrib(gen)]); 
    std::cout << "First centroid initialized." << std::endl;

    for (int i = 1; i < k; ++i) {
	    // Calculating distances from each point to the nearest centroid
        std::vector<float> distances(data.size(), std::numeric_limits<double>::infinity());
        for (int j = 0; j < data.size(); ++j) {
            for (int c = 0; c < centroids.size(); ++c) {
                float dist = euclidean_distance(data[j], centroids[c]);
                distances[j] = std::min(distances[j], dist);
            }
        }

	// Selecting a new centroid with probability proportional to the square of the distance
        float total_dist = std::accumulate(distances.begin(), distances.end(), 0.0);
        std::uniform_real_distribution<> prob_dist(0.0, total_dist);
        float random_value = prob_dist(gen);
        float cumulative_sum = 0.0;

        for (int j = 0; j < data.size(); ++j) {
            cumulative_sum += distances[j];
            if (cumulative_sum >= random_value) {
                centroids.push_back(data[j]);
                break;
            }
        }
        std::cout << "Centroid " << i + 1 << " initialized." << std::endl;
    }
    return centroids;
}

// Function to get the most frequent label in a cluster
int get_most_frequent_label(const std::vector<int>& labels_in_cluster) {
    std::map<int, int> label_counts;
    for (int label : labels_in_cluster) {
        label_counts[label]++;
    }

    int most_frequent_label = -1;
    int max_count = -1;
    for (const auto& entry : label_counts) {
        if (entry.second > max_count) {
            most_frequent_label = entry.first;
            max_count = entry.second;
        }
    }
    return most_frequent_label;
}

// Function for assigning labels to clusters
void assign_labels_to_clusters(std::vector<std::vector<int>>& clusters, const std::vector<int>& labels, std::vector<int>& cluster_labels) {
   	#ifdef USE_OPENMP
	#pragma omp parallel for
	#endif
       	for (int i = 0; i < clusters.size(); ++i) {
	    // Extract labels of all points in the current cluster
        std::vector<int> labels_in_cluster;
        for (int index : clusters[i]) {
            labels_in_cluster.push_back(labels[index]);
        }

	   // Assigning a label to a cluster
        cluster_labels[i] = get_most_frequent_label(labels_in_cluster);
    }
}


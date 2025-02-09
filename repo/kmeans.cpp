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
#include <chrono>
#include "kmeans_utils.h"

// K-means algorithm
std::vector<int> classify_images(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& centroids) {
    std::vector<int> labels(data.size());

    std::cout << "Classifying images into " << centroids.size() << " clusters." << std::endl;
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
        labels[i] = closest_centroid;
    }
    std::cout << "Classification completed." << std::endl;
    return labels;
}

// Function to calculate accuracy
double calculate_accuracy(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
    int correct = 0;

    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == predicted_labels[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / true_labels.size();
}

int main() {
	std::cout << "Reading training data..." << std::endl;
       	std::vector<std::vector<float>> train_data = readIDXImages("/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/data/mnist/emnist-balanced-train-images-idx3-ubyte", 60000, 784);
       	std::vector<int> train_labels = readIDXLabels("/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/data/mnist/emnist-balanced-train-labels-idx1-ubyte", 60000); 
    
    // Initializing centroids with the help of K-means++ 
    int k = 10;               // Number of clusters
    std::vector<std::vector<float>> centroids = initialize_centroids_kmeans_pp(train_data, k);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(0,train_data.size() - 1);
    std::cout << "First centroid initialized." << std::endl;


    // Main K-means cycle
    bool centroids_changed = true;
    std::vector<int> labels(train_data.size(), -1);

    auto start_time = std::chrono::high_resolution_clock::now();

    while (centroids_changed) {
        centroids_changed = false;

	// Labling the nearest centroid
	std::cout << "Reassigning labels..." << std::endl;
	labels = classify_images(train_data, centroids);

	// Check cluster sizes
	std::vector<int> cluster_sizes(k, 0);
       	for (size_t i = 0; i < labels.size(); ++i) {
	       	cluster_sizes[labels[i]]++;
       	}
       
	std::cout << "Cluster sizes: ";
	for (int size : cluster_sizes) {
		std::cout << size << " ";
       	}
	std::cout << std::endl;

	// Updating centroids
	for (int j = 0; j < k; ++j) {
            std::vector<std::vector<float>> cluster_points;
            for (size_t i = 0; i < train_data.size(); ++i) {
                if (labels[i] == j) {
                    cluster_points.push_back(train_data[i]);
                }
            }

	    // Calculating new centroid
            std::vector<float> new_centroid(784, 0.0);  // for  28x28 size pictures
            for (const auto& point : cluster_points) {
                for (size_t d = 0; d < 784; ++d) {
                    new_centroid[d] += point[d];
                }
            }
            for (size_t d = 0; d < 784; ++d) {
                new_centroid[d] /= cluster_points.size();
            }

            // Checking if centroid is changed
            if (euclidean_distance(centroids[j], new_centroid) > 1e-2) {
                centroids_changed = true;
		std::cout << "Centroid " << j << " updated." << std::endl;
	    }	

            centroids[j] = new_centroid;
	 
	    //Output the updated centroid
            std::cout << "Updated centroid " << j << ": ";
            for (size_t d = 0; d < 784; ++d) {
                std::cout << centroids[j][d] << " ";
            }
            std::cout << std::endl;
	}
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    std::cout << "K-Means execution time: " << elapsed_time.count() << " seconds" << std::endl;

    std::vector<int> cluster_labels(k);
    std::vector<std::vector<int>> clusters(k);
    for (size_t i = 0; i < labels.size(); ++i) {
        clusters[labels[i]].push_back(i);
    }
   
    assign_labels_to_clusters(clusters, train_labels, cluster_labels);
   
    for (int i = 0; i < cluster_labels.size(); ++i) {
	    std::cout << "Cluster " << i << " label: " << cluster_labels[i] << std::endl;
    }


    // Accuracy evaluation on training data
    double train_accuracy = calculate_accuracy(labels, train_labels);
    std::cout << "Training accuracy: " << train_accuracy << std::endl;

    // Loading test data
    std::cout << "Reading test data..." << std::endl;
    std::vector<std::vector<float>> test_data = readIDXImages("/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/data/mnist/emnist-balanced-test-images-idx3-ubyte", 10000, 784); 
    std::vector<int> test_labels = readIDXLabels("/home/alunos/tei/2024/tei26610/hpc2/my-repo/hpc/data/mnist/emnist-balanced-test-labels-idx1-ubyte", 10000);

    // Accuracy evaluation on training data
    double test_accuracy = calculate_accuracy(test_labels, classify_images(test_data, centroids));
    std::cout << "Test accuracy: " << test_accuracy << std::endl;

    return 0;
}

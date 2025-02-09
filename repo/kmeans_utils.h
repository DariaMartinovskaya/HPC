// kmeans_utils.h
#ifndef KMEANS_UTILS_H
#define KMEANS_UTILS_H

#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <limits>
#include <numeric>
#include <map>

std::vector<std::vector<float>> readIDXImages(const std::string& filename, int num_images, int image_size);
std::vector<int> readIDXLabels(const std::string& filename, int num_labels);
float euclidean_distance(const std::vector<float>& p1, const std::vector<float>& p2);
std::vector<std::vector<float>> initialize_centroids_kmeans_pp(const std::vector<std::vector<float>>& data, int k);
int get_most_frequent_label(const std::vector<int>& labels_in_cluster);
void assign_labels_to_clusters(std::vector<std::vector<int>>& clusters, const std::vector<int>& labels, std::vector<int>& cluster_labels);

// GPU-specific functions
//void kmeans_gpu(std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& centroids, int k);
//std::vector<int> classify_images_gpu(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& centroids);
//void update_centroids_gpu(std::vector<std::vector<float>>& centroids, const std::vector<std::vector<float>>& data, const std::vector<int>& labels, int k);

#endif // KMEANS_UTILS_H


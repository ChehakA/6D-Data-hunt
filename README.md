# 6D Cluster Analysis and Visualization

This project implements K-Means clustering on a high-dimensional (6D) dataset, followed by dimensionality reduction using Principal Component Analysis (PCA) for visualization purposes. The project focuses on understanding complex patterns in the data by clustering and visualizing the results in 3D space.

## Project Overview

- **Dataset**: The dataset used for this analysis contains six features (x1, x2, x3, x4, x5, x6) that describe various attributes of the data points.
- **Preprocessing**: The data is first standardized to ensure each feature has a mean of zero and a standard deviation of one.
- **Dimensionality Reduction**: PCA is applied to reduce the dataset to 3 principal components (x1, x2, x3), which are used for visualization.
- **Clustering**: The K-Means clustering algorithm is used to group the data points into clusters based on their similarities.
- **Visualization**: The 3D scatter plots visualize the results of clustering and show how each cluster behaves in the reduced 3D PCA space.
  
## Features

- K-Means clustering on a 6D dataset.
- Dimensionality reduction using PCA for better visualization.
- Cluster centroids and sizes displayed.
- 3D scatter plot of each cluster.
- Saves each cluster’s 3D plot as an image (`cluster_<cluster_number>_3D.png`).

## Setup and Installation

To run this project, follow the steps below:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/6D-Cluster-Analysis-and-Visualization.git
   cd 6D-Cluster-Analysis-and-Visualization

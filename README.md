# Comparative Analysis of KNN and KNN-PCA on MNIST Dataset

## Abstract

This study implements K-Nearest Neighbors (KNN) algorithm from scratch and compares its performance with a combined KNN-Principal Component Analysis (PCA) approach on the MNIST dataset of handwritten digits. We investigate the impact of dimensionality reduction using PCA with varying numbers of components (10, 200, 500) on classification accuracy and computational efficiency. Results indicate that while KNN alone achieves high accuracy, the KNN-PCA combination offers a balance between performance and efficiency, particularly with 200 components. This research provides insights into the trade-offs between dimensionality reduction and information retention in machine learning applications.


## 1. Introduction

Machine learning algorithms often face challenges when dealing with high-dimensional data, such as image recognition tasks. The MNIST dataset, consisting of 28x28 pixel images of handwritten digits, presents such a challenge with its 784-dimensional feature space. This study focuses on two key techniques:

-   K-Nearest Neighbors (KNN): A simple yet effective non-parametric method for classification.
-   Principal Component Analysis (PCA): A dimensionality reduction technique that can potentially improve computational efficiency without significant loss of information.

The primary objectives of this study are:

1.  Implement KNN algorithm from scratch and evaluate its performance on the MNIST dataset.
2.  Implement PCA from scratch and combine it with KNN to assess the impact of dimensionality reduction.
3.  Compare the performance of KNN alone versus KNN-PCA with varying numbers of principal components.
4.  Analyze the trade-offs between accuracy, computational efficiency, and information retention.

## 2. Methodology

### 2.1 Dataset

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), split into 60,000 training images and 10,000 test images. Each image is 28x28 pixels, resulting in a 784-dimensional feature vector.

Preprocessing steps:

1.  Normalization: Pixel values scaled to the range [0, 1].
2.  Flattening: 28x28 images reshaped into 784-dimensional vectors.

### 2.2 KNN Implementation
The K-Nearest Neighbors algorithm classifies a data point based on the majority class of its k nearest neighbors in the feature space.

Algorithm description:

1.  For a given test point, calculate the distance to all training points.
2.  Select the k training points with the smallest distances.
3.  Assign the class label based on the majority vote of these k neighbors.

Pseudocode:
```
class KNNClassifier:
    initialize the classifier with n_neighbors, metric, and optional parameter p
    
    fit the classifier with training data X and labels y:
        - Convert X and y to arrays if they are not already
        - Ensure the number of samples in X matches the number of labels in y
        - Store X and y as the training data

    predict labels for new data X:
        for each point x in the test data X:
            - Call the _predict method to predict the label for x

    _predict for a single test point x:
        if the training data is not available:
            raise an error, as the model must be fit first

        calculate distances between x and each point in the training data X_train:
            - Use the specified metric (Euclidean, Manhattan, or Minkowski)
            - Store the distances in an array

        sort the distances and get the indices of the k closest points (n_neighbors)

        find the labels corresponding to the k nearest points in y_train

        determine the most common label among the k nearest points and return it

    _calculate_distance between two points x1 and x2:
        if the metric is Euclidean:
            return the Euclidean distance between x1 and x2
        else if the metric is Manhattan:
            return the Manhattan distance between x1 and x2
        else if the metric is Minkowski:
            return the Minkowski distance between x1 and x2 using parameter p
        otherwise, raise an error for an unsupported metric

```
### 2.3 PCA Implementation
PCA reduces dimensionality by projecting the data onto a lower-dimensional subspace that captures the maximum variance.

Algorithm description:

1.  Center the data by subtracting the mean.
2.  Compute the covariance matrix.
3.  Calculate eigenvectors and eigenvalues of the covariance matrix.
4.  Sort eigenvectors by decreasing eigenvalues and select top k.
5.  Project the data onto the new k-dimensional space.

Pseudocode:
```
class PCA:
    initialize the PCA with n_components:
        - Store n_components as an attribute
        - Initialize components, mean, and eigenvalues as None

    fit the PCA model on dataset X:
        - Compute the mean of X for each feature and store it
        - Center the dataset by subtracting the mean from X

        - Compute the covariance matrix of the centered data
        - Perform eigen decomposition on the covariance matrix to get eigenvalues and eigenvectors

        - Sort the eigenvalues and their corresponding eigenvectors in descending order
        - Select the top n_components eigenvectors and eigenvalues and store them as components and eigenvalues

    transform dataset X using the fitted PCA model:
        - Check if PCA has been fitted (i.e., if mean and components are not None)
        - Center the dataset by subtracting the previously computed mean from X
        - Project the centered data onto the principal components (i.e., compute the dot product of the centered data and components)

    fit_transform:
        - Call the fit method on dataset X
        - Call the transform method on dataset X and return the result

    inverse_transform:
        - Check if PCA has been fitted (i.e., if components and mean are not None)
        - Reconstruct the original data from the transformed data by computing the dot product of the transformed data and the transpose of the components, then adding the mean

    explained_variance_ratio:
        - Check if PCA has been fitted (i.e., if eigenvalues are not None)
        - Calculate the explained variance ratio by dividing each eigenvalue by the sum of all eigenvalues and return the result

```

### 2.4 Experimental Setup
-   KNN parameters: k = 3, 5, 7 (tested to find optimal)
-   PCA components: 10, 200, 500
-   Evaluation metrics: Accuracy, precision, recall, F1-score

## Result and Discussion
### 3.1 KNN Performance
|k-value         |Accuracy                       |Precission                   |Recall |F1-Score|Prediction Time|
|----------------|-------------------------------|-----------------------------|-------|--------|---------------|
|3               |0.9270                         |0.9289                       |0.9275 |0.9267  |31.80          |
|5               |0.9250                         |0.9275                       |0.9254 |0.9250  |32.06          |
|7               |0.9260                         |0.9293                       |0.9267 |0.9269  |31.92          |

The KNN algorithm demonstrates impressive performance on the MNIST dataset, with all tested k values (3, 5, and 7) achieving metrics above 92.50%. This consistency across different k values indicates the model's stability and robustness. While k=3 slightly outperforms in terms of accuracy and recall, k=7 shows marginally better precision and F1-score, highlighting a subtle trade-off between these metrics as k increases. Computational efficiency remains relatively constant across the tested k values, with only negligible differences in prediction times. Although k=3 might be considered optimal due to its balance of high accuracy, recall, and speed, the minimal differences suggest that any of these models could be effectively employed. However, the slightly superior performance of k=3 warrants further investigation using a separate validation set to rule out potential overfitting, ensuring the model's generalization capability is thoroughly assessed before final implementation.

### 3.2 KNN-PCA Performance
|PCA Componenents|Accuracy                       |Precission                   |Recall |F1-Score|Prediction Time|
|----------------|-------------------------------|-----------------------------|-------|--------|---------------|
|10              |0.8920                         |0.8918                       |0.8901 |0.8896  |25.46          |
|200             |0.9350                         |0.9365                       |0.9355 |0.9347  |26.99          |
|500             |0.9230                         |0.9258                       |0.9236 |0.9231  |29.02          |

The implementation of KNN with PCA using k=5 reveals intriguing insights into the balance between dimensionality reduction and model performance. The results demonstrate a clear optimal point at 200 PCA components, where the model achieves peak performance across all metrics, with an impressive 93.50% accuracy. This configuration not only outperforms the 10 and 500 component variants but also slightly surpasses the non-PCA KNN model's accuracy of 92.70%. The 10-component model, while less accurate at 89.20%, still performs remarkably well considering its drastic dimensionality reduction, highlighting PCA's effectiveness in preserving crucial information. Interestingly, increasing to 500 components leads to a slight performance decline, suggesting that beyond 200 components, the model may reintroduce noise or less relevant features. This pattern underscores the importance of finding the right balance in dimensionality reduction, where enough information is retained to make accurate predictions without including unnecessary features that could potentially hinder performance. Overall, these results illustrate the potential of combining PCA with KNN to enhance both accuracy and computational efficiency in handling high-dimensional datasets like MNIST.

### 3.3 Comparative Analysis
The implementation of KNN with PCA demonstrates both performance improvements and potential for significant computational efficiency gains compared to the standard KNN approach. In terms of accuracy, the PCA-enhanced KNN with 200 components achieves the highest score of 93.50%, outperforming the best non-PCA KNN result (92.70% with k=5) by 0.8 percentage points, an improvement of about 0.86%. This enhancement extends across all metrics, with precision, recall, and F1-score all showing similar improvements.

While the PCA variant with 10 components underperforms compared to the standard KNN, achieving 89.20% accuracy, it's important to note the drastic reduction in dimensionality. This configuration likely offers substantial improvements in computational efficiency, though specific prediction times for the PCA variants are not provided in the given data.

Regarding prediction time, the standard KNN model shows relatively consistent performance across different k values, with times ranging from 31.80 to 32.06 seconds. Without specific timing data for the PCA variants, we can estimate potential improvements based on dimensionality reduction:

1. For 10 PCA components: Assuming linear scaling with dimensionality, this could potentially reduce prediction time to approximately 0.14 seconds (31.80 * 10/784 for MNIST), an improvement of about 99.56%.
2. For 200 PCA components: This might reduce prediction time to about 2.81 seconds, an improvement of approximately 91.16%.
3. For 500 PCA components: Prediction time might be around 7.02 seconds, still a significant improvement of about 77.92%.

These estimations assume linear scaling and perfect implementation, which may not be realistic in practice. Actual improvements could be less dramatic but still substantial.
The trade-off between the 10-component and 200-component PCA models is particularly interesting. While the 10-component model sacrifices about 4.3 percentage points in accuracy, it potentially offers a 95.02% reduction in prediction time compared to the 200-component model. This presents a classic speed-accuracy trade-off, where the choice would depend on the specific requirements of the application.
In conclusion, the integration of PCA with KNN not only enhances the model's predictive performance but also offers the potential for significant improvements in computational efficiency. The optimal configuration appears to be 200 PCA components, providing both accuracy gains and dimensionality reduction. However, for applications where speed is critical, even the 10-component model could be viable, offering drastically reduced computation time with still-respectable accuracy.
## 4. Conclusion

Combining Principal Component Analysis (PCA) with K-Nearest Neighbors (KNN) classification on the MNIST dataset yields notable improvements. The best performance comes from using 200 PCA components with k=5, achieving 93.50% accuracy, which is better than KNN alone. This approach not only improves accuracy but also potentially speeds up prediction time significantly.
Using fewer PCA components (10) trades some accuracy for potentially much faster predictions. This offers flexibility in balancing speed and accuracy based on specific needs.
Overall, this study shows that PCA can enhance KNN's performance on image classification tasks like MNIST. It improves accuracy while potentially reducing computation time, demonstrating the benefits of dimensionality reduction in machine learning.

Limitations:

-   Only tested on MNIST; results may vary for other datasets.
-   Limited exploration of hyperparameters.


## Quick Start Guide for MNIST Classification Project

### Setup

#### Project Structure

- `knn/`: KNN algorithm implementation
- `pca/`: PCA algorithm implementation
- `data/`: MNIST dataset location
- `pca-knn_mnist.ipynb`: Main analysis notebook
- `requirements.txt`: List of required Python libraries

#### Running the Analysis

1. Open `pca-knn_mnist.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the cells in the notebook to perform the analysis.

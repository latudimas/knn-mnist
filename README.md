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
function KNN_classify(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    class_votes = {}
    for _, label in neighbors:
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1
    
    return max(class_votes, key=class_votes.get)
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
NEED TO ADD
```

### 2.4 Experimental Setup
-   KNN parameters: k = 3, 5, 7 (tested to find optimal)
-   PCA components: 10, 200, 500
-   Evaluation metrics: Accuracy, precision, recall, F1-score

## Result and Discussion
### 3.1 KNN Performance
|k-value         |Accuracy                       |Precission                   |Recall |F1-Score
|----------------|-------------------------------|-----------------------------|-------|-|
|Single backticks|`'Isn't this fun?'`            |'Isn't this fun?'            |rec|f1|
|Quotes          |`"Isn't this fun?"`            |"Isn't this fun?"            |re|f1|
|Dashes          |`-- is en-dash, --- is em-dash`|-- is en-dash, --- is em-dash|re|f1|

KNN achieved high accuracy on the MNIST dataset, with k=5 performing best. This demonstrates the effectiveness of KNN for image classification tasks, likely due to the local structure in the digit images.

### 3.2 KNN-PCA Performance
|PCA Components         |Accuracy                       |Precission                   |Recall |F1-Score
|----------------|-------------------------------|-----------------------------|-------|-|
|Single backticks|`'Isn't this fun?'`            |'Isn't this fun?'            |rec|f1|
|Quotes          |`"Isn't this fun?"`            |"Isn't this fun?"            |re|f1|
|Dashes          |`-- is en-dash, --- is em-dash`|-- is en-dash, --- is em-dash|re|f1|

PCA with 500 components nearly matched the performance of KNN alone, while significantly reducing dimensionality. The 200-component version offered a good balance between accuracy and efficiency.
You can delete the current file by clicking the **Remove** button in the file explorer. The file will be moved into the **Trash** folder and automatically deleted after 7 days of inactivity.

### 3.3 Comparative Analysis
-   KNN alone achieved the highest accuracy but at the cost of higher computational complexity.
-   KNN-PCA with 500 components closely matched KNN's performance with reduced dimensionality.
-   KNN-PCA with 200 components offered the best trade-off between accuracy and efficiency.
-   KNN-PCA with 10 components showed a notable drop in accuracy, indicating significant information loss.
## 4. Conclusion

This study demonstrates the effectiveness of combining KNN with PCA for handwritten digit classification. Key findings include:

-   KNN alone achieves high accuracy but at higher computational cost.
-   PCA can significantly reduce dimensionality while largely preserving classification performance.
-   A sweet spot exists (around 200 components) where dimensionality reduction greatly improves efficiency with minimal accuracy loss.

Limitations:

-   Only tested on MNIST; results may vary for other datasets.
-   Limited exploration of hyperparameters.

Future directions:

-   Explore other dimensionality reduction techniques (e.g., t-SNE, UMAP).
-   Investigate the impact of different distance metrics in high-dimensional spaces.
-   Apply these techniques to more complex datasets to test generalizability.

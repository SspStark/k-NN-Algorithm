# k-Nearest Neighbors
## Handwritten Digit Recognition
Domain Problem: Digit Recognition
- Recognize handwritten numbers
- Subproblem: Recognize handwritten digits
- Ex: Google Lens

Humans can perform any task with ease, but can't describe(step-by-step) how they are doing it!
- Ex: Car Driving, Image Classification, Face Recognization and many more....
- But Machines can't do this easily like humans do, so with Supervised learning by giving Labeled data, which is a supervision to a machine will achive this ability to recognize and perform tasks by their own.

This Handwritten Digit Recognization comes under Classification Supervised Learning, because
- Discrete valued target
- Target values: 0,1,2,3,4,5,6,7,8,9

ML Problem: Making ML Software to recognize digits or numbers from the given image by training it with labeled data of numbers.

Data Collection
- To acquire the labeled data for this project we can use MNIST Dataset or it's new version EMNIST Dataset which contains 240,000 training images and 40,000 testing images because every user will write numbers differently so we need different types of habdwriting images of numbers.

Understanding the Data
- The Data contains greyscale images and corresponding labels.
- Image size: 28 x 28 pixels.
- In preprocessing step the color of images may be different colors but to make it easier to train the machine the images are converted to black & white images.
- In greyscale the value 0 is pitch black and the value 255 is white, based on the values the shade will change.
- The machine will take this image as matrix with values of greyscale.

## Computer Vision
Enabling computers to understand the content of digital images or videos. In order to search images or videos, computers needs to know what the image or video contains.
- Ex: Object Detection, Face Recognition, Image Classification, Feature Matching
- steps in computer vision: 1.Acquiring Image, 2.Processing the Image, 3.Understanding the Image.

There are many classification algorithms to work on this project and k-NN is one of them.

## k-NN Algorithm
The k-Nearest Neighbors (k-NN) algorithm is a simple and intuitive supervised machine learning algorithm used for classification and regression tasks. In the context of classification, it falls under the category of instance-based learning.
### k-NN for Classification:
Data Representation
- The algorithm relies on a dataset where each data point is represented by a set of features.

Nearest Neighbors:
- To classify a new data point, the algorithm identifies the k training data points in the dataset that are closest to the new point based on a distance metric (commonly Euclidean distance).
- We choose the most frequent target label among the target labels of the k nearest points.

Majority Voting:
- For classification, the algorithm assigns the class label that is most frequent among the k-nearest neighbors to the new data point. This is often referred to as majority voting.

Choosing k:
- The choice of the parameter k (the number of neighbors) is a crucial aspect. A smaller k can lead to more sensitive models, while a larger k can make the model more robust but potentially less sensitive to local variations.

For a new input:
- Compute the distance of the new input to all inputs.
- Sort the data based on the distance.
- Choose the majority among k nearest data.

Considerations:
- The choice of the parameter k is critical and can significantly impact the model's performance.
- The algorithm's sensitivity to the choice of distance metric and k requires careful tuning for optimal results.
- While k-NN is simple, it may not perform well in high-dimensional spaces or with large datasets.

k-NN is a non-parametric, lazy learning algorithm, meaning it does not make strong assumptions about the underlying data distribution and postpones learning until prediction time. It is straightforward to implement and understand, making it a useful algorithm for certain types of problems, especially in smaller datasets.

**Euclidean Distance**: Vector Notation
- position can also be represented as a 2D vector, [a1,a2] & [b1,b2]
- squareroot((a1-b1)^2 + (a2-b2)^2)


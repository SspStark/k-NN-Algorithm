# k-Nearest Neighbors
## Handwritten Digit Recognition
Domain Problem: Digit Recognition
- Recognize handwritten numbers
- Subproblem: Recognize handwritten digits
- Ex: Google Lens

Humans can perform any task with ease, but can't describe (step-by-step) how they are doing it!
- Ex: Car Driving, Image Classification, Face Recognization and many more....
- But Machines can't do this easily like humans do, so with Supervised learning by giving Labeled data, which is a supervision to a machine will achive this ability to recognize and perform tasks by their own.

This Handwritten Digit Recognization comes under Classification Supervised Learning, because
- Discrete valued target
- Target values: 0,1,2,3,4,5,6,7,8,9

ML Problem: Making ML Software to recognize digits or numbers from the given image by training it with labeled data of numbers.

Data Collection
- To acquire the labeled data for this project we can use the data from the websites MNIST Dataset or it's new version EMNIST Dataset which contains 240,000 training images and 40,000 testing images because every user will write numbers differently so we need different types of handwriting images of numbers.

Understanding the Data from MNIST Dataset
- The Data contains greyscale images and corresponding labels.
- Image size: 28 x 28 pixels.
- In preprocessing step the color of images may be different colors but to make it easier to train the machine the images are converted to black & white images.
- In greyscale the value 0 is pitch black and the value 255 is white, based on the values the shade will change.
- The machine will take this image as matrix with values of greyscale.

## Computer Vision
Enabling computers to understand the content of digital images or videos. In order to search images or videos, computers needs to know what the image or video contains.
- Ex: Object Detection, Face Recognition, Image Classification, Feature Matching
- steps in computer vision: 1.Acquiring Image, 2.Processing the Image, 3.Understanding the Image.

For this project we need a classification algorithm and There are many classification algorithms.
- k-NN, Logistic Regression, SVM, Decision Trees, Neural Networks .............
## k-NN Algorithm
The k-Nearest Neighbors (k-NN) algorithm is a simple and intuitive supervised machine learning algorithm used for classification and regression tasks. In the context of classification, it falls under the category of instance-based learning.

k-NN Simply represents that we are predecting the result based on nearest neighbours and k represents how many nearest neighbours we should consider.
- Eg:- Saying "Show me your friends and I'll tell you what kind of person you are".
### k-NN for Classification:
Data Representation
- The algorithm relies on a dataset where each data point is represented by a set of features.

Nearest Neighbors:
- To classify a new data point, the algorithm identifies the k training data points in the dataset that are closest to the new point based on a distance metric (commonly Euclidean distance).
- We choose the most frequent target label among the target labels of the k nearest points.

#### Euclidean Distance: Vector Notation

The position of two points can be represented as n-dimensional vectors:  
$\mathbf{a} = [a_1, a_2, \dots, a_n] \quad \text{and} \quad \mathbf{b} = [b_1, b_2, \dots, b_n]$

The Euclidean distance between these two vectors is given by the formula:

$$d(\mathbf{a}, \mathbf{b}) = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2 + \dots + (a_n - b_n)^2}$$

Majority Voting:
- For classification, the algorithm assigns the class label that is most frequent among the k-nearest neighbors to the new data point. This is often referred to as majority voting.

For a new input:
- Compute the distance of the new input to all inputs.
- Sort the data based on the distance.
- Choose the majority among k nearest data.

Choosing k:
- The choice of the parameter k (the number of neighbors) is a crucial aspect. A smaller k can lead to more sensitive models, while a larger k can make the model more robust but potentially less sensitive to local variations.
- So k value should not be very small and should not be very large.
- k is a Hyperparameter, to choose the k at first we take a random value, by tuning and based on Performance Metric we choose the k value.

Considerations:
- k-NN Model doesn't learn any complicated mathematical models, Just memorizes the training data.
- The choice of the parameter k is critical and can significantly impact the model's performance.
- The algorithm's sensitivity to the choice of distance metric and k requires careful tuning for optimal results.
- While k-NN is simple, it may not perform well in high-dimensional spaces or with large datasets.
- In very high dimentional spaces, almost all the points are far away from each other.
- The Dimentionality increases the probability of a point falling in middle decreases.

k-NN is a non-parametric, lazy learning algorithm, meaning it does not make strong assumptions about the underlying data distribution and postpones learning until prediction time. It is straightforward to implement and understand, making it a useful algorithm for certain types of problems, especially in smaller datasets.

When we are sorting the data with k-NN we can get Distance Ties and Voting Ties then we have to break these ties.

Distance Tie
- When sorting the data suppose k=4 and we have to consider 4 nearest neighbours but the 4th and 5th point distance is same.
- To break this tie we can go with random choosing.
- Other way is instead of choosing randomly, consider all the points which are equidistant from the test input.

Voting Tie
- suppose we sorted the nearest neighbours based on distance and k=4 so in those 4 we need consider majority ones but we got 2 same and other 2 same now the majority is same.
- To break this tie we have *Weighted Voting*, nearby training examples can be given higher weightage `weight inversely proportional to 1/euclidean distance`
- We have a variant of k-NN which is called **Distance Weighted k-NN**, Weighted voting can be used even when there are no ties.
- other methods are Choosing the target label which has the majority in the entire training set (or) keep decreasing k upto 1 till the tie breaks.

Voronoi Cells
- let's say k=1, each training example(point) has a small area surrounding it.
- Any new input in this area will be predicted as the target label of the training example,that area is called *voronoi cell*
- All the training examples have a voronoi cell around them formimg a complex shape is called **Voronoi Tessellation** and the boundary separating the voronoi cells of 2 classes for 1-NN is called **Decision Boundary**.
- A set of all points that are closest to the instance than to any other instance in the training set.
- A minor change in the training data changes the decision boundary drastically.
- The boundary b/w voronoi cells in the voronoi tessellation is a set of points which are equidistant from 2 different training examples.
### Performance Metrics
In prediction we can have 4 scenarios 1.True Positive, True Negetive, False Positive and False Negative.
- TP: Actual positives which are predicted as positives.
- FN: Actual positives which are predicted as negatives.
- TN: Actual negatives which are predicted as negatives.
- FP: Actual negatives which are predicted as positives.

Calculating Performance
- Accuracy = (TP + TN)/(TP + TN + FP + FN)
- Recall: Here we only consider the actual positives and we find how many of these actual positives we are predicting as positives. TP/(TP + FN), The ratio of true positives and the total number of actual positive instances.
- Precision: Here we consider all the positive prediction and find the actual positives. TP/(TP + FP), The ratio of true positives and the total number of instances which are predicted as positives.
- F1 Score: It balance both Recall and Precision, 2PR/P+R
- Macro-average for Muiti-class Metrics is average of F1 Scores
- We use Macro-average metric when a particular class is important than other.
- We use Weighted-average when the classes are imbalanced and all have equal importance.
- We use Micro-average when we have to predict multiple targets with multiple labels scenario.
- When we use Micro-average to predict single target with single label scenario the FP and FN will be same which makes precision, recall and F1 score same.

Prediction on New Inputs
- For a 28 x 28 pixel image the possible inputs are 256^(28 x 28) can be given to the model, for a color image it will be 256^(28 x 28 x 3) because color images have RGB.
- For a 512 x 512, the size of the input space is 256^(512 X 512).In this large space we have portion of data called Population.
- The Population is the collection of all items of interest to our ML problem, which in the case of a Handwritten Digit Recognition, is the images of every single individual handwritten digit images across the globe.
- Since the training data cannot contain the images of every handwritten, it should be a representative sample of the Population.
- The Representative sample is a subset of the population that accurately reflects the members of the entire population.

Splitting the prepared data for ML into training data, validation data, testing data.
- For a balanced data where the all different label examples are equal then we use Random Shuffle Split.
- For a Imbalanced data we use Stratified Shuffle Split.

No Free Lunch Theorem
- In order to make predictions, we have to make some assumptions about data.
- There will always be a dataset which violates these assumptions.
- Different algorithms make very different assumptions.
- There is NO **one best algorithm**.

k-NN Assumption
- Similar points have same labels
- Two points are similar, if they are close to each other(distance b/w them is less)

Distance Measures
- In high Dimensional spaces Euclidean Distance is not preferable based on researches done by ML researchers.
- There are other ways of measuring distances b/w points to get better results, one is **Manhattan** or **L^1 Norm** distance.
- Manhattan distance: $$d(\mathbf{a}, \mathbf{b}) = |a_1 - b_1| + |a_2 - b_2| + \dots + |a_n - b_n|$$
- others distance measures are *Minkowski or L^n Norm, Chebyshev* etc.
- Minkowski distance: nth root(sum(|ai-bi| ^ n))
- when n=1, it will become manhattan distance and n=2, it will become Euclidean distance.
- when n is infinity, it will become **Chebyshev Distance**
- Chebyshev distance: max value(differences of corressponding points of a & b vectors: |ai-bi|)

Notation
- notation refers to a system of symbols and rules used to represent information or communicate ideas. It's a way of expressing complex concepts or structures using a set of symbols or characters. It makes the communication and understanding easy.
- "m" refers to number of training examples or rows
- "n" refers to numbers of features or columns
- x - inputs, here x is a vector and y - output
- Dataset D={(x^[m],y[m])}, here x^[m] is feature vector and y[m] is label.
- X - input space, R - real number ==> X=R^n, f(vector x)=y - Unknown target function, h~f - Hypothesis function
- xn^[m] - m is written in superscript and n is written in subscript

Data Terminology
- Example: observation, record, instance, sample
- Feature: variable, attribute, dimension, predictor, covariate
- Target: output, outcome, label/class(classification), ground truth
- Prediction: Output from Model
### Hands-On-k-NN
- [Working with MNIST Dataset](https://colab.research.google.com/drive/1C-i6nnbCs0Kx0g_mE5sYhz0GbMFjWekU)
- [Handwritten Digit Recognition using k-NN](https://colab.research.google.com/drive/1JMVAty8vRY1AWNO-OBRCjptm2NifZtBi)
- [Handwritten Digit Recognition using k-NN Solution](https://colab.research.google.com/drive/16_2O4dzRshPCYu5QHIh3zEGTWNhbNBU9)
- [Applying k-NN to MNIST Dataset](https://colab.research.google.com/drive/1pHgIFlAvUPpQoxLciLvho18g1snFy29K#scrollTo=G-boAKvZkKRJ)

### Improving the Performance of k-NN
Computational Complexity - Naive k-NN.\
Improving Computational Performance of k-NN with Data Structures based on memory and time taken for an algorithm. Dividing the dataset upfront to speed-up the distance computation.
- Bucketing, KD Trees, Ball Trees, Locality Sensitive Hashing (LSH).

Dimensionality Reduction: Transforming the data into a low-dimensional space without losing meaningful properties of the original data.
- Feature Extraction(Projection)
- Feature Selection
- Editing/ Pruning
- Creating Prototypes.

Predictive Performance: The predictive performance of a k-NN model depends on several factors including.
- The right value of k
- Distance Measures
- Scaling Features

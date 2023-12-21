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

ML Problem: Making ML Softwrae to recognize digits or numbers from the given image by training it with labeled data of numbers.

Data Collection
- To acquire the labeled data for this project we can use MNIST Dataset or it's new version EMNIST Dataset which contains 240,000 training images and 40,000 testing images because every user will write numbers differently so we need different types of habdwriting images of numbers.

Understanding the Data
- The Data contains greyscale images and corresponding labels.
- Image size: 28 x 28 pixels.
- In preprocessing step the color of images may be different colors but to make it easier to train the machine the images are converted to black & white images.
- In greyscale the value 0 is pitch black and the value 255 is white, based on the values the shade will change.
- The machine will take this image as matrix with values of greyscale.

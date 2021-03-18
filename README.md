# Project 2: Ethical Facial Recognition
<p>Much of the power of AI today comes from the use of data-driven deep learning to train increasingly accurate models by using growing amounts of data. However, the strength of these techniques can also be a weakness. The AI systems learn what they’re taught, and if they are not taught with robust and diverse datasets, accuracy and fairness could be at risk.</p><br><br>
<p>The challenge in training AI is manifested in a very apparent and profound way with facial recognition technology. There can be difficulties in making facial recognition systems that meet fairness expectations.The heart of the problem is not with the AI technology itself, per se, but with how the AI-powered facial recognition systems are trained. For the facial recognition systems to perform as desired – and the outcomes to become increasingly accurate –  training data must be diverse and offer a breadth of coverage</p>
<br><br>
<p>Team has gone through many research paper while formulating the solution to the problem. Please refere [Understanding Data](https://github.com/dsp-uga/rhoam-p2/wiki/Understanding-Data) for more reference.</p><br><br>

<p> To deal with the Ethics of the data we have also created a Ethics Checklist and answered all the Ethics questions on our Ethics wiki page. Please visit
  [Ethics](https://github.com/dsp-uga/rhoam-p2/wiki/Ethics) for more reference</p>
  
# Model
<p>The model takes advantage of both RDD and Dataframe preprocessing in order to load the dataset in a paralleized computing manner. The images are loaded in and immediately flipped to proper RGB color organization as the byte data loads in BGR color order. Once in RGB, the images are also cropped in the parallelized environment before being converted to Numpy arrays in memory. These Numpy arrays are loaded into a convolutional Keras deep neural network. The model analyzes spatial data to label images and predict either a male or female representative.</p>

# Results and Issues
<p>The model had no issues with the small training and testing set achieving around a 70% accuracy on very few epochs of training. The model ran into issues with the large dataset as the native spark.load.format('image') method uses a Java process that does not support CMYK images that are present in the larger datasets. This caused the model to be forced to use linear loading in order to check each and every image for the unsupported image format before loading into the model. Of the 650,000 images in the training set, around 30 are of the CMYK format which is very unlucky for our model. We were unable to achieve an accuracy for the larger testing sets as linear preprocessing requires over 24 hours for data loading.</p>

# YouTube Emotions Detection

## Overview
* The project deals with the idea of getting a bunch of comments from YouTube and classifying them based on the emotions they express
* I used several techniques such as transfer learning, noisy pre-labeling, GloVe embeddings
* I created a Flask API to serve the models and deployed it on heroku through docker
* I used both the Twitter and YouTube API to create the datasets
* I built a website to access the API that serves the model easily


<br>


## **Polarity**
In ```polarity``` I used the dataset from [Kaggle](https://www.kaggle.com/kazanova/sentiment140) where tweets are classified as positive or negative, cleaned that data and then built a neural network using transfer learning and tensorflow hub. In particular, I used [USE(Universal Sentence Encoder)](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) to deal with text.
The classification of comments was between 3 classes (positive, negative, neutral) based on different theresholds.
Finally, I built a Flask API and deployed it on Heroku using Docker.


<br>


## **Emotion**

### Twitter
Here I followed the idea explained in this [article](https://towardsdatascience.com/building-better-ai-apps-with-tf-hub-88716b302265).
Basically we use the Twitter API to get the text from tweets based on different hashtags. Then we can label that text with the hashtag that was used in the tweet. For example, I searched for hashtags such as #happy, #advice, #love, and labelled the examples accordingly to the hashtags.
After building the dataset, I processed the data and coded the model.

### Semi-Supervised
Since I wasn't satisfied by the results obtained using the method described above and polarity has only 3 classes, here I built the dataset from scratch using the YouTube API: I took around 130 examples and labelled them manually.
The classes in which the comments are classified are: positive (joy/happiness/amusement/love), negative(anger/hate/disgust), neutral/other, constructive feedback/idea and sadness.
After processing the data, I build the model using transfer learning with [USE (Universal Sentence Encoder)](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) to handle text and [an implementation](https://github.com/bckenstler/CLR) of cyclical learning described in this [paper](https://arxiv.org/abs/1506.01186). 
Then to expand the dataset I used the API again to gather more unlabelled data and tried 2 different techniques:
1. made the model predict and label the unlabelled data if the accuracy of the prediction was high enough, then trained the model on the new data, and kept repeating the process until the unlabelled dataset was not empty -> this method didn't work that well
2. used noisy pre-labelling to label the entire dataset and then manually check the prediction of the model (which is what I'm currently doing)

Finally I built a Flask API to serve the model and, using Docker, deployed it on Heroku.

Currently I'm checking the dataset built by the model and experimenting with different architectures.


<br>


## Resources
* For tensorflow hub and USE: https://curiousily.com/posts/sentiment-analysis-with-tensorflow-2-and-keras-using-python/, https://towardsdatascience.com/building-better-ai-apps-with-tf-hub-88716b302265 and https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
* [For the Twitter dataset](https://medium.com/neuronio/from-sentiment-analysis-to-emotion-recognition-a-nlp-story-bcc9d6ff61ae)
* For cyclical learning rates: https://arxiv.org/abs/1506.01186 and https://github.com/bckenstler/CLR
* 

## Packages
**Python version**: 3.8                                   
**Packages**:
```
pip install pandas
pip install numpy  
pip install matplotlib
pip install tensorflow
pip install seaborn
pip install tensorflow-hub
pip install tensorflow-text
pip install flask
```
# YouTube Emotions Detection

## Overview
* The project deals with the idea of getting a bunch of comments from YouTube and classifying them based on the emotions they express
* I used several techniques, such as transfer learning, noisy pre-labeling, GloVe embeddings, etc
* I created a Flask API to serve the model and, through docker, deployed it on Heroku
* To create the datasets I used both the Twitter and YouTube API
* To have easy access to the API that serves the model, I built a [website](https://youtube-emotions.netlify.app/) using React.js


<br>


## **Polarity**
In ```polarity``` I used a [dataset](https://www.kaggle.com/kazanova/sentiment140) from Kaggle where tweets are classified as positive or negative, cleaned the data, and then built a neural network using transfer learning and TensorFlow hub. More specifically, I used the [Universal Sentence Encoder (USE)](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) to deal with text.
The classification of comments was between 3 classes (positive, negative, neutral) based on different thresholds.
Finally, I built a Flask API with gunicorn and deployed it on Heroku using Docker.


<br>


## **Emotion**

### Twitter
Here I followed the idea explained in this [article](https://towardsdatascience.com/building-better-ai-apps-with-tf-hub-88716b302265).
Basically, we use the Twitter API to get the text from tweets based on different hashtags. Then we can label that text with the hashtag that was used in the tweet. For example, I searched for hashtags such as #happy, #advice, #love, and labeled the examples accordingly to the hashtags.
After building the dataset, I processed the data and coded the model (using a similar architecture as the one from ```polarity```).

### Semi-Supervised
Since I wasn't satisfied by the results obtained using the methods described above, here I built the dataset from scratch using the YouTube API: I took around 130 examples and labeled them manually.

The classes I chose are: *positive* (joy/happiness/amusement/love), *negative* (anger/hate/disgust), *neutral/other*, *constructive feedback/idea* and *sadness*.

After processing the data, I built the model using transfer learning with [USE](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) to handle the text and [an implementation](https://github.com/bckenstler/CLR) of cyclical learning described in this [paper](https://arxiv.org/abs/1506.01186).             
Then to expand the dataset I used the YouTube API again to gather more unlabelled data and tried 2 different techniques:
1. I made the model predict and label the unlabelled data if the accuracy of the prediction was high enough, then trained the model on the new data and kept repeating the process until the unlabelled dataset was not empty. This method didn't work that well

2. I used noisy pre-labeling to label the entire dataset and then manually check the predictions of the model

Finally, I built a Flask API with gunicorn to serve the model and, using Docker, deployed it on Heroku.

Currently, I'm checking the dataset built by the model and experimenting with different architectures.


<br>


## Resources
* For TensorFlow hub and USE: https://curiousily.com/posts/sentiment-analysis-with-tensorflow-2-and-keras-using-python/, https://towardsdatascience.com/building-better-ai-apps-with-tf-hub-88716b302265, and https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
* [For the Twitter dataset](https://medium.com/neuronio/from-sentiment-analysis-to-emotion-recognition-a-nlp-story-bcc9d6ff61ae)
* For cyclical learning rates: https://arxiv.org/abs/1506.01186 and https://github.com/bckenstler/CLR


<br>


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
pip install flask-cors
pip install requests
pip install gunicorn
pip install scikit-learn
```
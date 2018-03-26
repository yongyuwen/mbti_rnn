# mbti_rnn
## RNN to predict MBTI from Tweets

This repo contains a custom many-to-one TensorFlow RNN built specifically for the classification of MBTI types from Tweets.
Raw dataset can be downloaded from https://www.kaggle.com/datasnaek/mbti-type. (Due to GitHub upload limit, actual processed dataset and word embedding are not included in this repo). 

RNN supports several cell types (Basic RNN, GRUs, LSTMs), multiple layers, as well as dropout regularization.

### Requirements
This RNN **requires** the use of a pre-trained word embedding, as well as labels to be one-hot encoded. However, the raw code can be easily modified to remove these requirements. 

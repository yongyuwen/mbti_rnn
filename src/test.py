from rnn import RNN, data_pipeline
import pickle
import tensorflow as tf

#Load data
data, embedding = pickle.load( open( "256.p", "rb" ) )
training_data, validation_data = data

num_steps = len(training_data[0][0])
num_classes = len(training_data[1][0])
assert num_classes == 16
epochs = 40
checkpoint = "./tmp/rnn_model_3.ckpt"
save="./saves/LSTM_BatchSize96_DropOutHalf"

#Create Data Pipeline
pipeline = data_pipeline(batch_size=64, shuffle_buffer_size=100000)

#Create network
net = RNN('GRU', state_size=427, num_steps=num_steps, num_layers=1, num_classes=num_classes, embedding=embedding, build_with_dropout=True, dropout=0.3)

#Train the model
with tf.Session() as sess:
    net.train(sess, epochs=epochs, learning_rate= 3e-4, pipeline=pipeline, training_data=training_data, validation_data=validation_data,
              checkpoint=checkpoint, save=None)

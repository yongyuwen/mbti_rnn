"""rnn.py
~~~~~~~~~~~~~~
Written by Yong Yu Wen, 2018

(Built using tensorflow-gpu 1.6.0, cuda 9.0 and cuDNN 7.05)

A TensorFlow-based many-to-one recurrent neural network specifically
for the classification of MBTI types based on social media posts.
Raw un-processed dataset used for this task can be found at
https://www.kaggle.com/datasnaek/mbti-type

Supports several cell types (Basic RNN, GRUs, LSTMs), multiple layer,
training with word embeddings, as well as dropout regularization.

This program incorporates ideas from Denny Britz and Spitis (Github display name)
and their websites http://www.wildml.com and https://r2rt.com
"""

import tensorflow as tf
import time
import pickle


class RNN(object):
    def __init__(self, cell_type, state_size, num_steps, num_layers,
                 num_classes, embedding=None, build_with_dropout=False, dropout=1.0):
        """
        Creates the RNN object
        :param cell_type: Type of RNN cell. Supports Basic RNN, GRUs and LSTMs
        :param state_size: Number of hidden states
        :param num_steps: Number of time steps
        :param num_layers: Number of layers
        :param num_classes: Number of classes in the output
        :param embedding: Word embedding
        :param build_with_dropout: Whether to use dropout in the RNN
        :param dropout: Dropout keep probability
        """
        self.x = tf.placeholder(tf.int32, [None, num_steps], name='input_placeholder')
        self.y = tf.placeholder(tf.int32, [None, num_classes], name='labels_placeholder')
        with tf.name_scope("embedding"):
            self.embeddings = tf.get_variable(name="embeddings", shape=embedding.shape,
                                              initializer=tf.constant_initializer(embedding), trainable=True)
        self.state_size = state_size
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.build_with_dropout = build_with_dropout
        self.dropout = dropout
        self.cell_type = cell_type
        self.cell = self._make_MultiRNNCell()
        self.saver = tf.train.Saver()

    def _make_cell(self):
        """
        Private function to create RNN cell. Required for TensorFlow's MultiRNNCell function
        """
        if self.cell_type == 'GRU':
            cell = tf.nn.rnn_cell.GRUCell(self.state_size)
        elif self.cell_type == 'LSTM':
            cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        else:
            cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
        if self.build_with_dropout:
            return tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout)
        else:
            return cell

    def _make_MultiRNNCell(self):
        """
        Private function to create multi-layer RNNs
        """
        cell = tf.nn.rnn_cell.MultiRNNCell([self._make_cell() for _ in range(self.num_layers)])

        if self.build_with_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
        return cell


    def train(self, sess, epochs, learning_rate, pipeline, training_data, validation_data,
              checkpoint=None, save=None):
        """
        Trains the neural network using the Adam Optimizer (by default)
        :param sess: TensorFlow Session
        :param epochs: Number of epochs
        :param learning_rate: Learning rate for the optimizer
        :param pipeline: Pipeline object to feed data into the network for training
        :param training_data: Training dataset (in Numpy array format, labels one-hot encoded)
        :param validation_data: Validation dataset (in Numpy array format, labels one-hot encoded)
        :param checkpoint: Location to save model checkpoint
        :param save: Location to save trained model
        """
        rnn_inputs = tf.nn.embedding_lookup(self.embeddings, self.x)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(self.cell, rnn_inputs, dtype=tf.float32) #initial_state=init_state
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.state_size, self.num_classes])
            b = tf.get_variable('b', [self.num_classes], initializer=tf.constant_initializer(0.0))
        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
        last = tf.reshape(rnn_outputs[-1], [-1, self.state_size])
        predictions = (tf.matmul(last, W) + b)

        y_reshaped = tf.reshape(self.y, [-1, self.num_classes])
        total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y_reshaped))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        #model evaluation
        correct_prediction=tf.equal(tf.argmax(predictions,1),tf.argmax(y_reshaped,1))
        model_accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        #~~~~~~~~~~~~~~Training of the actual dataset~~~~~~~~~~~~~~~~~~
        training_x, training_y = training_data
        validation_x, validation_y = validation_data

        sess.run(tf.global_variables_initializer())

        if save:
            try:
                self.saver.restore(sess, save)
                print("Save restored \n")
            except:
                print("No Save found. Running new training cycle")
                
        start_time = time.time() #Track time taken for model

        training_accuracies = []
        validation_accuracies = []
        
        for epoch in range(epochs):
            sess.run(pipeline.iterator_train.initializer, feed_dict={pipeline.features_placeholder: training_x,
                                                                     pipeline.labels_placeholder: training_y})
            training_loss = 0
            steps = 0
            training_state = None
            avg_loss = []
            accuracies = []
            if epoch >0 and checkpoint:
                self.saver.save(sess, checkpoint)
                print("Saving checkpoint for epoch", epoch)

            while True:
                try:
                    steps += 1
                    batch_x, batch_y = sess.run(pipeline.next_element_train)
                    feed_dict={self.x: batch_x, self.y: batch_y}

                    training_loss_, _, accuracy = sess.run([total_loss,
                                                            train_step,
                                                            model_accuracy],
                                                           feed_dict)
                    avg_loss.append(training_loss_)
                    accuracies.append(accuracy)
                    if steps%100 == 0:
                        print("Avg training_loss_ for Epoh {} step {} =".format(epoch, steps), tf.reduce_mean(avg_loss).eval())
                        avg_loss = []
                        accuracies = []

                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    print("Avg accuracy for Epoch {} step {} =".format(epoch, steps), tf.reduce_mean(accuracies).eval())
                    training_accuracies.append(tf.reduce_mean(accuracies).eval())
                    accuracies = []
                    break

            #Print Validation Accuracy per Epoch
            sess.run(pipeline.iterator_val.initializer, feed_dict={pipeline.features_placeholder: validation_x,
                                                                   pipeline.labels_placeholder: validation_y})
            val_accuracies = []
            while True:
                try:
                    val_x, val_y = sess.run(pipeline.next_element_val)
                    feed_dict={self.x: val_x, self.y: val_y}
                    accuracy = sess.run(model_accuracy, feed_dict)
                    val_accuracies.append(accuracy)
                except tf.errors.OutOfRangeError:
                    print("Validation Accuracy for epoch {} is ".format(epoch), tf.reduce_mean(val_accuracies).eval())
                    validation_accuracies.append(tf.reduce_mean(val_accuracies).eval())
                    break

        end_time = time.time()
        total_time = end_time - start_time
        print("Finished training network.")
        print("Time to train network: {}s".format(total_time))
        pickle.dump((training_accuracies, validation_accuracies), open( "accuracies.p", "wb" ) )
        print("Pickled Accuracies")

        if save:
            self.saver.save(sess, save)
            print("Model is saved in", save)

        


class data_pipeline(object):
    def __init__(self, batch_size, shuffle_buffer_size):
        """
        Pipeline Object to shuffle and split data into batches before feeding into neural network
        :param batch_size: Integer Value of the desired batch size
        :param shuffle_buffer_size: Buffer Size for shuffling dataset. See TensorFlow docs for mroe information
        """
        self.features_placeholder = tf.placeholder(tf.int32)
        self.labels_placeholder = tf.placeholder(tf.int32)

        self.dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))

        #Train input pipeline
        self.dataset_train = self.dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size)
        self.iterator_train = self.dataset_train.make_initializable_iterator()
        self.next_element_train = self.iterator_train.get_next()

        #Val input pipeline
        self.dataset_val = self.dataset.batch(batch_size)
        self.iterator_val = self.dataset_val.make_initializable_iterator()
        self.next_element_val = self.iterator_val.get_next()



import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks


def main():
    mnist = ks.datasets.mnist
    (x_train_r, y_train_r), (x_test_r, y_test_r) = mnist.load_data()

    num_classes = 10

    x_train = preprocess_input(x_train_r)
    y_train = preprocess_output(num_classes, y_train_r)

    x_test = preprocess_input(x_test_r)
    y_test = preprocess_output(num_classes, y_test_r)

    x = tf.placeholder(dtype="float", shape=[x_train.shape[0], None], name="input")
    y = tf.placeholder(dtype="float", shape=[y_train.shape[0], None], name="output")

    seed = 5
    with tf.name_scope('hyper_parameters'):

        learning_rate = 0.001
        beta = 0.001
        num_epochs = 10

        initializer = tf.contrib.layers.xavier_initializer(seed=seed)

        layer_params = [
            (400, tf.nn.relu, "1"),
            (200, tf.nn.relu, "2"),
            (100, tf.nn.relu, "3"),
            (10, identity, "4")
        ]

        tf.summary.scalar('number of epochs', num_epochs)
        tf.summary.scalar('learning rate', learning_rate)
        tf.summary.scalar('beta', beta)

        for n_outputs, activation, name_postfix in layer_params:
            tf.summary.scalar("numer of hidden units in layer {}".format(name_postfix), n_outputs)
            # tf.summary.scalar("numer of hidden units in layer {}".format(name_postfix), n_outputs)

    A, W, B = [], [], []
    a_prev = x
    for n_outputs, activation, name_postfix in layer_params:
        with tf.name_scope("Weights"):
            a, w, b = make_fc_layer(name_postfix, a_prev, n_outputs, activation, initializer)
            A.append(a)
            W.append(w)
            B.append(b)
            tf.summary.tensor_summary("W{}".format(name_postfix), w)
            a_prev = a


    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.transpose(A[-1]), labels=tf.transpose(y)))

    regularized_cost = tf.reduce_mean(cost + beta * tf.contrib.layers.apply_regularization(tf.nn.l2_loss, W))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(regularized_cost)

    init = tf.global_variables_initializer()

    costs = []
    with tf.Session() as session:
        writer = tf.summary.FileWriter("tmp/log/", session.graph)
        merged = tf.summary.merge_all()

        session.run(init)

        print("x: {} , y: {}".format(x_train.shape, y_train.shape))
        start = datetime.datetime.now()

        for i in range(num_epochs):
            epoch_cost = 0

            summary, _, epoch_cost = session.run([merged, optimizer, cost], feed_dict={x: x_train, y: y_train})
            writer.add_summary(summary)
            costs.append(epoch_cost)
            if i % 10 == 0:
                mean_time = (datetime.datetime.now() - start) / 10
                print("epoch {} cost: {} mean time {}".format(i, epoch_cost, mean_time))
                start = datetime.datetime.now()

        with tf.name_scope("Results"):
            correct_prediction = tf.equal(tf.argmax(A[-1]), tf.argmax(y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            train_accuracy = accuracy.eval({x: x_train, y: y_train})
            test_accuracy = accuracy.eval({x: x_test, y: y_test})

            tf.summary.scalar('train accuracy', train_accuracy)
            tf.summary.scalar('test accuracy', test_accuracy)
            tf.summary.scalar('costs', costs)


        print("Train accuracy = {train}\nTest accuracy = {test}".format(train=train_accuracy, test=test_accuracy))

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        writer.close()


def preprocess_output(num_classes, y_train_r):
    with tf.Session() as session:
        n_c = tf.constant(num_classes, name="num_classes")
        y_train = session.run(tf.one_hot(y_train_r, n_c, axis=0))
    return y_train


def preprocess_input(x_train_r):
    x_train_f = flatten(x_train_r)
    x_train = x_train_f / 255
    return x_train


def identity(input):
    return input


def make_fc_layer(name_postfix, input, n_outputs, activation, initializer):
    w, b = initialize_layer_params(
        name_postfix=name_postfix,
        n_inputs=input.shape[0],
        n_outputs=n_outputs,
        initializer=initializer)
    a = activation(tf.add(tf.matmul(w, input), b))
    return a, w, b


def initialize_layer_params(name_postfix, n_inputs, n_outputs, initializer):
    w = tf.get_variable(
        name="W" + name_postfix,
        shape=[n_outputs, n_inputs],
        initializer=initializer)
    b = tf.get_variable(
        name="B" + name_postfix,
        shape=[n_outputs, 1],
        initializer=tf.zeros_initializer()
    )
    return w, b


def flatten(raw_data):
    flattened_data = raw_data.reshape(raw_data.shape[0], -1).T
    return flattened_data


if __name__ == '__main__':
    main()

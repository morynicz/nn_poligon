import datetime

import tensorflow as tf
import tensorflow.keras as ks


def transformation(image, label):
    x = tf.reshape(tf.cast(image, tf.float32), (784, 1))
    y = tf.one_hot(tf.cast(label, tf.uint8), 10)
    return dict(image=x), y


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.map(transformation).shuffle(1000).repeat().batch(batch_size)
    return dataset # .make_one_shot_iterator().get_next()


def eval_input_fn(features, labels):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.map(transformation).batch(features.shape[0])
    return dataset # .make_one_shot_iterator().get_next()


def model_fn(features, labels, mode, params):
    # definition
    with tf.name_scope('hyper_parameters'):
        tf.summary.scalar('learning_rate', params["learning_rate"])
        tf.summary.scalar('beta', params["beta"])
        tf.summary.scalar('num_epochs', params["num_epochs"])
        tf.summary.scalar('batch_size', params["batch_size"])

    net = tf.feature_column.input_layer(features, params['feature_columns'])

    for units, name, activation, initializer, weight_regularizer in params["layer_params"]:
        with tf.name_scope("dense_layer"):
            net = tf.layers.dense(
                net,
                units=units,
                activation=None,
                kernel_initializer=initializer,
                kernel_regularizer=weight_regularizer,
                use_bias=False,
                name=name + "_dense")
            net = tf.layers.batch_normalization(
                net,
                axis=1,
                training=(mode == tf.estimator.ModeKeys.TRAIN),
                center=True,
                scale=False,
                name=name + "_batch_norm")
            net = activation(net, name + "_activation")
    with tf.name_scope("final_layer"):
        logits = tf.layers.dense(net, params['n_classes'], activation=None, name="output_layer")

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    accuracy = tf.metrics.accuracy(tf.argmax(labels, 1), predictions=predicted_classes, name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]).minimize(loss,
                                                                                       global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)


def preprocess_input(x_train_r):
    x_train_f = flatten(x_train_r)
    x_train = x_train_f / 255
    return x_train


def flatten(raw_data):
    flattened_data = raw_data.reshape(raw_data.shape[0], -1).T
    return flattened_data


def preprocess_output(num_classes, y_train_r):
    with tf.Session() as session:
        n_c = tf.constant(num_classes, name="num_classes")
        y_train = session.run(tf.one_hot(y_train_r, n_c, axis=0))
    return y_train


def main(argv):
    mnist = ks.datasets.mnist
    (x_train_r, y_train_r), (x_test_r, y_test_r) = mnist.load_data()

    feature_columns = [tf.feature_column.numeric_column(key='image', shape=[784])]

    learning_rate = 0.001
    beta = 0.0
    num_epochs = 1000
    batch_size = 512
    num_classes = 10
    seed = 5

    initializer = tf.contrib.layers.xavier_initializer(seed=seed)
    l2_regularizer = tf.contrib.layers.l2_regularizer(beta)

    layer_params = [
        (400, "1", tf.nn.relu, initializer, l2_regularizer),
        (200, "2", tf.nn.relu, initializer, l2_regularizer),
        (100, "3", tf.nn.relu, initializer, l2_regularizer)
    ]

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'feature_columns': feature_columns,
            'n_classes': num_classes,
            'layer_params': layer_params,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "beta": beta,
            "num_epochs": num_epochs
        },
        model_dir="models/{date}_{lr}_{ep}_{bs}_{beta}_{sizes}".format(
            date=datetime.datetime.now(),
            lr=learning_rate,
            ep=num_epochs,
            bs=batch_size,
            beta=beta,
            sizes="#".join([str(layer) for layer, _, _, _, _ in layer_params]))
    )

    # train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(x_train_r, y_train_r, batch_size), max_steps=num_epochs)
    # eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(x_test_r, y_test_r))

    # tf.estimator.train_and_evaluate(estimator=classifier, train_spec=train_spec, eval_spec=eval_spec)

    classifier.train(input_fn=lambda: train_input_fn(x_train_r, y_train_r, batch_size),
                     max_steps=num_epochs)

    with tf.name_scope('evaluation'):
        print("training finished")
        eval_train = classifier.evaluate(input_fn=lambda: eval_input_fn(x_train_r, y_train_r))
        summary = tf.summary.scalar('train_success_rate', eval_train["accuracy"])
        print('\nTrain set accuracy: {accuracy:0.3f}\n'.format(**eval_train))
        eval_test = classifier.evaluate(input_fn=lambda: eval_input_fn(x_test_r, y_test_r))
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_test))
        summary2 = tf.summary.scalar('test_success_rate', eval_test["accuracy"])
        summary = tf.summary.merge([summary, summary2])

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

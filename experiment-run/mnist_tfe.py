"""
Based off of https://www.tensorflow.org/tutorials/estimators/cnn
"""

import numpy as np
import tensorflow as tf


# Our application logic will be added here
def main(argv=[]):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load data
    train, test = tf.keras.datasets.mnist.load_data()

    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    #train_data = mnist.train.images # Returns np.array
    train_data, train_labels = train
    train_data = np.asarray(train_data, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int32)

    eval_data, eval_labels =  test
    eval_data = np.asarray(eval_data, dtype=np.float32)
    eval_labels = np.asarray(eval_labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        #model_fn=cnn_model_fn_old,
        #model_dir="/tmp/mnist_convnet_model",
    )

    tf.contrib.estimator.add_metrics(mnist_classifier, custom_metric)
    # Set up logging for predictions
    #tensors_to_log = {"probabilities": "softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(
    #    tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
    )
        #hooks=[logging_hook])
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[4, 4],
            padding="same",
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def custom_metric(labels, predictions):
    # This function will be called by the Estimator, passing its predictions.
    # Let's suppose you want to add the "mean" metric...

    # Accessing the class predictions (careful, the key name may change from one canned Estimator to another)
    predicted_classes = predictions["class_ids"]  

    # Defining the metrix (value and update tensors):
    custom_metric = tf.metrics.mean(labels, predicted_classes, name="custom_metric")

    # Returning as a dict:
    return {"custom_metric": custom_metric}


if __name__ == "__main__":
    #tf.app.run()
    main()

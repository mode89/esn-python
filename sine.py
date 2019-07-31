#!/usr/bin/env python3

import math
import tensorflow as tf

HIDDEN_STATE_SIZE = 16

def main():
    normalInitializer = tf.glorot_normal_initializer(seed=0)

    hiddenState = tf.get_variable(
        name="hidden_state",
        shape=HIDDEN_STATE_SIZE,
        trainable=False,
        initializer=tf.constant_initializer(0.0))
    weights = tf.get_variable(
        name="hidden_weights",
        shape=(HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE),
        trainable=False,
        initializer=tf.initializers.orthogonal(seed=0))
    feedbackWeights = tf.get_variable(
        name="feedback_weights",
        shape=HIDDEN_STATE_SIZE,
        trainable=False,
        initializer=normalInitializer)

    denseLayer = tf.layers.Dense(1, tf.tanh)

    outputs = denseLayer(tf.reshape(hiddenState, (1, HIDDEN_STATE_SIZE)))

    referenceOutputs = tf.placeholder(
        tf.float32, shape=1, name="reference_outputs")
    loss = tf.losses.absolute_difference(
        tf.reshape(referenceOutputs, (1, 1)), outputs)
    optimizer = tf.train.AdamOptimizer()
    trainOp = optimizer.minimize(loss)

    nextHiddenState = tf.linalg.matvec(weights, hiddenState) + \
        referenceOutputs * feedbackWeights
    nextHiddenState = tf.tanh(nextHiddenState)
    updateOp = tf.assign(hiddenState, nextHiddenState)
    hiddenStateMean = tf.reduce_mean(hiddenState)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(hiddenStateMean))
        for i in range(100000):
            output = math.sin(i * 0.1)
            results = session.run({
                "loss": loss,
                "train": trainOp,
            }, {
                referenceOutputs: [output]
            })
            print(results["loss"])
            session.run(updateOp, {
                referenceOutputs: [output]
            })

if __name__ == "__main__":
    main()

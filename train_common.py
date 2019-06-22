from datetime import datetime
import os.path
import math
import re
import time

import numpy as np
import tensorflow as tf

# np.set_printoptions(precision=1, threshold=5, linewidth=500, edgeitems=2)

def _tower_loss(scope, images, labels, network, dataset, num_classes, top_name, tf_training, kargs):
    logits = network.network(images, num_classes=num_classes, scope=top_name, 
        is_training=tf_training, kargs=kargs)

    total_loss, re_loss = network.loss(scope, logits, labels)
    metric_op = network.metric_op(logits, labels)

    return total_loss, re_loss, metric_op


def _average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def get_global_step(store_model_path):
    ckpt = tf.train.get_checkpoint_state(store_model_path)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])), trainable=False)
    else:
        global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
    return global_step

def get_lr_and_max_steps(examples_per_epoch, batch_size, num_gpus, lr_decay_factor, epochs_per_decay, 
    initial_lr, global_step, staircase, max_epochs):
    num_batches_per_epoch = (examples_per_epoch / batch_size / num_gpus)
    if isinstance(lr_decay_factor, float):
        decay_steps = int(num_batches_per_epoch * epochs_per_decay)
        lr = tf.train.exponential_decay(initial_lr, global_step, decay_steps, lr_decay_factor, staircase=staircase)
        max_steps = int(max_epochs * num_batches_per_epoch)
    elif isinstance(lr_decay_factor, list):
        boundaries = [(num_batches_per_epoch * epoch) for epoch in epochs_per_decay]
        vals = [initial_lr * decay for decay in lr_decay_factor]
        lr = tf.train.piecewise_constant(global_step, boundaries, vals)
        max_steps = int(max_epochs * num_batches_per_epoch)
    else:
        raise ValueError("unknown lr policy")
    return lr, max_steps

def get_ops(opt, tf_training, network, dataset, num_classes, top_name, train_args):
    num_gpus = train_args.num_gpus
    lr = train_args.learning_rate
    global_step = train_args.global_step
    train_dataset, test_dataset = train_args.data_queue

    tower_grads = []
    top_ks = []

    ## test 
    # images, labels = dataset.distorted_inputs(128, **{"padding": True, "bright": True, "mirroring": True})
    # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
    #                 [images, labels], capacity=2 * 1)

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                    # Dequeues one batch for the GPU

                    image_batch, label_batch = tf.cond(tf_training, train_dataset.get_next, test_dataset.get_next)
                    # image_batch, label_batch = batch_queue.dequeue()
                    loss, re_loss, top_k_op = _tower_loss(scope, image_batch, label_batch, network, 
                        dataset, num_classes, top_name, tf_training, train_args)
                    tf.get_variable_scope().reuse_variables()
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
                    top_ks.append(top_k_op)
    top_k_op = tf.reduce_sum(top_ks)
    grads = _average_gradients(tower_grads)
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    train_op = apply_gradient_op

    summary_op = tf.summary.merge(summaries)
    return train_op, summary_op, loss, re_loss, top_k_op

def run_op(ops, tf_training, store_model_path, train_args):
    """

    """
    train_batch_size = train_args.train_batch_size
    test_batch_size = train_args.test_batch_size
    num_gpus = train_args.num_gpus
    global_step = train_args.global_step
    max_steps = train_args.max_steps
    examples_per_epoch_for_test = train_args.examples_per_epoch_for_test


    train_op, summary_op, loss, re_loss, top_k_op = ops
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    init = tf.global_variables_initializer()
    assign_init = tf.get_collection("init")
    # gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(
            # gpu_options=gpu_options,
            allow_soft_placement=True,
            log_device_placement=False))
    tf.train.start_queue_runners(sess=sess)

    sess.run(init)
    sess.run(assign_init)

    ckpt = tf.train.get_checkpoint_state(store_model_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')

    summary_writer = tf.summary.FileWriter(store_model_path, sess.graph)

    train_true_count = 0
    max_test_acc = 0.0
    start_time = time.time()
    start_step = int(tf.train.global_step(sess, global_step))
    sess.graph.finalize()
    
    for step in range(start_step, max_steps):
        if step % 100 == 0:
            duration = time.time() - start_time
            _, summary_str, loss_value, re_loss_value, train_predictions = sess.run(ops, feed_dict={tf_training: True})
            num_examples_per_step = train_batch_size * num_gpus
            if step == start_step:
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / num_gpus
            else:
                examples_per_sec = 100 * num_examples_per_step / duration
                sec_per_batch = duration / num_gpus / 100

            train_true_count = np.sum(train_predictions)
            # train_true_count = 0
            train_acc = train_true_count / num_examples_per_step
            format_str = (
                '%s: step (%d)%d, loss = %.2f, regularization_loss = %.2f, train_acc = %.4f (%.1f examples/sec; %.3f sec/batch)'
                ) % (datetime.now(), step, int(tf.train.global_step(sess, global_step)),
                loss_value, re_loss_value, train_acc, examples_per_sec, sec_per_batch)
            print (format_str)
            summary_writer.add_summary(summary_str, step)

            train_true_count = 0
            start_time = time.time()

        else:
            sess.run([train_op], feed_dict={tf_training: True})
            # print("%s: step %d" % (datetime.now(), step))

        # Save the model checkpoint periodically.
        if step != 0 and step % 1000 == 0 or (step + 1) == max_steps:
            test_true_count = 0
            num_iter = int(math.ceil(examples_per_epoch_for_test / (test_batch_size * num_gpus)))
            for i in range(num_iter):
                loss_val, test_predictions = sess.run([loss, top_k_op], feed_dict={tf_training: False})
                test_true_count += np.sum(test_predictions)
            test_acc = test_true_count / (num_iter * test_batch_size * num_gpus)
            format_str = ('%s: Test, loss = %.2f, test_acc = %.4f') % (datetime.now(), loss_val, test_acc)
            print(format_str)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                checkpoint_path = os.path.join(store_model_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=int(tf.train.global_step(sess, global_step)))
    summary_writer.close()




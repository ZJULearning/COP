import os
import tensorflow as tf

label_bytes = 2 # 2 for CIFAR100
height = 32
width = 32
depth = 3
image_bytes = height * width * depth
record_bytes = label_bytes + image_bytes
num_examples_for_train = 50000
num_examples_for_test = 10000
num_classes = 100

data_augmentation_args = {}

def _parse_one_record(record):
    record = tf.decode_raw(record, tf.uint8)
    label = tf.reshape(tf.cast(tf.strided_slice(record, [1], [label_bytes]), tf.int32), [])
    depth_major = tf.reshape(
        tf.strided_slice(record, [label_bytes],[label_bytes + image_bytes]),
        [depth, height, width])
    uint8image = tf.transpose(depth_major, [1, 2, 0])
    image = tf.cast(uint8image, tf.float32)
    return image, label

def _data_augmentation(image, label):
    global data_augmentation_args
    print("Use data_augmentation_args: ", data_augmentation_args)
    ## pad and crop
    if data_augmentation_args["padding"]:
        reshaped_image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
        image = tf.random_crop(reshaped_image, [height, width, depth])

    ## mirroring
    if data_augmentation_args["mirroring"]:
        image = tf.image.random_flip_left_right(image)

    ## random brightness and contrast
    if data_augmentation_args["bright"]:
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

    mean = data_augmentation_args["mean"]
    std = data_augmentation_args["std"]
    if isinstance(mean, list):
        mean = tf.reshape(mean, [1, 1, 3])
        std = tf.reshape(std, [1, 1, 3])
        image = (image - mean) / std
    else:
        image = (image - mean) / std
    return image, label

def train_input_fn(data_dir, batch_size, epochs, **kargs):
    global data_augmentation_args
    data_augmentation_args = kargs
    filenames = [os.path.join(data_dir, 'cifar-100-binary', 'train.bin')]
    for path in filenames:
        if not os.path.exists(path):
            raise ValueError(path + " not found")
    print("train data files lists: " + str(filenames))
    dataset = tf.data.Dataset.list_files(filenames, shuffle=True)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
        lambda name:tf.data.FixedLengthRecordDataset(name, record_bytes), cycle_length=4))
    # dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(2 * num_examples_for_train, epochs))
    dataset = dataset.map(_parse_one_record)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(_data_augmentation, batch_size))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

def test_input_fn(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'cifar-100-binary', 'test.bin')]
    print("test data files lists: " + str(filenames))
    for path in filenames:
        if not os.path.exists(path):
            raise ValueError(path + " not found")
    files = tf.data.Dataset.list_files(filenames, shuffle=False)
    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    dataset = dataset.repeat(-1)
    dataset = dataset.map(_parse_one_record)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        lambda image, label: ((image - 120.707) / 64.15, label),
        batch_size))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

if __name__ == '__main__':
    dataset = train_input_fn("/wangwenxiao/prune/data/", 128, 250)
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()
    index = 0;
    with tf.Session() as sess:
        while True:
            index += 1
            real_data = sess.run(data)
            if((index > 92000 and index % 10 == 0) or (index < 92000 and index % 1000 == 0)):
                print(index)



import os
import tensorflow as tf

height = 224
width = 224
depth = 3
image_bytes = height * width * depth
num_examples_for_train = 1281167
num_examples_for_test = 500
num_classes = 1001

def _parse_one_record(record, is_training, data_augmentation_args):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)

    features = tf.parse_single_example(record, feature_map)

    image = tf.image.decode_image(features['image/encoded'], channels=3)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    if is_training:
        image, label, bbox = _data_augmentation(image, label, bbox, data_augmentation_args)
    else:
        image, label, bbox = _process_for_eval(image, label, bbox, data_augmentation_args)

    return image, label

def _data_augmentation(image, label, bbox, data_augmentation_args):
    print("Use data_augmentation_args: ", data_augmentation_args)

    ## crop according to the bbox
    if data_augmentation_args["crop_bbox"]:
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image), bounding_boxes=bbox, min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1.0], max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)

        # print_op = tf.print(tf.shape(image), offset_y, offset_x, target_height, target_width)
        # with tf.control_dependencies([print_op]):
        image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image)

    ## resize and crop
    if data_augmentation_args["resize"]:
        new_size = tf.random_uniform([], minval=256, maxval=512+1, dtype=tf.int32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [new_size, new_size], align_corners=False)
        image = tf.squeeze(image)
        image = tf.image.random_crop(image, [height, width, depth])

    ## padding and crop
    if data_augmentation_args["padding"]:
        reshaped_image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
        image = tf.random_crop(reshaped_image, [height, width, depth])
        image = tf.cast(image, tf.float32)

    ## mirroring
    if data_augmentation_args["mirroring"]:
        image = tf.image.random_flip_left_right(image)

    ## random brightness and contrast(may be better without this)
    if data_augmentation_args["bright"]:
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    mean = data_augmentation_args["mean"]
    std = data_augmentation_args["std"]
    if isinstance(mean, list):
        mean = tf.reshape(mean, [1, 1, 3])
        std = tf.reshape(std, [1, 1, 3])
        image = (image - mean) / std
    else:
        image = (image - mean) / std
    image.set_shape([height, width, depth])

    return image, label, bbox

def _process_for_eval(image, label, bbox, data_augmentation_args):
    print("Use data_augmentation_args: ", data_augmentation_args)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [256, 256])
    image = tf.squeeze(image)
    image = tf.image.resize_image_with_crop_or_pad(image, height, width) # to 224*224

    mean = data_augmentation_args["mean"]
    std = data_augmentation_args["std"]
    if isinstance(mean, list):
        mean = tf.reshape(mean, [1, 1, 3])
        std = tf.reshape(std, [1, 1, 3])
        image = (image - mean) / std
    else:
        image = (image - mean) / std
    image.set_shape([height, width, depth])
    return image, label, bbox


def train_input_fn(data_dir, batch_size, epochs, **kargs):
    filenames =  [os.path.join(data_dir, 'train/train-%05d-of-01024' % i) for i in range(1024)]
    # print("train data files lists: " + str(filenames))
    for path in filenames:
        if not os.path.exists(path):
            raise ValueError(path + " not found")
    dataset = tf.data.Dataset.list_files(filenames, shuffle=True)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(100 * batch_size, epochs))

    dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda record: _parse_one_record(record, True, kargs), batch_size))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

def test_input_fn(data_dir, batch_size, **kargs):
    filenames = [os.path.join(data_dir, 'validation/validation-%05d-of-00128' % i)
            for i in range(128)]
    # print("test data files lists: " + str(filenames))
    for path in filenames:
        if not os.path.exists(path):
            raise ValueError(path + " not found")
    dataset = tf.data.Dataset.list_files(filenames, shuffle=False)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    dataset = dataset.repeat(-1)

    dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda record: _parse_one_record(record, False, kargs), batch_size))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset


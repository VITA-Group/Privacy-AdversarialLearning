import tensorflow as tf
from common_flags import COMMON_FLAGS

def distort_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    return image

def read_and_decode_videos(filename_queue, use_crop=True, use_random_crop=False,
                           use_center_crop=True, use_frame_distortion=False):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'video_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })

    video = tf.decode_raw(features['video_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    video = tf.reshape(video, [COMMON_FLAGS.DEPTH, COMMON_FLAGS.HEIGHT, COMMON_FLAGS.WIDTH, COMMON_FLAGS.NCHANNEL])
    if use_crop:
        if use_random_crop:
            video = tf.random_crop(video, [COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL])
        elif use_center_crop:
            video = tf.slice(video, [0, 4, 24, 0], [COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL])
        else:
            raise ValueError('Not specifying the cropping type')
    if use_frame_distortion:
        video_distort = []
        for i in range(video.get_shape()[0]):
            frame = distort_image(video[i])
            video_distort.append(frame)
        video = tf.stack(video_distort)
        print('Video distort shape is ', video.get_shape())
    video = tf.cast(video, tf.float32)
    return video, label

def inputs_videos(filenames, batch_size, num_epochs, num_threads, num_examples_per_epoch, shuffle=True, distort=False):
    if not num_epochs:
        num_epochs = None
    for filename in filenames:
        if not tf.gfile.Exists(filename):
            raise ValueError('Failed to find file: ' + filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=shuffle, name='string_input_producer'
        )
        video, label = read_and_decode_videos(filename_queue)

        print('Video shape is ', video.get_shape())
        print('Label shape is ', label.get_shape())
        if distort:
            video_distort = []
            for i in range(video.get_shape()[0]):
                frame = distort_image(video[i])
                video_distort.append(frame)
            video = tf.stack(video_distort)
            print('Video distort shape is ', video_distort.get_shape())
        min_fraction_of_examples_in_queue = 0.5
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        if shuffle:
            videos, labels = tf.train.shuffle_batch(
                [video, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                min_after_dequeue=min_queue_examples,
                name='batching_shuffling'
            )
        else:
            videos, labels = tf.train.batch(
                [video, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                allow_smaller_final_batch=False,
                name='batching_shuffling'
            )
        print('Videos shape is ', videos.get_shape())
        print('Labels shape is ', labels.get_shape())
        print('######################################################################')

    return videos, labels

def read_and_decode_images(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string),
    })

    image = tf.decode_raw(features['image_raw'], tf.float32)
    label = tf.decode_raw(features['label_raw'], tf.float32)
    image = tf.reshape(image, [COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL])
    image.set_shape([COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL])
    label.set_shape([COMMON_FLAGS.NUM_CLASSES_BUDGET])
    print(image.get_shape())
    print(label.get_shape())
    #image.set_shape([crop_height, crop_width, nchannel])
    return image, label

def inputs_images(filenames, batch_size, num_epochs, num_threads, num_examples_per_epoch, shuffle=True, distort=False):
    if not num_epochs:
        num_epochs = None
    for filename in filenames:
        if not tf.gfile.Exists(filename):
            raise ValueError('Failed to find file: ' + filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=shuffle, name='string_input_producer'
        )
        image, label = read_and_decode_images(filename_queue)
        if distort:
            image = distort_image(image)
        print('Image shape is ', image.get_shape())
        print('Label shape is ', label.get_shape())
        min_fraction_of_examples_in_queue = 0.5
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        if shuffle:
            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                min_after_dequeue=min_queue_examples,
                name='batching_shuffling'
            )
        else:
            images, labels = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                allow_smaller_final_batch=False,
                name='batching_shuffling'
            )
        print('Images shape is ', images.get_shape())
        print('Labels shape is ', labels.get_shape())
        print('######################################################################')

    return images, labels

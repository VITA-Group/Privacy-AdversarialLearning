import tensorflow as tf

depth, height, width, nchannel = 16, 120, 160, 3
crop_height, crop_width = 112, 112

def distort_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    return image

def read_and_decode_videos(filename_queue, use_frame_distortion=False, use_random_crop=False, use_center_crop=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'video_raw': tf.FixedLenFeature([], tf.string),
        'action_label': tf.FixedLenFeature([], tf.int64),
        'actor_label': tf.FixedLenFeature([], tf.int64)
    })

    video = tf.decode_raw(features['video_raw'], tf.uint8)
    action_label = tf.cast(features['action_label'], tf.int32)
    actor_label = tf.cast(features['actor_label'], tf.int32)
    video = tf.reshape(video, [depth, height, width, nchannel])
    if use_random_crop:
        video = tf.random_crop(video, [depth, crop_height, crop_width, nchannel])
    if use_center_crop:
        video = tf.slice(video, [0, 4, 24, 0], [depth, crop_height, crop_width, nchannel])
    if use_frame_distortion:
        video_distort = []
        for i in range(video.get_shape()[0]):
            frame = distort_image(video[i])
            video_distort.append(frame)
        video = tf.stack(video_distort)
        print('Video distort shape is ', video.get_shape())
    video = tf.cast(video, tf.float32)
    return video, action_label, actor_label

def inputs_videos(filenames, batch_size, num_epochs, num_threads, num_examples_per_epoch, shuffle=True, distort=False):
    '''
    Args:
        shuffle: testing: shuffle=True, allow_smaller_final_batch=True;
                 training: shuffle=False, allow_smaller_final_batch=True;
    '''
    if not num_epochs:
        num_epochs = None
    for filename in filenames:
        if not tf.gfile.Exists(filename):
            raise ValueError('Failed to find file: ' + filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=shuffle, name='string_input_producer'
        )
        video, action_label, actor_label = read_and_decode_videos(filename_queue)

        # print('Video shape is ', video.get_shape())
        # print('Action label shape is ', action_label.get_shape())
        # print('Actor label shape is ', actor_label.get_shape())
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
            videos, action_labels, actor_labels = tf.train.shuffle_batch(
                [video, action_label, actor_label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                min_after_dequeue=min_queue_examples,
                allow_smaller_final_batch=True,
                name='batching_shuffling'
            )
        else:
            videos, action_labels, actor_labels = tf.train.batch(
                [video, action_label, actor_label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                allow_smaller_final_batch=True,
                name='batching_shuffling'
            )
        # print('Videos shape is ', videos.get_shape())
        # print('Action label shape is ', action_labels.get_shape())
        # print('Actor label shape is ', actor_labels.get_shape())
        # print('######################################################################')

    return videos, action_labels, actor_labels

def read_and_decode_images(filename_queue, use_normalization=False):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features['image_raw'], tf.float32)
    label = tf.decode_raw(features['label_raw'], tf.float32)
    if use_normalization:
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.float32) / 255.0
    image = tf.reshape(image, [120, 160, 3])
    label = tf.reshape(label, [120, 160, 3])
    return image, label

def inputs_images(filenames, batch_size, num_epochs, num_threads, num_examples_per_epoch, shuffle=True):
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

        print('Image shape is ', image.get_shape())
        print('Label shape is ', label.get_shape())
        print('######################################################################')
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
                name='batching_unshuffling'
            )
        print('Images shape is ', images.get_shape())
        print('Labels shape is ', labels.get_shape())
    return images, labels
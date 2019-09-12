import tensorflow as tf
from tf_flags import FLAGS

def distort_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    return image

def read_and_decode_images(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_yaw_raw': tf.FixedLenFeature([], tf.float32),
        'label_pitch_raw': tf.FixedLenFeature([], tf.float32),
        'label_roll_raw': tf.FixedLenFeature([], tf.float32),
        'label_yaw_cont_raw': tf.FixedLenFeature([], tf.float32),
        'label_pitch_cont_raw': tf.FixedLenFeature([], tf.float32),
        'label_roll_cont_raw': tf.FixedLenFeature([], tf.float32),
        'gender': tf.FixedLenFeature([], tf.int64),
    })

    image = tf.decode_raw(features['image_raw'], tf.float32)
    label_yaw = tf.cast(features['label_yaw_raw'], tf.float32)
    label_pitch = tf.cast(features['label_pitch_raw'], tf.float32)
    label_roll = tf.cast(features['label_roll_raw'], tf.float32)
    label_yaw_cont = tf.cast(features['label_yaw_cont_raw'], tf.float32)
    label_pitch_cont = tf.cast(features['label_pitch_cont_raw'], tf.float32)
    label_roll_cont = tf.cast(features['label_roll_cont_raw'], tf.float32)
    gender = tf.cast(features['gender'], tf.int32)
    #identity = tf.cast(features['identity'], tf.int32)
    image = tf.reshape(image, [224, 224, 3])
    return image, label_yaw, label_pitch, label_roll, label_yaw_cont, label_pitch_cont, label_roll_cont, gender

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
        image, label_yaw, label_pitch, label_roll, label_yaw_cont, label_pitch_cont, label_roll_cont, gender = read_and_decode_images(filename_queue)

        print('Image shape is ', image.get_shape())
        print('Label_Yaw shape is ', label_yaw.get_shape())
        print('Label_Pitch shape is ', label_pitch.get_shape())
        print('Label_Roll shape is ', label_roll.get_shape())
        print('Label_Yaw_Cont shape is ', label_yaw_cont.get_shape())
        print('Label_Pitch_Cont shape is ', label_pitch_cont.get_shape())
        print('Label_Roll_Cont shape is ', label_roll_cont.get_shape())

        print('######################################################################')
        min_fraction_of_examples_in_queue = 0.5
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        if shuffle:
            images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, labels_pitch_cont, labels_roll_cont, genders = tf.train.shuffle_batch(
                [image, label_yaw, label_pitch, label_roll, label_yaw_cont, label_pitch_cont, label_roll_cont, gender],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                min_after_dequeue=min_queue_examples,
                name='batching_shuffling'
            )
        else:
            images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, labels_pitch_cont, labels_roll_cont, genders = tf.train.shuffle_batch(
                [image, label_yaw, label_pitch, label_roll, label_yaw_cont, label_pitch_cont, label_roll_cont, gender],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                allow_smaller_final_batch=False,
                name='batching_unshuffling'
            )
        print('Images shape is ', images.get_shape())
        print('Labels_Yaw shape is ', labels_yaw.get_shape())
        print('Labels_Pitch shape is ', labels_pitch.get_shape())
        print('Labels_Roll shape is ', labels_roll.get_shape())
        print('Labels_Yaw_Cont shape is ', labels_yaw_cont.get_shape())
        print('Labels_Pitch_Cont shape is ', labels_pitch_cont.get_shape())
        print('Labels_Roll_Cont shape is ', labels_roll_cont.get_shape())
        print('Genders shape is ', genders.get_shape())
        #print('Identities shape is ', identities.get_shape())
    #return images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, labels_pitch_cont, labels_roll_cont, genders, identities
    return images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, labels_pitch_cont, labels_roll_cont, genders
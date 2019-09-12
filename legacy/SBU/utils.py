import errno

import os
import tensorflow as tf

# Restore model only with give variable list
def restore_model_ckpt(sess, ckpt_dir, varlist, modulename):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("##########################{}##########################".format(modulename))
        print(varlist)
        saver = tf.train.Saver(varlist)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Session restored from pretrained model at {}!'.format(ckpt.model_checkpoint_path))
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

# Restore pretrained mobilenet model, leaving the output global average pooling layer untouched
def restore_model_pretrained_MobileNet(sess, cfg, multiplier):
    varlist = [v for v in tf.trainable_variables() if
               any(x in v.name.split('/')[0] for x in ["BudgetModule_{}".format(multiplier)])]
    varlist = [v for v in varlist if not any(x in v.name for x in ["Conv2d_1c_1x1"])]
    print("###############################BudgetModule_{}###############################".format(multiplier))
    print(varlist)
    vardict = {v.name[:-2].replace('BudgetModule_{}'.format(multiplier), 'MobilenetV1'): v for v in varlist}

    mobilenet_dict = {1.0: cfg['MODEL']['PRETRAINED_MOBILENET_10'],
                      0.75: cfg['MODEL']['PRETRAINED_MOBILENET_075'],
                      0.5: cfg['MODEL']['PRETRAINED_MOBILENET_050'],
                      0.25: cfg['MODEL']['PRETRAINED_MOBILENET_025'],
                      }
    saver = tf.train.Saver(vardict)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=mobilenet_dict[multiplier])
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(
            '#############################Session restored from pretrained model at {}!###############################'.format(
                ckpt.model_checkpoint_path))
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), mobilenet_dict[multiplier])

# Restore pretrained C3D model, leaving the output "out" and "d2" layer untouched
def restore_model_pretrained_C3D(sess, cfg):
    if os.path.isfile(cfg['MODEL']['PRETRAINED_C3D']):
        varlist = [v for v in tf.trainable_variables() if
                   any(x in v.name for x in ["UtilityModule"])]
        varlist = [v for v in varlist if not any(x in v.name.split('/')[1] for x in ["out", "d2"])]
        vardict = {v.name[:-2].replace('UtilityModule', 'var_name'): v for v in varlist}
        saver = tf.train.Saver(vardict)
        saver.restore(sess, cfg['MODEL']['PRETRAINED_C3D'])
        print(
            '#############################Session restored from pretrained model at {}!#############################'.format(
                cfg['MODEL']['PRETRAINED_C3D']))
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cfg['MODEL']['PRETRAINED_C3D'])

def get_tensors_ops_graph(sess):
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)
    print('----------------------------Trainable Variables-----------------------------------------')
    for var, val in zip(tvars, tvars_vals):
        print(var.name, val)
    print('----------------------------------------Operations-------------------------------------')
    for op in tf.get_default_graph().get_operations():
        print(str(op.name))
    print('----------------------------------Nodes in the Graph---------------------------------------')
    print([n.name for n in tf.get_default_graph().as_graph_def().node])


# Compute the accuracy given logits and labels
def accuracy(logits, labels):
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def inspect_tensors_in_model(model_name):
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    print_tensors_in_checkpoint_file(file_name=model_name, tensor_name='', all_tensors=False)

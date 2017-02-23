from __future__ import absolute_import
from __future__ import print_function
from keras import backend as K
from pkg_resources import parse_version
from keras.callbacks import Callback


if K.backend() == 'tensorflow':
    import tensorflow as tf

class TensorBoard(Callback):
    """Tensorboard basic visualizations.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    TensorBoard is a visualization tool provided with TensorFlow.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html).
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by Tensorboard
        histogram_freq: frequency (in epochs) at which to compute activation
            histograms for the layers of the model. If set to 0,
            histograms won't be computed.
        write_graph: whether to visualize the graph in Tensorboard.
            The log file can become quite large when
            write_graph is set to True.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False):
        super(TensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images

    def set_model(self, model):
        print('[TB] Setting model...')
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                # print('[TB] Layer: %s ' % layer.name)

                # TODO: do not harcode the layer to plot
                if layer.name=='convolution2d_19':
                    l = layer.output[0]
                    l = tf.expand_dims(l, -1)
                    print('[%s] layer output shape: %s' % (layer.name, str(l.get_shape())))
                    tf.summary.image(layer.name, l)

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)

                    if self.write_images:

                        w_img = tf.squeeze(weight)

                        shape = w_img.get_shape()

                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)

                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)

                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)

                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name), layer.output)

        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # print('[TB] EPOCH FINISHED!\n')
        # print(epoch)

        if self.model.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                # print('[TBD] Inside histogram_freq')
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.model.validation_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.model.validation_data
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                #print('[TBD] Inside histogram_freq, result: %s' % str(result[0]))
                print('[TBD] Inside histogram_freq, result: %s' % str(len(result)))
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            # print('[TB] name:%s, value:%s' % (str(name), str(value)))
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

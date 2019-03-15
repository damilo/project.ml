import numpy as np
import tensorflow as tf
from functools import partial
from datetime import datetime
import os


class MlpSpafi ():
    
    def __init__ (self, name = 'MlpSpafi', restore = False, lr = 0.0001, beta1 = 0.9, beta2 = 0.999):
        self.__metric_val_max = 0
        
        now = datetime.utcnow ().strftime ('%Y%m%d%H%M%S')
        self.__tflog_path = './tf_logs/run_{}-{}'.format (name, now)
        self.__tfsave_file_ = './tf_saves/run_{}/{}'.format (name, 'best_acc_val.ckpt')

        self.__tf_init ()
        self.__construct_graph (lr, beta1, beta2, restore)
    
    def __tf_init (self):
        # random number generation in the TensorFlow backend have a well-defined initial state
        random_state = 42
        tf.set_random_seed (random_state)
        # force TensorFlow to use single thread
        session_conf = tf.ConfigProto (intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
        sess = tf.Session (graph = tf.get_default_graph (), config = session_conf)
    
    # construction phase
    def __construct_graph (self, lr, beta1, beta2, restore = False):
        n_inputs = 4096
        n_outputs = 2
        
        tf.reset_default_graph ()
        
        self.__X = tf.placeholder (
            shape = (None, n_inputs),
            dtype = tf.float32,
            name = 'X'
        )
        self.__y = tf.placeholder (
            shape = (None),
            dtype = tf.int32,
            name = 'y'
        )
        
        he_init = tf.contrib.layers.variance_scaling_initializer (
            factor = 2.0,
            mode = 'FAN_AVG',
            uniform = False
        )
        fc_layer = partial (
            tf.layers.dense,
            kernel_initializer = he_init,
            activation = tf.nn.elu
        )
        with tf.name_scope ('dnn'):
            fc1 = fc_layer (
                inputs = self.__X,
                units = 1024,
                name = 'fc1'
            )
            fc2 = fc_layer (
                inputs = fc1,
                units = 512,
                name = 'fc2'
            )
            fc3 = fc_layer (
                inputs = fc2,
                units = 128,
                name = 'fc3'
            )
            self.__logits = tf.layers.dense (
                inputs = fc3,
                units = n_outputs,
                activation = None,
                name = 'outputs'
            )
        
        with tf.name_scope ('loss'):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits (
                labels = self.__y,
                logits = self.__logits
            )
            loss = tf.reduce_mean (xentropy, name = 'loss')
        
        with tf.name_scope ('train'):
            self.__optimizer = tf.train.AdamOptimizer (learning_rate = lr, beta1 = beta1, beta2 = beta2)
            self.__training_op = self.__optimizer.minimize (loss)
        
        with tf.name_scope ('eval'):
            y_pred_distr = tf.nn.softmax (self.__logits)
            y_pred = tf.argmax (y_pred_distr, axis = 1, output_type = self.__y.dtype)
            self.__metric = self.__metric_precision (self.__y, y_pred)
        
        self.__init_g = tf.global_variables_initializer ()
        self.__saver = tf.train.Saver ()

        self.__loss_summary = tf.summary.scalar ('loss', loss)
        self.__metric_train_summary = tf.summary.scalar ('metric_train', self.__metric)
        self.__metric_val_summary = tf.summary.scalar ('metric_val', self.__metric)
        
        self.__file_writer = tf.summary.FileWriter (self.__tflog_path, tf.get_default_graph ())
    
    # execution phase
    def fit (self, X_train, y_train, X_val, y_val):
        # general params
        N_EPOCHS = 10
        BATCH_SIZE = 80
        n_batches = int (np.ceil (X_train.shape[0] / BATCH_SIZE))
        
        # logging
        log_it = n_batches // 5

        # early stopping
        early_stop = False
        patience = 1
        epoch_best = 0

        with tf.Session () as sess:
            self.__init_g.run ()

            for epoch in range (N_EPOCHS):

                for it in range (n_batches):
                    
                    print ('\repoch # {}, batch # {} / {}'.format (epoch, it+1, n_batches), flush = True, end = '')

                    feed_dict = {
                        self.__X : X_train[it*BATCH_SIZE:(it+1)*BATCH_SIZE],
                        self.__y : y_train[it*BATCH_SIZE:(it+1)*BATCH_SIZE]
                    }

                    sess.run (self.__training_op, feed_dict = feed_dict)
                    
                    # logging
                    if (it % log_it == 0):
                        summary_loss = self.__loss_summary.eval (feed_dict = feed_dict)
                        summary_metric_train = self.__metric_train_summary.eval (feed_dict = feed_dict)
                        summary_metric_val = self.__metric_val_summary.eval (feed_dict = {self.__X : X_val, self.__y : y_val})

                        step = epoch * n_batches + it
                        self.__file_writer.add_summary (summary_loss, step)
                        self.__file_writer.add_summary (summary_metric_train, step)
                        self.__file_writer.add_summary (summary_metric_val, step)
                
                # validation & metric
                metric_train = self.__metric.eval (feed_dict = feed_dict)
                metric_val = self.__metric.eval (feed_dict = {self.__X : X_val, self.__y : y_val})
                print (', train metric:', metric_train, ', val metric:', metric_val)
                
                # early stopping
                if (metric_val > self.__metric_val_max):
                    self.__metric_val_max = metric_val
                    save_file = self.__saver.save (sess, self.__tfsave_file_)
                    epoch_best = epoch
                    print ('val metric improved, model saved to', save_file)
                else:
                    if (epoch > epoch_best + patience):
                        print ('early stopping after epoch {}'.format (epoch, it))
                        early_stop = True
                if (early_stop):
                    self.__file_writer.close ()
                    break
        
        self.__file_writer.close ()
    
    # score
    def score (self, X_test, y_test):
        acc_test = -1
        with tf.Session () as sess:
            self.__saver.restore (sess, self.__tfsave_file_)
            metric_test = self.__metric.eval (feed_dict = {self.__X : X_test, self.__y : y_test})

        return metric_test
    
    # predicions
    def predict (self, X):
        with tf.Session () as sess:
            self.__saver.restore (sess, self.__tfsave_file_)
            Z = self.__logits.eval (feed_dict = {self.__X : X})
            y_pred_distr = tf.nn.softmax (Z).eval ()
            y_pred = np.argmax (y_pred_distr, axis = 1)

        return y_pred, y_pred_distr
    
    # metrics
    def __metric_precision (self, y_true, y_pred):        
        tp = tf.reduce_sum (y_true * y_pred)
        fp = tf.reduce_sum (tf.clip_by_value (y_pred - y_true, 0, 1))
        
        return (tp / (tp + fp))
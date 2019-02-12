import numpy as np
import tensorflow as tf
from functools import partial
from datetime import datetime
import os


class MlpMnist ():
    
    # construction phase
    def __init__ (self, name = 'MlpMnist', restore_ckpt = None, lr = 0.001, beta1 = 0.9, beta2 = 0.999):
        
        self._max_acc_val_ = 0
        self.name = name
        
        now = datetime.utcnow ().strftime ('%Y%m%d%H%M%S')
        self._tflog_path_ = './tf_logs/run_{}-{}'.format (self.name, now)
        self._tfsave_path_ = './tf_saves/run_{}/{}'.format (self.name, 'best_acc_val.ckpt')
        
        self._save_path_best_ = None
        if restore_ckpt is not None:
            self._save_path_best_ = restore_ckpt
        
        
        self.__tf_init ()
        self.__construct_graph (lr, beta1, beta2)
    
    
    
    def __tf_init (self):
        # random number generation in the TensorFlow backend have a well-defined initial state
        random_state = 42
        tf.set_random_seed (random_state)
        # force TensorFlow to use single thread
        session_conf = tf.ConfigProto (intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session (graph=tf.get_default_graph (), config=session_conf)
    
    
    # construction phase
    def __construct_graph (self, lr, beta1, beta2):
        n_inputs = 28*28
        n_layers = 5
        n_units = 100
        n_outputs = 5
        
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
            units = n_units,
            kernel_initializer = he_init,
            activation = tf.nn.elu
        )
        with tf.name_scope ('dnn'):
            fc1 = fc_layer (
                inputs = self.__X,
                name = 'fc1'
            )
            fc2 = fc_layer (
                inputs = fc1,
                name = 'fc2'
            )
            fc3 = fc_layer (
                inputs = fc2,
                name = 'fc3'
            )
            fc4 = fc_layer (
                inputs = fc3,
                name = 'fc4'
            )
            fc5 = fc_layer (
                inputs = fc4,
                name = 'fc5'
            )
            self.__logits = tf.layers.dense (
                inputs = fc5,
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
            correct = tf.nn.in_top_k (self.__logits, self.__y, 1)
            self.__accuracy = tf.reduce_mean (tf.cast (correct, tf.float32))
        
        
        self.__init = tf.global_variables_initializer ()
        self.__saver = tf.train.Saver ()
        
        
        self.__loss_summary = tf.summary.scalar ('loss', loss)
        self.__acc_train_summary = tf.summary.scalar ('acc_train', self.__accuracy)
        self.__acc_val_summary = tf.summary.scalar ('acc_val', self.__accuracy)
        
        self.__file_writer = tf.summary.FileWriter (self._tflog_path_, tf.get_default_graph ())
    
    
    # execution phase
    def fit (self, X_train, y_train, X_val, y_val):
        
        N_EPOCHS = 10
        BATCH_SIZE = 40
        n_batches = int (np.ceil (X_train.shape[0] / BATCH_SIZE))

        # early stopping
        early_stop = False
        patience = 1
        epoch_best = 0

        with tf.Session () as sess:
            self.__init.run ()

            for epoch in range (N_EPOCHS):

                for it in range (n_batches):

                    feed_dict = {
                        self.__X : X_train[it*BATCH_SIZE:(it+1)*BATCH_SIZE],
                        self.__y : y_train[it*BATCH_SIZE:(it+1)*BATCH_SIZE]
                    }

                    sess.run (self.__training_op, feed_dict = feed_dict)
                    
                    # logging
                    if (it % 100 == 0):
                        summary_loss = self.__loss_summary.eval (feed_dict = feed_dict)
                        summary_acc_train = self.__acc_train_summary.eval (feed_dict = feed_dict)
                        summary_acc_val = self.__acc_val_summary.eval (feed_dict = {self.__X : X_val, self.__y : y_val})

                        step = epoch * n_batches + it
                        self.__file_writer.add_summary (summary_loss, step)
                        self.__file_writer.add_summary (summary_acc_train, step)
                        self.__file_writer.add_summary (summary_acc_val, step)

                
                acc_train = self.__accuracy.eval (feed_dict = feed_dict)
                acc_val = self.__accuracy.eval (feed_dict = {self.__X : X_val, self.__y : y_val})
                print (epoch, ', train acc:', acc_train, ', val acc:', acc_val)
                
                
                # early stopping
                if (acc_val > self._max_acc_val_):
                    self._max_acc_val_ = acc_val
                    self._save_path_best_ = self.__saver.save (sess, self._tfsave_path_)
                    epoch_best = epoch
                    print ('val acc improved, model saved to', self._save_path_best_)
                else:
                    if (epoch > epoch_best + patience):
                        print ('early stopping after epoch {}'.format (epoch, it))
                        early_stop = True
            
                if (early_stop):
                    self.__file_writer.close ()
                    break
        
        self.__file_writer.close ()
    
    
    # predicition
    def score (self, X_test, y_test):

        if (self._save_path_best_ is None):
            return -1

        acc_test = -1
        with tf.Session () as sess:
            self.__saver.restore (sess, self._save_path_best_)
            acc_test = self.__accuracy.eval (feed_dict = {self.__X : X_test, self.__y : y_test})

        return acc_test
    
    def predict (self, X):
        
        if (self._save_path_best_ is None):
            return -1
        
        with tf.Session () as sess:
            self.__saver.restore (sess, self._save_path_best_)

            Z = self.__logits.eval (feed_dict = {self.__X : X})
            y_pred_distr = tf.nn.softmax (Z).eval ()
            y_pred = np.argmax (y_pred_distr, axis = 1)

        return y_pred, y_pred_distr
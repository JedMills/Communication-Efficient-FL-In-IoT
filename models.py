"""
Implementation of MNIST and CIFAR models used in IoT paper.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # set otherwise tf prints gpu info   
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import (   Dense, Flatten, Conv2D, MaxPool2D,
                                        BatchNormalization, ReLU, Softmax,
                                        Dropout)


class FedAvgModel(Model):
    """
    Abstract class for models used in paper. Provides easy methods for getting
    and setting model and optimizer weights, and training/testing within FedAvg
    code. Children must implement fwd_train and fwd_test.
    """
    
    def __init__(self, optimizer, loss_calc):
        """
        Returns a new FedAvgModel. Uses the associated optimizer and loss 
        function as part of the training/testing methods.

        Parameters:
        optimizer (tf.keras.optimizers) class callable *not* instance
        loss_cals (tf.keras.losses)     loss callable *not* instance
        """
        super(FedAvgModel, self).__init__()
        self.optimizer = optimizer()
        self.loss_calc = loss_calc()
    
    
    def set_weights(self, new_ws):
        """ Set model weights to given 2d list of layers. """
        for i in range(len(new_ws)):
            self.layer_list[i].set_weights(new_ws[i])
    
    
    def get_weights(self):
        """ Returns list of layers of weights. """
        return [w.get_weights() for w in self.layer_list]

        
    def get_optim_weights(self):
        """ Returns list of arrays of current optimizer values. """
        return self.optimizer.get_weights()


    def set_optim_weights(self, ws):
        """ Set the optimizer values using given 2d list of layers. """
        self.optimizer.set_weights(ws)

        
    def fwd_train(self, x):
        """
        Perform one forward training pass.
        
        Parameters:
        x (array):  input data of shape [batch_size, data_shape]
        
        Returns:
        array of shape [batch_size, n_outputs]
        """
        raise NotImplementedError()

        
    def fwd_test(self, x):
        """
        Perform one forward pass during testing.
        
        Parameters:
        x (array):  input data of shape [batch_size, data_shape]
        
        Returns:
        array of shape [batch_size, n_outputs]
        """
        raise NotImplementedError()

        
    def train_step(self, x, y):
        """
        Perform one training step using the given input data and labels, using
        the loss function and optimizer given when created.
        
        Parameters:
        x (array):  input data of shape [batch_size, sample_shape]
        y (array):  one-hot labels, shape [batch_size, num_classes]
        
        Returns:
        (mean accuracy, mean loss) over data 
        """
        grads, loss, preds = self.get_grads_loss_preds(x, y)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # calc top-1 accuracy
        acc = np.mean(np.argmax(preds.numpy(), axis=0) == np.argmax(y, axis=0))
        
        return acc, loss

                
    def get_grads_loss_preds(self, x, y):
        """
        Perform one forward and backward pass using the given data.
        
        Parameters:
        x (array):  input data of shape [batch_size, sample_shape]
        y (array):  one-hot labels, shape [batch_size, num_classes]
        
        Returns:
        grads (list):   list of gradients computed over batch
        loss (float):   average loss over batch
        preds (array):  predictions for batch, shape [batch_size, n_classes]
        """
        # forward pass
        with tf.GradientTape() as tape:
            preds = self.fwd_train(x)
            loss = self.loss_calc(y, preds)
        
        # backward pass
        grads = tape.gradient(loss, self.trainable_variables)
        loss = loss.numpy()
        
        return grads, loss, preds

        
    def train(self, x, y, B):
        """
        Splits data into batches of size B and performs SGD for 1 epoch.
        
        Parameters:
        x (array):  samples, shape [total_samples, sample_shape]
        y (array):  one-hot labels, shape [total_samples, num_classes]
        B (int):    batch size
        """
        b_pr_e = x.shape[0] // B 
        
        for b in range(b_pr_e):
            self.train_step(x[b*B:(b+1)*B], y[b*B:(b+1)*B])
        
        if b_pr_e * B < x.shape[0]:
            self.train_step(x[b_pr_e*B:], y[b_pr_e*B:])

                        
    def _test_xy(self, x, y):
        """
        Return average loss and top-1 accuracy on one batch.
        
        Parameters:
        x (array):  samples array of shape [batch_size, sample_shape]
        y (array):  one-hot labels of shape [batch_size, num_classes]
        
        Returns:
        (mean loss, mean accuracy) over data
        """
        preds = self.fwd_test(x)
        loss = self.loss_calc(y, preds).numpy()
        acc = np.mean(np.argmax(preds.numpy(), axis=1) ==  np.argmax(y, axis=1))
        
        return loss, acc 

            
    def test(self, x, y, B):
        """
        Compute mean error and accuracy over given data using batch size B
        
        Parameters:
        x (array):  samples of shape [total_samples, sample_shape]
        y (array):  one-hot labels of shape [total_samples, num_classes]
        B (int):    batch size to use during testing
        
        Returns:
        (mean error, mean accuracy) over data
        """
        b_pr_t = x.shape[0] // B 
        avg_err = 0.0 
        avg_acc = 0.0 
        tot_samples = 0
        
        for b in range(b_pr_t):
            loss, acc = self._test_xy(x[b*B:(b+1)*B], y[b*B:(b+1)*B])
            avg_err += loss
            avg_acc += acc 
            tot_samples += 1
        
        # run on last few samples if B does not fit exactly into total_samples
        if b_pr_t * B < x.shape[0]:
            loss, acc = self._test_xy(x[b_pr_t*B:], y[b_pr_t*B:])
            avg_err += loss 
            avg_acc += acc 
            tot_samples += 1 
        
        avg_err /= tot_samples
        avg_acc /= tot_samples
        
        return avg_err, avg_acc


class MNIST2NNModel(FedAvgModel):
    """ MNIST-2NN model described in Section V A. """

    name = 'MNIST2NNModel'

    def __init__(self, optimizer, loss_calc, in_size, outputs):
        """
        Create a new MNIST-2NN model.
        
        Parameters:
        optimizer (tf.keras.optimizers):    class callable *not* instance
        loss_cals (tf.keras.losses):        loss callable *not* instance
        in_size (int):                      number of sample features
        outputs (int):                      number of classes
        """
        super(MNIST2NNModel, self).__init__(optimizer, loss_calc)
        self.dense1 = Dense(200, activation='relu')
        self.dense2 = Dense(200, activation='relu')
        self.out    = Dense(outputs, activation='softmax')
        
        self.layer_list = [self.dense1, self.dense2, self.out]
        # because of TF 2's eager execution I can't find a way of initialising 
        # the model weights and optimizer values easily without running a single
        # training step when the model is created
        self.train_step(np.zeros((1,in_size), dtype=np.float32), 
                        np.zeros((1,outputs), dtype=np.float32))
        
        
    def fwd_train(self, x):
        """
        Perform one forward pass during training.
        
        Parameters:
        x (array):  input data of shape [batch_size, data_shape]
        
        Returns:
        array of shape [batch_size, n_outputs]
        """
        a = self.dense1(x)
        a = self.dense2(a)
        return self.out(a)
        
        
    def fwd_test(self, x):
        """
        Perform one forward pass during testing.
        
        Parameters:
        x (array):  input data of shape [batch_size, data_shape]
        
        Returns:
        array of shape [batch_size, n_outputs]
        """
        return self.fwd_train(x)
        
        
class MNISTCNNModel(FedAvgModel):
    """ MNIST-CNN model described in section V A. """

    name = 'MNISTCNNModel'

    def __init__(self, optimizer, loss_calc, img_size, channels, outputs):
        """
        Create a new MNIST-CNN model.
        
        Parameters:
        optimizer (tf.keras.optimizers):    class callable *not* instance
        loss_cals (tf.keras.losses):        loss callable *not* instance
        img_size (int):                     height and width of image
        channels (int):                     number of channels in image
        outputs (int):                      number of classes
        """
        super(MNISTCNNModel, self).__init__(optimizer, loss_calc)
        self.conv1      = Conv2D(32, 5, activation='relu', padding='same')
        self.pool1      = MaxPool2D((2, 2), (2, 2))
        self.conv2      = Conv2D(64, 5, activation='relu', padding='same')
        self.pool2      = MaxPool2D((2, 2), (2, 2))
        self.flatten    = Flatten()
        self.dense1     = Dense(512, activation='relu')
        self.out        = Dense(outputs, activation='softmax')
        
        self.layer_list = [self.conv1, self.conv2, self.dense1, self.out]
        # because of TF 2's eager execution I can't find a way of initialising 
        # the model weights and optimizer values easily without running a single
        # training step when the model is created
        self.train_step(np.zeros((1,img_size,img_size,channels), dtype=np.float32), 
                        np.zeros((1,outputs), dtype=np.float32))
        
        
    def fwd_train(self, x):
        """
        Perform one forward pass during training.
        
        Parameters:
        x (array):  input data of shape [batch_size, data_shape]
        
        Returns:
        array of shape [batch_size, n_outputs]
        """
        x_2d = tf.reshape(x, [-1, 28, 28, 1])
        a = self.conv1(x_2d)
        a = self.pool1(a)
        a = self.conv2(a)
        a = self.pool2(a)
        a = self.flatten(a)
        a = self.dense1(a)
        return self.out(a)
        
        
    def fwd_test(self, x):
        """
        Perform one forward pass during testing.
        
        Parameters:
        x (array):  input data of shape [batch_size, data_shape]
        
        Returns:
        array of shape [batch_size, n_outputs]
        """
        return self.fwd_train(x)
        
        
class CIFARCNNModel(FedAvgModel):
    """ CIFAR-CNN fully-convolutional model as described in section V A. """

    name = 'CIFARCNNModel'
    
    def __init__(self, optimizer, loss_calc, img_size, channels, outputs):
        """
        Create a new CIFAR-CNN model.
        
        Parameters:
        optimizer (tf.keras.optimizers):    class callable *not* instance
        loss_cals (tf.keras.losses):        loss callable *not* instance
        img_size (int):                     height and width of image
        channels (int):                     number of channels in image
        outputs (int):                      number of classes
        """
        super(CIFARCNNModel, self).__init__(optimizer, loss_calc)
        self.conv1  = Conv2D(32, 3, activation='relu', padding='same')
        self.bn1    = BatchNormalization()
        self.conv2  = Conv2D(32, 3, activation='relu', padding='same')
        self.bn2    = BatchNormalization()
        self.drop1  = Dropout(0.2)
        
        self.conv3  = Conv2D(64, 3, activation='relu', padding='same')
        self.bn3    = BatchNormalization()
        self.conv4  = Conv2D(64, 3, activation='relu', padding='same')
        self.bn4    = BatchNormalization()
        self.drop2  = Dropout(0.3)
        
        self.conv5  = Conv2D(128, 3, activation='relu', padding='same')
        self.bn5    = BatchNormalization()
        self.conv6  = Conv2D(128, 3, activation='relu', padding='same')
        self.bn6    = BatchNormalization()
        self.drop3  = Dropout(0.4)
        
        self.flat   = Flatten()
        
        self.out    = Dense(outputs, activation='softmax')
        
        self.layer_list = [ self.conv1, self.bn1, self.conv2, self.bn2, 
                            self.conv3, self.bn3, self.conv4, self.bn4,
                            self.conv5, self.bn5, self.conv6, self.bn6,
                            self.out]
        self.train_step(np.zeros((1,img_size,img_size,channels), dtype=np.float32), 
                        np.zeros((1,outputs), dtype=np.float32))
        
        
    def _fwd(self, x, train):
        """
        Internal method used by fwd_train and fwd_test: model contains BN layers 
        that need a boolean passed whether training (using batch statistics) or
        testing (using running averages).
        """
        a = self.conv1(x)
        a = self.bn1(a, training=train)
        a = self.conv2(a)
        a = self.bn2(a, training=train)
        a = self.drop1(a)
        
        a = self.conv3(a)
        a = self.bn3(a, training=train)
        a = self.conv4(a)
        a = self.bn4(a, training=train)
        a = self.drop2(a)
        
        a = self.conv5(a)
        a = self.bn5(a, training=train)
        a = self.conv6(a)
        a = self.bn6(a, training=train)
        a = self.drop3(a)
        
        a = self.flat(a)
        
        return self.out(a)
        
        
    def fwd_train(self, x):
        """
        Perform one forward pass during training.
        
        Parameters:
        x (array):  input data of shape [batch_size, data_shape]
        
        Returns:
        array of shape [batch_size, n_outputs]
        """
        return self._fwd(x, True)
        
        
        
    def fwd_test(self, x):
        """
        Perform one forward pass during testing.
        
        Parameters:
        x (array):  input data of shape [batch_size, data_shape]
        
        Returns:
        array of shape [batch_size, n_outputs]
        """
        return self._fwd(x, False)


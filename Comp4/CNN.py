# https://navoshta.com/traffic-signs-classification/
# https://github.com/navoshta/traffic-signs
# https://github.com/uranus4ever/Traffic-Sign-Classifier-CNN/blob/master/Traffic_Sign_Classifier-Copy1.ipynb

import numpy as np
from sklearn.utils import shuffle
from skimage import exposure
import matplotlib.pyplot as plt 
import cv2

# Step 0: Sample the dataset.
# for i in range(1, 39209, 1000):
#     plt.imshow(X_train[i]) 
#     plt.show()

# Step 1: Load and explore the data.
data = np.load('./train_and_test.npz')
X_train, y_train, X_test = data['X_train'], data['y_train'], data['X_test']
# print(y_train)
# for i in [1000, 5000, 7000, 20000, 28000]:
#     print(np.where(y_train[i] == 1)[0] == 23)
# print(y_train[0].shape[0])
# print(X_train.shape[0])
# examples = [209, 4679, 4681]
# temp = np.array([X_train[i] for i in examples])

num_of_samples_index = np.zeros(43)
j = 0 # index recorder
for i in range(0, 43):
    found = False
    while not found and j < X_train.shape[0]:
        if np.where(y_train[j] == 1)[0] == i:
            found = True
            img = X_train[j]
            # print ('sign class ', y_train[i])
            # print ('found at index ', j)  
        j += 1
    num_of_samples_index[i] = j
# print(num_of_samples_index)
num_of_samples = np.zeros(43)
for i in range(0, 42):
    num_of_samples[i] = num_of_samples_index[i + 1] - num_of_samples_index[i]
num_of_samples[42] = y_train.shape[0] - num_of_samples_index[42] + 1
# print(num_of_samples)

# Step 2: Preprocess the image
def crop(Xs, margin = 4):
    new_Xs = np.zeros((Xs.shape[0], 32 - 2 * margin, 32 - 2 * margin, 3))
    for i in range(Xs.shape[0]):
        new_Xs[i] = Xs[i][margin : (Xs[i].shape[0] - margin), margin : (Xs[i].shape[1] - margin)]
    return new_Xs

def preprocess(Xs):
    # Crop, with certain pixels.
    new_Xs = crop(Xs)

    # To make the network more interpretable, scale features to be in [0, 1] by dividing matrix by 255. 
    new_Xs = (new_Xs / 255.).astype(np.float32)

    # # Enhance contrast.
    for i in range(new_Xs.shape[0]):
        new_Xs[i] = exposure.equalize_adapthist(new_Xs[i])

    # convert into gray images
    # https://brohrer.github.io/convert_rgb_to_grayscale.html
    # linear approximation of gamma correction
    new_Xs = 1 * new_Xs[:, :, :, 0] + 0 * new_Xs[:, :, :, 1] + 0 * new_Xs[:, :, :, 2]

    # Add a single grayscale channel
    # https://stackoverflow.com/questions/52746659/how-to-convert-a-rgb-image-3-channel-to-grayscale-1-channel-and-save-it
    # new_Xs = new_Xs.reshape(new_Xs.shape + (1,))     
    return new_Xs

# imgs = preprocess(temp)
# for i in imgs:
#     plt.imshow(i, cmap="gray") 
#     plt.show()

# flipping

# rotation
# Use rotation to create new images to enlarge the dataset to increase training accurancy.
# https://github.com/uranus4ever/Traffic-Sign-Classifier-CNN/blob/master/Traffic_Sign_Classifier-Copy1.ipynb
def rotate_img(img):
    mid_x, mid_y = int(img.shape[0] / 2), int(img.shape[1] / 2)
    angle = 30.0 * np.random.rand() - 15
    Mat = cv2.getRotationMatrix2D((mid_x, mid_y), angle, 1.0)
    return cv2.warpAffine(img, Mat, img.shape[: 2])

ylabel_small = []
for i in range(43):
    if num_of_samples[i] <= 400:
        ylabel_small.append(i)
# print(ylabel_small)

X_train_new, X_train_update = [], []
y_train_new, y_train_update = [], []
X_train_new = preprocess(X_train)
y_train_new = y_train
for idx in range(X_train_new.shape[0]):    
    img = X_train[idx]
    if np.where(y_train[idx] == 1)[0] in ylabel_small: 
        # print(idx)  
        X_train_update.append(img)
        y_train_update.append(y_train[idx])     
        X_train_update.append(rotate_img(img))
        y_train_update.append(y_train[idx])
        # Call rotation multiple times to improve data set accuracy.
        # X_train_update.append(rotate_img(img))
        # y_train_update.append(y_train[idx])
    else:
        # print(idx)
        X_train_update.append(img)
        y_train_update.append(y_train[idx])

# print('Labels whose quantity less than 400 =',ylabel_small)
# print('origin train data number = ', len(X_train))
# print('Updated train data number =', len(X_train_update))
# print('Updated train label number =', len(y_train_update))


# Preprocess test data
X_test_update = preprocess(X_test)

# shuffle the data
from sklearn.utils import shuffle
X_train_update, y_train_update = shuffle(X_train_update, y_train_update)
# print("Updated Train Image Shape: {}".format(X_train[0].shape))

# Step 3: Neural networks model
# https://www.kaggle.com/jmvictor5656/traffic-sign-classifier
class EarlyStopping(object):
    """
    Provides early stopping functionality. Keeps track of model accuracy, 
    and if it doesn't improve over time restores last best performing 
    parameters.
    """
    
    def __init__(self, saver, session, patience = 100, minimize = True):
        """
        Initialises a `EarlyStopping` isntance.

        Parameters
        ----------
        saver     : 
                    TensorFlow Saver object to be used for saving and restoring model.
        session   : 
                    TensorFlow Session object containing graph where model is restored.
        patience  : 
                    Early stopping patience. This is the number of epochs we wait for 
                    accuracy to start improving again before stopping and restoring 
                    previous best performing parameters.
                  
        Returns
        -------
        New instance.
        """
        self.minimize = minimize
        self.patience = patience
        self.saver = saver
        self.session = session
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.restore_path = None

    def __call__(self, value, epoch):
        """
        Checks if we need to stop and restores the last well performing values if we do.

        Parameters
        ----------
        value     : 
                    Last epoch monitored value.
        epoch     : 
                    Last epoch number.
                  
        Returns
        -------
        `True` if we waited enough and it's time to stop and we restored the 
        best performing weights, or `False` otherwise.
        """
        if (self.minimize and value < self.best_monitored_value) or (not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.restore_path = self.saver.save(self.session, os.getcwd() + "/early_stopping_checkpoint")
        elif self.best_monitored_epoch + self.patience < epoch:
            if self.restore_path != None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print("ERROR: Failed to restore session")
            return True
        
        return False

import tensorflow as tf
from tensorflow.contrib.layers import flatten
EPOCHS = 30
BATCH_SIZE = 128

def fully_connected(input, size):

    weights = tf.get_variable( 'weights', 
        shape = [input.get_shape()[1], size],
        initializer = tf.contrib.layers.xavier_initializer()
      )
    biases = tf.get_variable( 'biases',
        shape = [size],
        initializer = tf.constant_initializer(0.0)
      )
    return tf.matmul(input, weights) + biases

def fully_connected_relu(input, size):
    return tf.nn.relu(fully_connected(input, size))

def conv_relu(input, kernel_size, depth):
    """
    Performs a single convolution layer pass.
    """
    weights = tf.get_variable( 'weights', 
        shape = [kernel_size, kernel_size, input.get_shape()[3], depth],
        initializer = tf.contrib.layers.xavier_initializer()
      )
    biases = tf.get_variable( 'biases',
        shape = [depth],
        initializer = tf.constant_initializer(0.0)
      )
    conv = tf.nn.conv2d(input, weights,
        strides = [1, 1, 1, 1], padding = 'SAME')
    return tf.nn.relu(conv + biases)

def pool(input, size):
    """
    Performs a max pooling layer pass.
    """
    return tf.nn.max_pool(
        input, 
        ksize = [1, size, size, 1], 
        strides = [1, size, size, 1], 
        padding = 'SAME'
    )

def model_pass(input, params, is_training):
    """
    Performs a full model pass.
    
    Parameters
    ----------
    input         : Tensor
                    NumPy array containing a batch of examples.
    params        : Parameters
                    Structure (`namedtuple`) containing model parameters.
    is_training   : Tensor of type tf.bool
                    Flag indicating if we are training or not (e.g. whether to use dropout).
                    
    Returns
    -------
    Tensor with predicted logits.
    """
    # Convolutions

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(input, kernel_size = params.conv1_k, depth = params.conv1_d) 
    with tf.variable_scope('pool1'): 
        pool1 = pool(conv1, size = 2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob = params.conv1_p), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size = params.conv2_k, depth = params.conv2_d)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size = 2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob = params.conv2_p), lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size = params.conv3_k, depth = params.conv3_d)
    with tf.variable_scope('pool3'):
        pool3 = pool(conv3, size = 2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob = params.conv3_p), lambda: pool3)
    
    # Fully connected
    
    # 1st stage output
    pool1 = pool(pool1, size = 4)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])
    
    # 2nd stage output
    pool2 = pool(pool2, size = 2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])    
    
    # 3rd stage output
    shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])
    
    flattened = tf.concat(1, [pool1, pool2, pool3])
    
    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size = params.fc4_size)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob = params.fc4_p), lambda: fc4)
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size = params.num_classes)
    return logits

def plot_curve(axis, params, train_column, valid_column, linewidth = 2, train_linestyle = "b-", valid_linestyle = "g-"):
    """
    Plots a pair of validation and training curves on a single plot.
    """
    model_history = np.load(Paths(params).train_history_path + ".npz")
    train_values = model_history[train_column]
    valid_values = model_history[valid_column]
    epochs = train_values.shape[0]
    x_axis = np.arange(epochs)
    axis.plot(x_axis[train_values > 0], train_values[train_values > 0], train_linestyle, linewidth=linewidth, label="train")
    axis.plot(x_axis[valid_values > 0], valid_values[valid_values > 0], valid_linestyle, linewidth=linewidth, label="valid")
    return epochs

# Plots history of learning curves for a specific model.
def plot_learning_curves(params):
    """
    Plots learning curves (loss and accuracy on both training and validation sets) for a model identified by a parameters struct.
    """
    curves_figure = pyplot.figure(figsize = (10, 4))
    axis = curves_figure.add_subplot(1, 2, 1)
    epochs_plotted = plot_curve(axis, parameters, train_column = "train_accuracy_history", valid_column = "valid_accuracy_history")

    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("accuracy")
    pyplot.ylim(50., 115.)
    pyplot.xlim(0, epochs_plotted)

    axis = curves_figure.add_subplot(1, 2, 2)
    epochs_plotted = plot_curve(axis, parameters, train_column = "train_loss_history", valid_column = "valid_loss_history")

    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(0.0001, 10.)
    pyplot.xlim(0, epochs_plotted)
    pyplot.yscale("log")
    

# Step 4: training
def train_model(params, X_train, y_train, X_valid, y_valid, X_test):
    """
    Performs model training based on provided training dataset 
    according to provided parameters, and then evaluates trained 
    model with testing dataset. 
    Part of the training dataset may be used for validation during
    training if specified in model parameters.

    Parameters
    ----------
    params        : Parameters
                    Structure (`namedtuple`) containing model parameters.
    X_train       : 
                    Training dataset. 
    y_train       : 
                    Training dataset labels. 
    X_valid       : 
                    Validation dataset. 
    y_valid       : 
                    Validation dataset labels. 
    X_test        : 
                    Testing dataset. 
    """
    
    # Initialisation routines: generate variable scope, create logger, note start time.
    paths = Paths(params)
    start = time.time()
    model_variable_scope = paths.var_scope
    
    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
        tf_x_batch = tf.placeholder(tf.float32, shape = (None, params.image_size[0], params.image_size[1], 1))
        tf_y_batch = tf.placeholder(tf.float32, shape = (None, params.num_classes))
        is_training = tf.placeholder(tf.bool)
        current_epoch = tf.Variable(0, trainable=False)  # count the number of epochs

        # Model parameters.
        if params.learning_rate_decay:
            learning_rate = tf.train.exponential_decay(params.learning_rate, current_epoch, decay_steps = params.max_epochs, decay_rate = 0.01)
        else:
            learning_rate = params.learning_rate
            
        # Training computation.
        with tf.variable_scope(model_variable_scope):
            logits = model_pass(tf_x_batch, params, is_training)
            if params.l2_reg_enabled:
                with tf.variable_scope('fc4', reuse = True):
                    l2_loss = tf.nn.l2_loss(tf.get_variable('weights'))
            else:
                l2_loss = 0

        predictions = tf.nn.softmax(logits)
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf_y_batch)
        loss = tf.reduce_mean(softmax_cross_entropy) + params.l2_lambda * l2_loss  

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate
        ).minimize(loss)

    with tf.Session(graph = graph) as session:
        session.run(tf.global_variables_initializer())

        # A routine for evaluating current model parameters
        def get_accuracy_and_loss_in_batches(X, y):
            p = []
            sce = []
            batch_iterator = BatchIterator(batch_size = 128)
            for x_batch, y_batch in batch_iterator(X, y):
                [p_batch, sce_batch] = session.run([predictions, softmax_cross_entropy], feed_dict = {
                        tf_x_batch : x_batch, 
                        tf_y_batch : y_batch,
                        is_training : False
                    }
                )
                p.extend(p_batch)
                sce.extend(sce_batch)
            p = np.array(p)
            sce = np.array(sce)
            accuracy = 100.0 * np.sum(np.argmax(p, 1) == np.argmax(y, 1)) / p.shape[0]
            loss = np.mean(sce)
            return (accuracy, loss)
        
        # If we chose to keep training previously trained model, restore session.
        if params.resume_training: 
            try:
                tf.train.Saver().restore(session, paths.model_path)
            except Exception as e:
                log("Failed restoring previously trained model: file does not exist.")
                pass
        
        saver = tf.train.Saver()
        early_stopping = EarlyStopping(tf.train.Saver(), session, patience = params.early_stopping_patience, minimize = True)
        train_loss_history = np.empty([0], dtype = np.float32)
        train_accuracy_history = np.empty([0], dtype = np.float32)
        valid_loss_history = np.empty([0], dtype = np.float32)
        valid_accuracy_history = np.empty([0], dtype = np.float32)
        if params.max_epochs > 0:
            log("================= TRAINING ==================")
        else:
            log("================== TESTING ==================")       
        log(" Timestamp: " + get_time_hhmmss())
        
        for epoch in range(params.max_epochs):
            current_epoch = epoch
            # Train on whole randomised dataset in batches
            batch_iterator = BatchIterator(batch_size = params.batch_size, shuffle = True)
            for x_batch, y_batch in batch_iterator(X_train, y_train):
                session.run([optimizer], feed_dict = {
                        tf_x_batch : x_batch, 
                        tf_y_batch : y_batch,
                        is_training : True
                    }
                )

            # If another significant epoch ended, we log our losses.
            if (epoch % params.log_epoch == 0):
                # Get validation data predictions and log validation loss:
                valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)

                # Get training data predictions and log training loss:
                train_accuracy, train_loss = get_accuracy_and_loss_in_batches(X_train, y_train)

                if (epoch % params.print_epoch == 0):
                    log("-------------- EPOCH %4d/%d --------------" % (epoch, params.max_epochs))
                    log("     Train loss: %.8f, accuracy: %.2f%%" % (train_loss, train_accuracy))
                    log("Validation loss: %.8f, accuracy: %.2f%%" % (valid_loss, valid_accuracy))
                    log("      Best loss: %.8f at epoch %d" % (early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    log("   Elapsed time: " + get_time_hhmmss(start))
                    log("      Timestamp: " + get_time_hhmmss())
            else:
                valid_loss = 0.
                valid_accuracy = 0.
                train_loss = 0.
                train_accuracy = 0.
                
            valid_loss_history = np.append(valid_loss_history, [valid_loss])
            valid_accuracy_history = np.append(valid_accuracy_history, [valid_accuracy])
            train_loss_history = np.append(train_loss_history, [train_loss])
            train_accuracy_history = np.append(train_accuracy_history, [train_accuracy])
            
            if params.early_stopping_enabled:
                # Get validation data predictions and log validation loss:
                if valid_loss == 0:
                    _, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
                if early_stopping(valid_loss, epoch): 
                    log("Early stopping.\nBest monitored loss was {:.8f} at epoch {}.".format(
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch
                    ))
                    break

        # Evaluate on test dataset.
        # test_accuracy, test_loss = get_accuracy_and_loss_in_batches(X_test, y_test)
        # valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
        # log("=============================================")
        # log(" Valid loss: %.8f, accuracy = %.2f%%)" % (valid_loss, valid_accuracy)) 
        # log(" Test loss: %.8f, accuracy = %.2f%%)" % (test_loss, test_accuracy)) 
        # log(" Total time: " + get_time_hhmmss(start))
        # log("  Timestamp: " + get_time_hhmmss())

        # Save model weights for future use.
        saved_model_path = saver.save(session, paths.model_path)
        log("Model file: " + saved_model_path)
        np.savez(paths.train_history_path, train_loss_history = train_loss_history, train_accuracy_history = train_accuracy_history, valid_loss_history = valid_loss_history, valid_accuracy_history = valid_accuracy_history)
        log("Train history file: " + paths.train_history_path)
        
        plot_learning_curves(params)
        
        pyplot.show()

from sklearn.model_selection import train_test_split

parameters = Parameters(
    # Data parameters
    num_classes = 43,
    image_size = (24, 24),
    # Training parameters
    batch_size = 256,
    max_epochs = 1001,
    log_epoch = 1,
    print_epoch = 1,
    # Optimisations
    learning_rate_decay = False,
    learning_rate = 0.0001,
    l2_reg_enabled = True,
    l2_lambda = 0.0001,
    early_stopping_enabled = True,
    early_stopping_patience = 100,
    resume_training = True,
    # Layers architecture
    conv1_k = 5, conv1_d = 32, conv1_p = 0.9,
    conv2_k = 5, conv2_d = 64, conv2_p = 0.8,
    conv3_k = 5, conv3_d = 128, conv3_p = 0.7,
    fc4_size = 1024, fc4_p = 0.5
)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_update, y_train_update, test_size = 0.25)
X_test = X_test_update
Prediction = train_model(parameters, X_train, y_train, X_valid, y_valid, X_test)

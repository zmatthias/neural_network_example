import tflearn
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def tempnet():

    network = tflearn.input_data(shape=[None,1], name='input')
    network = tflearn.batch_normalization(network)
    #network = fully_connected(network, 1, activation='relu')

    network = fully_connected(network, 3, activation='softmax')

    momentum = tflearn.Momentum(learning_rate=0.06, lr_decay=0.8, decay_step=1000)

    network = regression(network, optimizer=momentum,
                         loss='mean_square', name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_simplenet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
    return model
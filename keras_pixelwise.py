import numpy as np
np.random.seed(1234)  # for reproducibility
from sklearn.cross_validation import train_test_split
from scipy.misc import imread
import time
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
#import cv2
#from keras import backend as K
import theano.tensor as T
import pickle


def load_data(data_dir, h=640, w=640, sub_im_width=64, sample_stride=64):
    all_image_filenames = os.listdir(data_dir)
    satellite_filenames = [f for f in all_image_filenames if 'satellite' in f]
    segmented_filenames = [f for f in all_image_filenames if 'segmented' in f]
    total_num_img = len(satellite_filenames)
    all_satellite_data = np.zeros((total_num_img, h, w, 3), dtype=np.uint8)
    all_class_data_as_RGB= np.zeros((total_num_img, h, w, 3), dtype=np.uint8)
    all_class_data_as_class = np.zeros((total_num_img, h, w, 1), dtype=np.uint8)
    colors_to_classes = {(0, 0, 0): 0, (0, 0, 255): 1, (0, 255, 0): 2}  # 0: background; 1: water; 2: road
    classes_to_colors = {0: (0, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
    idx_to_write = 0
    print 'Loading image data...'
    for idx, satellite_filename in enumerate(satellite_filenames):
        satellite_img = imread('{}/{}'.format(data_dir, satellite_filename))
        all_satellite_data[idx] = satellite_img
    for idx, segmented_filename in enumerate(segmented_filenames):
        segmented_img = imread('{}/{}'.format(data_dir, segmented_filename))
        all_class_data_as_RGB[idx] = segmented_img
    for RGB_color in [(0, 0, 255), (0, 255, 0)]:
        for img_idx in range(total_num_img):
            print 'Done. \nFinding pixelwise classes from pixel color for image {}'.format(img_idx)
            locs = np.where(np.logical_and(
                        (all_class_data_as_RGB[img_idx, :, :, 0] == RGB_color[0]),
                        (all_class_data_as_RGB[img_idx, :, :, 1] == RGB_color[1]),
                        (all_class_data_as_RGB[img_idx, :, :, 2] == RGB_color[2])))
            print 'Done. \nWriting class data from color pixels to array...'
            all_class_data_as_class[img_idx, 
                                    locs[0], 
                                    locs[1]] = colors_to_classes[RGB_color]
            del locs
    del all_class_data_as_RGB    
    print 'Done. \nSubsetting image data...'
    #offset = sub_im_width/2.  # Need to start where subset image frame will fill
    h_start_pxs = np.arange(0, h-sub_im_width+1, sample_stride)
    w_start_pxs = np.arange(0, w-sub_im_width+1, sample_stride)
    total_num_subsampled_img = total_num_img * len(h_start_pxs) * len(w_start_pxs)
    X = np.zeros((total_num_subsampled_img, 
                  sub_im_width, 
                  sub_im_width, 
                  3), dtype=np.uint8)
    y = np.zeros((total_num_subsampled_img, 
                  sub_im_width, 
                  sub_im_width, 
                  3), dtype=np.uint8)
    idx_to_write = 0
    for img_idx in range(total_num_img):
        for h_start_px in h_start_pxs:
            for w_start_px in w_start_pxs:
                h_end_px = h_start_px + sub_im_width
                w_end_px = w_start_px + sub_im_width
                im_subset = all_satellite_data[img_idx][h_start_px:h_end_px, w_start_px:w_end_px]
                im_subset_as_class = all_class_data_as_class[img_idx][h_start_px:h_end_px, w_start_px:w_end_px]
                X[idx_to_write] = im_subset
                y[idx_to_write] = im_subset_as_class
                idx_to_write += 1
    print 'Done. \nReshaping image data...'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    num_train_img = X_train.shape[0]
    num_test_img = X_test.shape[0]
    X_train = X_train.reshape(num_train_img, 3, sub_im_width, sub_im_width)
    X_test = X_test.reshape(num_test_img, 3, sub_im_width, sub_im_width)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.
    y_train = np_utils.to_categorical(y_train, len(classes_to_colors))
    y_test = np_utils.to_categorical(y_test, len(classes_to_colors))
    print 'Done.'
    return X_train, y_train, X_test, y_test


def set_basic_model_param():
    ''' 
    INPUT:  (1) any additional information (to be converted to string)
                that will be added to the filename when the model is saved
    OUTPUT: (1) Dictionary of important values for formatting data and 
                compiling model. 
    For lightweight tuning of the model (ie. no change in overall structure) 
    it's easiest to keep all model parameters in one place.
    '''
    model_param = {'n_rows': 64, 
                   'n_cols': 64,
                   'n_chan': 3,
                   'n_classes': 3,
                   'n_epoch': 10,
                   'batch_size': 64,
                   'pool_size': 2,
                   'conv_size': 3,
                   'n_conv_nodes': 128,
                   'n_dense_nodes': 128,
                   'primary_dropout': 0.25,
                   'secondary_dropout': 0.5,
                   'model_build': 'v.0.1_nopool_growingconvs'} #_{}'.format(model_info)}
    return model_param


def compile_model(model_param):
    ''' 
    INPUT:  (1) Dictionary of model parameters
    OUTPUT: (1) Compiled (but untrained) Keras model
    Any large scale model architecture changes would happen here in 
    conjunction with adjustments to set_basic_model_param.
    '''
    print 'Compiling model...'
    model = Sequential()
    #input_img = K.placeholder((1, 3, model_param['n_cols'], model_param['n_rows']))
    #input_placeholder = T.tensor4()
    model_param_to_add = [ZeroPadding2D((1, 1),
                                        input_shape=(model_param['n_chan'],
                                                    model_param['n_rows'],
                                                    model_param['n_cols'])),
                          #input_placeholder,
                          Convolution2D(model_param['n_conv_nodes'],
                                        model_param['conv_size'],
                                        model_param['conv_size']),#,
                                        # border_mode='valid',
                                        # input_shape=(model_param['n_chan'],
                                                    # model_param['n_rows'],
                                                    # model_param['n_cols'])),
                          Activation('relu'),
                          ZeroPadding2D((1, 1)),
                          Convolution2D(model_param['n_conv_nodes'],
                                        model_param['conv_size'],
                                        model_param['conv_size']),
                          Activation('relu'),
                          ZeroPadding2D((1, 1)),
                          # MaxPooling2D(pool_size=(model_param['pool_size'],
                                                  # model_param['pool_size'])),
                          Convolution2D(model_param['n_conv_nodes'], 
                                        model_param['conv_size'],
                                        model_param['conv_size']),
                          Activation('relu'),
                          ZeroPadding2D((1, 1)),
                          Convolution2D(model_param['n_conv_nodes'], 
                                        model_param['conv_size'],
                                        model_param['conv_size']),
                          Activation('relu'),
                          # MaxPooling2D(pool_size=(model_param['pool_size'],
                                                  # model_param['pool_size'])),
                          Dropout(model_param['primary_dropout']),#,
                          Convolution2D(128, 1, 1),
                          Activation('relu'),
                          Convolution2D(3, 1, 1),
                          Activation('softmax')]
                          # Flatten(),
                          # Dense(model_param['n_dense_nodes']),
                          # Activation('relu'),
                          # Dropout(model_param['secondary_dropout']),
                          # Dense(model_param['n_classes']),
                          # Activation('softmax')]
        # for _ in range(7):
            # model_param_to_add.pop(-1)
        # #model_param_to_add += [Convolution2D(128, 1, 1)]#, init='uniform')]
        # model_param_to_add += [Convolution2D(3, 1, 1)]#, init='uniform')]

        #model_param_to_add += [Activation('softmax')]
        
        #model_param_to_add += [Activation('softmax')]
    for process in model_param_to_add:
        model.add(process)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    print 'Done.'
    return model


def fit_and_save_model(model, model_param, X_train, y_train, X_test, y_test):
    ''' 
    INPUT:  (1) Compiled (but untrained) Keras model
            (2) Dictionary of model parameters
            (3) 4D numpy array: the X training data, of shape (#train_images,
                #chan, #rows, #columns); for MNIST this is (60000, 1, 28, 28)
            (4) 1D numpy array: the training labels, y, of shape (60000,)
            (5) 4D numpy array: the X test data, of shape (#test_images, 
                #chan, #rows, #columns); for MNIST this is (10000, 1, 28, 28)
            (6) 1D numpy array: the test labels, of shape (10000,)
    OUTPUT: None, but the model will be saved to /models
    '''
    print 'Fitting model...\n'
    start = time.clock()
    def loss(y_true, y_pred):
        pass
    early_stopping_monitor = EarlyStopping(monitor='val_loss', 
                                                           patience=0,
                                                           verbose=1)
    hist = model.fit(X_train, y_train, batch_size=model_param['batch_size'], 
                    nb_epoch=model_param['n_epoch'], 
                    callbacks=[early_stopping_monitor],
                    show_accuracy=True, verbose=1,
                    validation_data=(X_test, y_test))
    print hist.history
    stop = time.clock()
    print 'Done.'
    total_run_time = (stop - start) / 60.
    score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
    print 'Test score: {}'.format(score[0])
    print 'Test accuracy: {}'.format(score[1])
    print 'Total run time: {}'.format(total_run_time)

    model_name = 'KerasBaseModel_{}'.format(model_param['model_build'])
    path_to_save_model = 'models/{}'.format(model_name)
    json_file_name = '{}.json'.format(path_to_save_model)
    weights_file_name = '{}.h5'.format(path_to_save_model)
    history_file_name = '{}.pkl'.format(path_to_save_model)
    if os.path.isfile(json_file_name) or os.path.isfile(weights_file_name):
        json_file_name = '{}_copy.json'.format(path_to_save_model)
        weights_file_name = '{}_copy.h5'.format(path_to_save_model)
        print 'Please rename the model next time to avoid conflicts!'
    json_string = model.to_json()
    open(json_file_name, 'w').write(json_string)
    model.save_weights(weights_file_name) 
    pickle.dump(hist.history, open(history_file_name, 'wb'))


if __name__ == '__main__':
    # sd, cd = load_data(lats, lngs)
    X_train, y_train, X_test, y_test = load_data('dataselected')
    # sd, cd, X, y = load_data(lats, lngs)
    # model_param = set_basic_model_param()
    # model = compile_model(model_param)
    # fit_and_save_model(model, model_param, X_train, y_train, X_test, y_test)

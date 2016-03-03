import numpy as np
np.random.seed(1234)  # for reproducibility
from sklearn.cross_validation import train_test_split
from scipy.misc import imread
import time
import os
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD
import h5py
import matplotlib # necessary to save plots remotely; comment out if local
matplotlib.use('Agg') # comment out if local
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
#import cv2
#from keras import backend as K
import theano.tensor as T
import pickle
from additional_functions import pixelwise_prediction, probas_tensor_to_pixelwise_prediction


class keras_models(object):
    def __init__(self, h=640, w=640, sub_im_width=64, sample_stride=32, 
                 n_chan=3, n_classes=4, n_epoch=10,
                 batch_size=16, pool_size=2, conv_size=3, n_conv_nodes=128,
                 n_dense_nodes=128, primary_dropout=0.25, secondary_dropout=0.5,
                 model_build='v.0.2'):
        self.h = h
        self.w = w
        self.sub_im_width = sub_im_width
        self.sample_stride = sample_stride
        self.n_chan = n_chan
        self.n_classes = n_classes
        self.n_classes_no_background = n_classes - 1
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.conv_size = conv_size
        self.n_conv_nodes = n_conv_nodes
        self.n_dense_nodes = n_dense_nodes
        self.primary_dropout = primary_dropout
        self.secondary_dropout = secondary_dropout
        self.model_build = model_build
        self.offset = sub_im_width / 2
        # 0: buildings; 1: water; 2: road, 3: background
        self.colors_to_classes = {(233, 229, 220): 3, (0, 0, 255): 1,
                                  (0, 255, 0): 2, (242, 240, 233): 0}
        self.classes_to_colors = {3: (233, 229, 220), 1: (0, 0, 255), 
                                  2: (0, 255, 0), 0: (242, 240, 233)}
  
    def load_data(self, data_dir, equal_classes=True, 
                  centerpix_or_pixelwise='centerpix'):
        all_image_filenames = os.listdir(data_dir)
        satellite_filenames = sorted([f for f in all_image_filenames 
                               if 'satellite' in f])
        segmented_filenames = sorted([f for f in all_image_filenames 
                               if 'segmented' in f])
        files_lined_up = np.all([satf[:23] == segf[:23] 
                                for satf, segf 
                                in zip(satellite_filenames, segmented_filenames)])
        print 'It is {} that your files are ordered correctly.'.format(files_lined_up)
        total_num_img = len(satellite_filenames)
        all_satellite_data = np.zeros((total_num_img, 
                                       self.h, 
                                       self.w, 
                                       self.n_chan), dtype=np.uint8)
        all_class_data_as_rgb = np.zeros((total_num_img, 
                                          self.h, 
                                          self.w, 
                                          self.n_chan), dtype=np.uint8)
        print 'Loading image data...'
        for idx, satellite_filename in enumerate(satellite_filenames):
            satellite_img = imread('{}/{}'.format(data_dir, satellite_filename))
            all_satellite_data[idx] = satellite_img[:self.h, :, :]
        for idx, segmented_filename in enumerate(segmented_filenames):
            segmented_img = imread('{}/{}'.format(data_dir, segmented_filename))
            all_class_data_as_rgb[idx] = segmented_img[:self.h, :, :]
        h_start_pxs = np.arange(0, self.h-self.sub_im_width+1, 
                                self.sample_stride)
        w_start_pxs = np.arange(0, self.w-self.sub_im_width+1, 
                                self.sample_stride)
        total_num_sampled_img = total_num_img * len(h_start_pxs) * len(w_start_pxs)
        X = np.zeros((total_num_sampled_img, 
                      self.sub_im_width, 
                      self.sub_im_width, 
                      self.n_chan), dtype=np.uint8)
        if centerpix_or_pixelwise == 'centerpix':
            y = np.zeros(total_num_sampled_img)
        elif centerpix_or_pixelwise == 'pixelwise':
            ''' More complicated. Need one-hot mapping from pixel colors. 
            all_class_data_pixelwise_as_onehot will be used later in this namespace.'''
            y = np.zeros((total_num_sampled_img, 
                        self.sub_im_width, 
                        self.sub_im_width, 
                        self.n_classes), dtype=np.uint8)
            print 'Done. \nCreating one-hot mapping from pixel colors...'
            all_class_data_pixelwise_as_onehot = np.zeros((total_num_img, 
                                                           self.h, 
                                                           self.w, 
                                                           self.n_classes),
                                                          dtype=np.float32)
            for rgb_color in self.colors_to_classes.keys():
                color_true = np.logical_and(
                            (all_class_data_as_rgb[:, :, :, 0] == rgb_color[0]),
                            (all_class_data_as_rgb[:, :, :, 1] == rgb_color[1]),
                            (all_class_data_as_rgb[:, :, :, 2] == rgb_color[2]))
                locs = np.where(color_true)
                all_class_data_pixelwise_as_onehot[locs[0], 
                                        locs[1], 
                                        locs[2],
                                        self.colors_to_classes[rgb_color]] = 1
                if rgb_color == (0, 0, 255):
                    water_true = color_true
                elif rgb_color == (0, 255, 0):
                    road_true = color_true
                elif rgb_color == (242, 240, 233):
                    building_true = color_true
                elif rgb_color == (233, 229, 220):
                    background_true = color_true
            water_or_road = np.logical_or(water_true, road_true)
            labeled_background_or_building = np.logical_or(background_true, 
                                                               building_true)
            other_locs = np.where(np.logical_or(water_or_road, 
                                       labeled_background_or_building)==False)
            background_idx = self.n_classes - 1
            all_class_data_pixelwise_as_onehot[other_locs[0],
                                    other_locs[1],
                                    other_locs[2],
                                    background_idx] = 1  # set as background
        ### LOAD Xy ###
        def load_Xy(centerpix_or_pixelwise):
            def pixelwise_y_loader():
                im_subset_as_class = all_class_data_pixelwise_as_onehot[img_idx][h_start_px:h_end_px, w_start_px:w_end_px]
                y[idx_to_write] = im_subset_as_class
            def centerpix_y_loader():
                centerpix_h = int(h_start_px + self.offset)
                centerpix_w = int(w_start_px + self.offset)
                im_centerpix_rgb = all_class_data_as_rgb[img_idx][centerpix_h, centerpix_w]
                cla = self.colors_to_classes.get(tuple(im_centerpix_rgb), 3)
                y[idx_to_write] = cla
            if centerpix_or_pixelwise == 'centerpix':
                y_loader_method = centerpix_y_loader
            elif centerpix_or_pixelwise == 'pixelwise':
                y_loader_method = pixelwise_y_loader
            idx_to_write = 0
            for img_idx in range(total_num_img):
                for h_start_px in h_start_pxs:
                    for w_start_px in w_start_pxs:
                        h_end_px = h_start_px + self.sub_im_width
                        w_end_px = w_start_px + self.sub_im_width
                        im_subset = all_satellite_data[img_idx][h_start_px:h_end_px,
                                                                w_start_px:w_end_px]
                        X[idx_to_write] = im_subset
                        y_loader_method()
                        idx_to_write += 1
            return X, y
        X, y = load_Xy(centerpix_or_pixelwise)
        print 'Done. \nReshaping image data...'
    
        if centerpix_or_pixelwise == 'pixelwise': 
            if equal_classes:
                classwise_pixcount_per_img = [np.sum(y[:, :, :, i]==1, axis=2).sum(1)
                                              for i in range(self.n_classes)]
                total_pix_in_img = self.sub_im_width**2
                min_road_pix = total_pix_in_img * 0.02
                min_water_pix = total_pix_in_img * 0.02
                min_building_pix = total_pix_in_img * 0.02
                building_img_locs = classwise_pixcount_per_img[0] > min_building_pix
                water_img_locs = classwise_pixcount_per_img[1] > min_water_pix
                road_img_locs = classwise_pixcount_per_img[2] > min_road_pix
                not_just_background_locs = np.logical_or(
                                            np.logical_or(road_img_locs,
                                                          water_img_locs),
                                                         building_img_locs)
                X = X[not_just_background_locs]
                y = y[not_just_background_locs]
                del not_just_background_locs
                del water_img_locs
                del road_img_locs
        elif centerpix_or_pixelwise == 'centerpix':
            if equal_classes:
                y_without_background = y[y != 3]
                X_without_background = X[y != 3]
                len_smallest_class = np.min([np.sum(y_without_background==i) 
                    for i in range(self.n_classes_no_background)])
                X_eq = np.zeros((len_smallest_class*self.n_classes_no_background,
                                 self.sub_im_width, 
                                 self.sub_im_width, 
                                 self.n_chan), dtype=np.uint8)
                y_eq = np.zeros(len_smallest_class*self.n_classes_no_background)
                X_by_class = [np.where(y_without_background==i)[0][:len_smallest_class]
                              for i in range(self.n_classes_no_background)]
                for i in range(self.n_classes_no_background):
                    X_eq[i::self.n_classes_no_background] = X_without_background[X_by_class[i]]
                    y_eq[i::self.n_classes_no_background] = y_without_background[X_by_class[i]]
                X = X_eq
                y = y_eq
        X = X.astype('float32')
        X /= 255.
        num_all_img = X.shape[0]
        X = X.reshape((num_all_img, self.n_chan, self.sub_im_width, self.sub_im_width))
        if centerpix_or_pixelwise == 'centerpix':
            y = np_utils.to_categorical(y, self.n_classes_no_background)
        elif centerpix_or_pixelwise == 'pixelwise':
            y = y.reshape(num_all_img, self.sub_im_width**2, self.n_classes)
        print 'Done.'
        return X, y


    def compile_model(self, segmentation=False):
        ''' 
        INPUT:  (1) Dictionary of model parameters
        OUTPUT: (1) Compiled (but untrained) Keras model
        Any large scale model architecture changes would happen here in 
        conjunction with adjustments to set_basic_model_param.
        '''
        print 'Compiling model...'
        model = Sequential()
        model_param_to_add = [ZeroPadding2D((1, 1),
                                            input_shape=(self.n_chan,
                                                         self.sub_im_width,
                                                         self.sub_im_width)),
                            # Convolution2D(self.n_conv_nodes/2,
                            Convolution2D(self.n_conv_nodes/2,
                                          self.conv_size,
                                          self.conv_size), 
                            LeakyReLU(alpha=0.01),
                            # BatchNormalization(),
                            # Activation('relu'),
                            ZeroPadding2D((1, 1)),
                            Convolution2D(self.n_conv_nodes/2,
                                          self.conv_size,
                                          self.conv_size),
                            LeakyReLU(alpha=0.01),
                            BatchNormalization(),
                            # Activation('relu'),
                            # MaxPooling2D(pool_size=(self.pool_size, self.pool_size)),
                            ZeroPadding2D((1, 1)),
                            Convolution2D(self.n_conv_nodes,
                                          self.conv_size,
                                          self.conv_size),
                            LeakyReLU(alpha=0.01),
                            # BatchNormalization(),
                            # Activation('relu'),
                            ZeroPadding2D((1, 1)),
                            Convolution2D(self.n_conv_nodes,
                                          self.conv_size,
                                          self.conv_size),
                            LeakyReLU(alpha=0.01),
                            BatchNormalization(),
                            # Activation('relu'),
                            # MaxPooling2D(pool_size=(self.pool_size, self.pool_size)),
                            # Dropout(self.primary_dropout),
                            Flatten(),
                            Dense(self.n_dense_nodes),
                            LeakyReLU(alpha=0.01),
                            # Activation('relu'),
                            # Dense(self.n_dense_nodes*2),
                            # LeakyReLU(alpha=0.01),
                            # Activation('relu'),
                            # Dropout(self.secondary_dropout),
                            Dense(self.n_classes_no_background),
                            Activation('softmax')]
        for process in model_param_to_add:
            model.add(process)
        # sgd = SGD(lr=0.001, decay=2e-4, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        print 'Done.'
        return model
    

    def behead_model(self, model, num_layers_to_chop=5):
        print 'Beheading {} layers from model...'.format(num_layers_to_chop)
        model_layers = model.layers
        for layer in range(num_layers_to_chop):
            model_layers.pop(-1)
        print 'Done.'
        return model_layers


    def add_pixelwise_head(self, model_layers):
        print 'Adding convolutional layers to model...'
        model = Sequential()
        model_layers += [Convolution2D(self.n_classes, 1, 1)]
        model_layers += [Reshape((self.sub_im_width*self.sub_im_width, 
                                        self.n_classes))]
        model_layers += [Activation('softmax')]
        for process in model_layers:
            model.add(process)
        print 'Done.'
        return model
        
   
    def load_model_weights(self, path_to_model, centerpix_or_pixelwise):
        ''' 
        INPUT:  (1) String: The path to the saved model architecture and weights, 
                    not including .json or .h5 at the end
        OUTPUT: (1) Trained and compiled Keras model
        '''
        print 'Loading model weights...'
        json_file_name = '{}.json'.format(path_to_model)
        weights_file_name = '{}.h5'.format(path_to_model)
        if centerpix_or_pixelwise == 'pixelwise':
            weights_file = h5py.File(weights_file_name)
            model_structure = model_from_json(open(json_file_name).read())
            model_layers = self.behead_model(model_structure, num_layers_to_chop=5)
            model = self.add_pixelwise_head(model_layers)
            # model = self.compile_model(segmentation=True)
            for layer_num in range(weights_file.attrs['nb_layers']):
                print 'Loading weights for layer {}'.format(layer_num)
                if layer_num >= len(model.layers)-5:
                    break
                weights_obj = weights_file['layer_{}'.format(layer_num)]
                weights = [weights_obj['param_{}'.format(p)]
                           for p in range(weights_obj.attrs['nb_params'])]
                model.layers[layer_num].set_weights(weights)
            weights_file.close()
        elif centerpix_or_pixelwise == 'centerpix':
            model = model_from_json(open(json_file_name).read())
            model.load_weights(weights_file_name)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        print 'Done loading model weights.'
        return model

 
    def fit_and_save_model(self, model, model_name_append, X, y):
        ''' 
        INPUT:  (1) Compiled (but untrained) Keras model
        OUTPUT: None, but the model will be saved to /models
        '''
        print 'Fitting model...\n'
        start = time.clock()
        early_stopping_monitor = EarlyStopping(monitor='val_loss', 
                                            patience=2,
                                            verbose=1)
        hist = model.fit(X, y, batch_size=self.batch_size,
                        nb_epoch=self.n_epoch,
                        callbacks=[early_stopping_monitor],
                        show_accuracy=True, verbose=1,
                        validation_split=0.166)
        print hist.history
        stop = time.clock()
        print 'Done.'
        total_run_time = (stop - start) / 60.
        print 'Total run time: {}'.format(total_run_time)
        model_name = 'KerasBaseModel_{}'.format(self.model_build)
        path_to_save_model = 'models/{}_{}'.format(model_name, model_name_append)
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
        return model, path_to_save_model


def run_centerpix_defined_model(data_folder, name_append):
    km = keras_models(n_epoch=10, sub_im_width=64, batch_size=32, n_classes=4,
                      h=620, sample_stride=20, n_conv_nodes=128, n_dense_nodes=128)
    X, y = km.load_data(data_folder, equal_classes=True,
                        centerpix_or_pixelwise='centerpix')
    # return X, y
    name_to_append = 'centerpix_{}'.format(name_append)
    model = km.compile_model(segmentation=False)
    model, path_to_centerpix_model = km.fit_and_save_model(model, name_to_append, X, y)
    return X, y, model, path_to_centerpix_model

def run_pixelwise_defined_model(data_folder, path_to_centerpix_model, name_append):
    km = keras_models(n_epoch=10, sub_im_width=64, batch_size=64, n_classes=4,
                      sample_stride=64, n_conv_nodes=128, n_dense_nodes=128)
    X_segmented, y_segmented = km.load_data(data_folder, equal_classes=True,
                                            centerpix_or_pixelwise='pixelwise')
    # segmented_model = km.compile_model(segmentation=True)
    segmented_model = km.load_model_weights(path_to_centerpix_model, 
                                            centerpix_or_pixelwise='pixelwise')
    name_to_append = 'pixelwise_{}'.format(name_append)
    segmented_model, path_to_pixelwise_model = km.fit_and_save_model(segmented_model, 
                                                                     name_to_append,
                                                                     X_segmented, 
                                                                     y_segmented) 
    return X_segmented, y_segmented, segmented_model
    # pixelwise_prediction(segmented_model, 'data640x640zoom18/lat_28.48830,long_-81.5087_satellite.png', 'data640x640zoom18/lat_28.48830,long_-81.5087_segmented.png')

if __name__ == '__main__':
    data_folder = 'data640x640new2Colzoom18'
    # name_append = '2xoversampled_nobatchnorm_32batch_c163264128d128'#_zeroinit'
    name_append = 'oversampled_no12811_equalerclasses'
    path_to_centerpix_model = 'models/KerasBaseModel_v.0.2_centerpix_oversampled'
    # X, y, model, path_to_centerpix_model = run_centerpix_defined_model(data_folder, name_append)
    X_seg, y_seg, seg_model = run_pixelwise_defined_model(data_folder, path_to_centerpix_model, name_append)

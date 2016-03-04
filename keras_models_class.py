import numpy as np
np.random.seed(1234)  # for reproducibility
from scipy.misc import imread, imsave
import time
import os
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.optimizers import SGD
import h5py
import matplotlib # necessary to save plots remotely; comment out if local
matplotlib.use('Agg') # comment out if local
from keras.callbacks import EarlyStopping
import theano
import pickle


class KerasModels(object):
    ''' 
    An object for making models that predict in either a pixelwise
    or classwise (ie. the img is defined by center pixel) fashion and 
    keeping track of hyperparameters. 
    '''
    def __init__(self, h=640, w=640, true_imwidth=640, sub_im_width=64, 
                 sample_stride=32, n_chan=3, n_classes=4, n_epoch=10, 
                 batch_size=16, pool_size=2, conv_size=3, n_conv_nodes=128,
                 n_dense_nodes=128, primary_dropout=0.25, secondary_dropout=0.5,
                 model_build='v.0.2'):
        ''' 
        INPUT:  (1) integer 'h': the height (in pixels) to read input images up to;
                    set at 20 pixels less than image height for a standard google 
                    maps image to cut off the google watermark
                (2) integer 'w': the width (in pixels) to read input images up to
                (3) integer 'true_imwidth': the true width (in pixels) of the image;
                    necessary in case training only on a subset of images but
                    predicting on entirety of image
                (4) integer 'sub_im_width': the width (in pixels) to 
                    make subsampled images
                (5) integer 'sample_stride': how frequently (in pixels) 
                    to sample the parent image for the subsampled images
                (6) integer 'n_chan': number of channels in the image
                (7) integer 'n_classes': number of classes (including background)
                (8) integer 'n_epoch': number of classes to train for
                (9) integer 'batch_size': the training batch size
                (10) integer 'pool_size': the edge length (in pixels) for pooling
                (11) integer 'conv_size': the edge length (in pixels) 
                    for convolution kernel 
                (12) integer 'n_conv_nodes': the parent number of convolutional nodes
                    (see compile_model for details)
                (13) integer 'n_dense_nodes': the parent number of dense nodes
                    (see compile_model for details)
                (14) float 'primary_dropout': the dropout after all conv layers
                (15) float 'secondary_dropout': the dropout after dense layers
                (16) string 'model_build': the version of the model

        Initialize all hyperparameters. 
        '''
        self.h = h
        self.w = w
        self.true_imwidth = true_imwidth
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
        ''' 
        INPUT:  (1) string 'data_dir': the location of the satellite and 
                    segmented images
                (2) bool 'equal_classes': make classes equally well represented
                    in training and testing data
                (3) string 'centerpix_or_pixelwise': 'centerpix' to load
                    a class labels defined by the center pixel
                    of each sub image; 'pixelwise' to load a 4D tensor of
                    pixelwise class labels
        OUTPUT: (1) 4D tensor 'X': All subset image data, of shape
                    (num_sampled_img, n_chan, sub_im_width, sub_im_width)
                (2a) 2D tensor 'y': if 'centerpix', classes as categorical
                    of shape (num_sampled_img, n_classes-1); background
                    will not be trained on
                (2b) 4D tensor 'y': if 'pixelwise', classes as categorical 
                    of shape (num_sampled_img, sub_im_width**2, n_classes); 
                    background is included
        '''
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
            all_satellite_data[idx] = satellite_img[:self.h, :self.w, :self.n_chan]
        for idx, segmented_filename in enumerate(segmented_filenames):
            segmented_img = imread('{}/{}'.format(data_dir, segmented_filename))
            all_class_data_as_rgb[idx] = segmented_img[:self.h, :self.w, :self.n_chan]
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
            ''' For pixelwise class labels, we need a one hot mapping from the 
            colors in the segmented image. all_class_data_pixelwise_as_onehot 
            contains this information and will be subset simultaneously with X
            later in this namespace. '''
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
            ''' 
            INPUT:  (1) string: 'centerpix' if classes are to be defined by the 
                        class of the center pixel; 'pixelwise' if pixelwise 
                        class data is desired
            OUTPUT: (1) 4D numpy array: subset training data of shape
                        (total_num_sampled_img, sub_im_width, sub_im_width, n_chan)
                    (2a) 1D numpy array: centerpix defined class labels of shape
                        (total_num_sampled_img)
                    (2b) 4D numpy array: subset pixelwise one-hot class labels of shape
                        (total_num_sampled_img, sub_im_width, sub_im_width, n_classes)
            '''
            def pixelwise_y_loader():
                ''' Helper function if 'pixelwise' classes.'''
                im_subset_as_class = (
                    all_class_data_pixelwise_as_onehot[img_idx][h_start_px:h_end_px, 
                                                                w_start_px:w_end_px])
                y[idx_to_write] = im_subset_as_class
            def centerpix_y_loader():
                ''' Helper function if 'centerpix' classes.'''
                centerpix_h = int(h_start_px + self.offset)
                centerpix_w = int(w_start_px + self.offset)
                im_centerpix_rgb = all_class_data_as_rgb[img_idx][centerpix_h, 
                                                                  centerpix_w]
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
        elif centerpix_or_pixelwise == 'centerpix':
            if equal_classes:
                y_without_background = y[y != 3]
                X_without_background = X[y != 3]
                len_smallest_class = np.min(
                                     [np.sum(y_without_background==i) 
                                      for i in range(self.n_classes_no_background)])
                X_eq = np.zeros((len_smallest_class*self.n_classes_no_background,
                                 self.sub_im_width, 
                                 self.sub_im_width, 
                                 self.n_chan), dtype=np.uint8)
                y_eq = np.zeros(len_smallest_class*self.n_classes_no_background)
                X_by_class = [np.where(y_without_background==i)[0][:len_smallest_class]
                              for i in range(self.n_classes_no_background)]
                for i in range(self.n_classes_no_background):
                    X_eq[i::self.n_classes_no_background] = (
                            X_without_background[X_by_class[i]])
                    y_eq[i::self.n_classes_no_background] = (
                            y_without_background[X_by_class[i]])
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


    def compile_model(self):
        ''' 
        INPUT:  None
        OUTPUT: (1) Compiled (but untrained) Keras model
        '''
        print 'Compiling model...'
        model = Sequential()
        model_param_to_add = [ZeroPadding2D((1, 1),
                                            input_shape=(self.n_chan,
                                                         self.sub_im_width,
                                                         self.sub_im_width)),
                            # Convolution2D(self.n_conv_nodes/2,
                            Convolution2D(self.n_conv_nodes/8,
                                          self.conv_size,
                                          self.conv_size), 
                            LeakyReLU(alpha=0.01),
                            # BatchNormalization(),
                            # Activation('relu'),
                            ZeroPadding2D((1, 1)),
                            Convolution2D(self.n_conv_nodes/4,
                                          self.conv_size,
                                          self.conv_size),
                            LeakyReLU(alpha=0.01),
                            # BatchNormalization(),
                            # Activation('relu'),
                            # MaxPooling2D(pool_size=(self.pool_size, self.pool_size)),
                            ZeroPadding2D((1, 1)),
                            Convolution2D(self.n_conv_nodes/2,
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
                            # BatchNormalization(),
                            # MaxPooling2D(pool_size=(self.pool_size, self.pool_size)),
                            Dropout(self.primary_dropout),
                            Flatten(),
                            # Dropout(self.primary_dropout),
                            Dense(self.n_dense_nodes),
                            LeakyReLU(alpha=0.01),
                            # Activation('relu'),
                            # Dense(self.n_dense_nodes*2),
                            # LeakyReLU(alpha=0.01),
                            # Activation('relu'),
                            Dropout(self.secondary_dropout),
                            Dense(self.n_classes_no_background),
                            Activation('softmax')]
        for process in model_param_to_add:
            model.add(process)
        # sgd = SGD(lr=0.001, decay=2e-4, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        print 'Done.'
        return model
    

    def behead_model(self, model):
        ''' 
        INPUT:  (1) Trained model
        OUTPUT: (1) list: all convolutional layers and activations
        '''
        print 'Beheading model...'
        model_layers = model.layers
        behead_idx = 0
        layer_names = [model_layers[idx].get_config()['name']
                       for idx in range(len(model_layers))]
        for idx, layer_name in enumerate(layer_names):
            if idx > 0:
                if layer_name == 'Flatten' and layer_names[idx-1] != 'Dropout':
                    behead_idx = idx
                elif layer_name == 'Flatten' and layer_names[idx-1] == 'Dropout':
                    behead_idx = idx - 1
        print 'Done.'
        return model_layers[:behead_idx]


    def add_pixelwise_head(self, model_layers):
        ''' 
        INPUT:  (1) list of layers to add the convolutional layers to
        OUTPUT: (1) Keras model

        Add fully convolutional layers in place of the dense layers 
        at the end of the model.
        '''
        print 'Adding convolutional layers to model...'
        model = Sequential()
        # model_layers += [Convolution2D(self.n_dense_nodes, 1, 1)]
        # model_layers += [Activation('relu')]
        model_layers += [Convolution2D(self.n_classes, 1, 1)]
        model_layers += [Reshape((self.sub_im_width*self.sub_im_width, 
                                        self.n_classes))]
        model_layers += [Activation('softmax')]
        for process in model_layers:
            model.add(process)
        print 'Done.'
        return model
        
   
    def load_model_weights(self, path_to_model, untilflatten_or_all):
        ''' 
        INPUT:  (1) String: The path to the saved model architecture and weights, 
                    not including .json or .h5 at the end
                (2) string: if 'untilflatten' the model weights will only be
                    loaded for convolutional layers; if 'all' then all model
                    weights will be loaded
        OUTPUT: (1) Trained and compiled Keras model
        '''
        print 'Loading model weights...'
        json_file_name = '{}.json'.format(path_to_model)
        weights_file_name = '{}.h5'.format(path_to_model)
        if untilflatten_or_all == 'untilflatten':
            weights_file = h5py.File(weights_file_name)
            model_structure = model_from_json(open(json_file_name).read())
            model_layers = self.behead_model(model_structure)
            model = self.add_pixelwise_head(model_layers)
            for layer_num in range(weights_file.attrs['nb_layers']):
                print 'Loading weights for layer {}'.format(layer_num)
                if layer_num >= len(model.layers)-5:
                    break
                weights_obj = weights_file['layer_{}'.format(layer_num)]
                weights = [weights_obj['param_{}'.format(p)]
                           for p in range(weights_obj.attrs['nb_params'])]
                model.layers[layer_num].set_weights(weights)
            weights_file.close()
        elif untilflatten_or_all == 'all':
            model = model_from_json(open(json_file_name).read())
            model.load_weights(weights_file_name)
        sgd = SGD(lr=0.1, decay=2e-4, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
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
                                            patience=1,
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


    def probas_tensor_to_pixelwise_prediction(self, model, X_sub):
        y_pred = model.predict(X_sub.reshape(1, 3, self.sub_im_width, 
                                                       self.sub_im_width)/255.)
        y_pred = y_pred.reshape(self.sub_im_width, self.sub_im_width, self.n_classes)
        pixelwise_prediction = np.argmax(y_pred[:, :, :], axis=2)
        pixelwise_color = np.zeros((self.sub_im_width, self.sub_im_width, 3))
        for class_num in range(self.n_classes):
            class_color = self.classes_to_colors[class_num]
            class_locs = np.where(pixelwise_prediction == class_num)
            class_locs_Xdim = class_locs[0]
            class_locs_Ydim = class_locs[1]
            for RGB_idx in range(self.n_chan):
                pixelwise_color[class_locs_Xdim, 
                                class_locs_Ydim,
                                RGB_idx] = class_color[RGB_idx]/255.
        return pixelwise_prediction, pixelwise_color


    def pixelwise_prediction(self, model, X_test_img_filename, y_test_img_filename):
        # offset = int(sub_im_width/2.)
        X_test_img = imread(X_test_img_filename)
        y_test_img = imread(y_test_img_filename)
        y_pred_img = np.zeros((self.true_imwidth, self.true_imwidth, 3))
        h_start_pxs = np.arange(0, self.true_imwidth, self.sub_im_width)
        w_start_pxs = np.arange(0, self.true_imwidth, self.sub_im_width)
        total_preds = len(h_start_pxs) * len(w_start_pxs)
        idx_to_write = 0
        # classwise_correct = {i: 0 for i in range(len(color_to_class))}
        for h_start_px in h_start_pxs:
            for w_start_px in w_start_pxs:
                print 'Calculating probas for part {} of {}'.format(idx_to_write,
                                                                    total_preds)
                h_end_px = h_start_px + self.sub_im_width
                w_end_px = w_start_px + self.sub_im_width
                im_subset = X_test_img[h_start_px:h_end_px, w_start_px:w_end_px]
                #im_subset_rgb = y_with_border[h_start_px+offset, w_start_px+offset]
                #im_subset_rgb = y_test_img[h_start_px+offset, w_start_px+offset]
                #y_true_temp = color_to_class.get(tuple(im_subset_rgb))
                #y_pred = model.predict_classes(im_subset, verbose=0)
                pixelwise_prediction, pixelwise_color = self.probas_tensor_to_pixelwise_prediction(model, im_subset)
                y_pred_img[h_start_px:h_end_px, w_start_px:w_end_px] = pixelwise_color
                # if y_true_temp == y_pred:
                    # classwise_correct[y_true_temp] += 1
                # y_true[idx_to_write] = y_true_temp
                #color_to_write = class_to_color[y_pred[0]]
                #y_pred_img[h_start_px, w_start_px] = color_to_write
                idx_to_write +=1
        # plt.imshow(y_pred_img)
        # plt.show()
        y_split_filename = y_test_img_filename.split('/')
        y_pred_filename = 'preds/{}_pred.png'.format(y_split_filename[1][:-4])
        print 'saving at {}_pred.png'.format(y_pred_filename)
        imsave(y_pred_filename, y_pred_img)
        # classwise_accs = {i: (classwise_correct[i]/float(len(y_true==1)))
                        # for i in range(len(class_to_color))}
        #return y_pred_img#  X_with_border, y_with_border, y_pred_img, classwise_accs


    def get_activations(model, layer, X_batch):
        '''
        INPUT:  (1) Keras Sequential model object
                (2) integer: The layer to extract weights from
                (3) 4D numpy array: All the X data you wish to extract
                    activations for
        OUTPUT: (1) numpy array: Activations for that layer
        '''
        input_layer = model.layers[0].input
        specified_layer_output = model.layers[layer].get_output(train=False)
        get_activations = theano.function([input_layer],
                                        specified_layer_output,
                                        allow_input_downcast=True)
        activations = get_activations(X_batch)
        return activations



def run_centerpix_defined_model(data_folder, name_append):
    km = KerasModels(n_epoch=10, sub_im_width=64, batch_size=32, n_classes=4,
                      h=620, sample_stride=20, n_conv_nodes=128, n_dense_nodes=128)
    X, y = km.load_data(data_folder, equal_classes=True,
                        centerpix_or_pixelwise='centerpix')
    name_to_append = 'centerpix_{}'.format(name_append)
    model = km.compile_model()
    model, path_to_centerpix_model = km.fit_and_save_model(model, name_to_append, X, y)
    return X, y, model, path_to_centerpix_model


def run_pixelwise_defined_model(data_folder, path_to_centerpix_model, name_append):
    km = KerasModels(n_epoch=10, sub_im_width=64, batch_size=64, n_classes=4,
                      sample_stride=64, n_conv_nodes=128, n_dense_nodes=128)
    X_segmented, y_segmented = km.load_data(data_folder, equal_classes=True,
                                            centerpix_or_pixelwise='pixelwise')
    segmented_model = km.load_model_weights(path_to_centerpix_model, 
                                            untilflatten_or_all='untilflatten')
    name_to_append = 'pixelwise_{}'.format(name_append)
    segmented_model, path_to_pixelwise_model = km.fit_and_save_model(segmented_model, 
                                                                     name_to_append,
                                                                     X_segmented, 
                                                                     y_segmented) 
    return X_segmented, y_segmented, segmented_model
    pixelwise_prediction(segmented_model, 'data640x640zoom18/lat_28.48830,long_-81.5087_satellite.png', 'data640x640zoom18/lat_28.48830,long_-81.5087_segmented.png')


def load_model_and_make_pred():
    ''' Lazy fn to run for prediction in an ipython session.'''
    from keras_models_class import KerasModels
    km  = KerasModels(n_epoch=10, sub_im_width=64, batch_size=64, n_classes=4,
                    sample_stride=64, n_conv_nodes=128, n_dense_nodes=128)
    fsat = 'data640x640new2Colzoom18/lat_26.03,long_-80.25_satellite.png'
    fseg = 'data640x640new2Colzoom18/lat_26.03,long_-80.25_segmented.png'
    model = km.load_model_weights('models/KerasBaseModel_v.0.2_pixelwise_oversampled_no12811_equalerclasses', untilflatten_or_all='all')
    km.pixelwise_prediction(model, fsat, fseg)


if __name__ == '__main__':
    data_folder = 'data640x640newColzoom18'
    # name_append = '2xoversampled_nobatchnorm_32batch_c163264128128d128_pooling'#_zeroinit'
    name_append = 'oversampled_no12811_equalerclasses_sgd'
    # path_to_centerpix_model = 'models/KerasBaseModel_v.0.2_centerpix_{}'.format(name_append)
    path_to_centerpix_model = 'models/KerasBaseModel_v.0.2_centerpix_2xoversampled_nobatchnorm_32batch_c163264128d128'
    # X, y, model, path_to_centerpix_model = run_centerpix_defined_model(data_folder, name_append)
    X_seg, y_seg, seg_model = run_pixelwise_defined_model(data_folder, path_to_centerpix_model, name_append)

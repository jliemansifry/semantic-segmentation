import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from PIL import Image, ImageOps
from scipy.misc import imread, imsave
from keras.utils import np_utils
from keras_pixelwise import compile_model, set_basic_model_param
import cv2
import theano
import h5py
from scipy.ndimage.filters import median_filter, maximum_filter

def load_model(path_to_model):
    ''' 
    INPUT:  (1) String: The path to the saved model architecture and weights, 
                not including .json or .h5 at the end
    OUTPUT: (1) Trained and compiled Keras model
    '''
    json_file_name = '{}.json'.format(path_to_model)
    weights_file_name = '{}.h5'.format(path_to_model)
    model = model_from_json(open(json_file_name).read())
    model.load_weights(weights_file_name)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model


def probas_tensor_to_pixelwise_prediction(model, X_sub):
    y_pred = model.predict(X_sub/255.).reshape((1, 64, 64, 3))
    pixelwise_prediction = np.argmax(y_pred[0, :, :, :], axis=2)
    class_to_color = {0: (0, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
    pixelwise_color = np.zeros((64, 64, 3))
    for class_num in range(3):
        class_color = class_to_color[class_num]
        class_locs = np.where(pixelwise_prediction == class_num)
        class_locs_Xdim = class_locs[0]
        class_locs_Ydim = class_locs[1]
        for RGB_idx in range(3):
            pixelwise_color[class_locs_Xdim, 
                            class_locs_Ydim,
                            RGB_idx] = class_color[RGB_idx]/255.
    return pixelwise_prediction, pixelwise_color


def apply_filter(filename, typ='median'):
    img = imread(filename)
    if typ == 'median':
        return median_filter(img, 3)
    else:
        return maximum_filter(img, 2)


def pixelwise_prediction(model, X_test_img_filename, y_test_img_filename, h=640, w=640, sub_im_width=64):
    imwidth = 640
    # offset = int(sub_im_width/2.)
    X_test_img = imread(X_test_img_filename)
    y_test_img = imread(y_test_img_filename)
    y_pred_img = np.zeros((imwidth, imwidth, 3))
    h_start_pxs = np.arange(0, imwidth, 64)
    w_start_pxs = np.arange(0, imwidth, 64)
    color_to_class = {(0, 0, 0): 0, (0, 0, 255): 1, (0, 255, 0): 2}  # 0: background; 1: water; 2: road
    # class_to_color = {0: (0, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
    idx_to_write = 0
    # classwise_correct = {i: 0 for i in range(len(color_to_class))}
    for h_start_px in h_start_pxs:
        for w_start_px in w_start_pxs:
            print idx_to_write
            h_end_px = h_start_px + sub_im_width
            w_end_px = w_start_px + sub_im_width
            im_subset = X_test_img[h_start_px:h_end_px, w_start_px:w_end_px]
            im_subset = im_subset.reshape(1, 3, sub_im_width, sub_im_width)
            #im_subset_rgb = y_with_border[h_start_px+offset, w_start_px+offset]
            #im_subset_rgb = y_test_img[h_start_px+offset, w_start_px+offset]
            #y_true_temp = color_to_class.get(tuple(im_subset_rgb))
            #y_pred = model.predict_classes(im_subset, verbose=0)
            pixelwise_prediction, pixelwise_color = probas_tensor_to_pixelwise_prediction(model, im_subset)
            y_pred_img[h_start_px:h_end_px, w_start_px:w_end_px] = pixelwise_color
            # if y_true_temp == y_pred:
                # classwise_correct[y_true_temp] += 1
            # y_true[idx_to_write] = y_true_temp
            #color_to_write = class_to_color[y_pred[0]]
            #y_pred_img[h_start_px, w_start_px] = color_to_write
            idx_to_write +=1
    plt.imshow(y_pred_img)
    plt.show()
    print 'saving at {}_pred.png'.format(y_test_img_filename[:-4])
    imsave('{}_pred.png'.format(y_test_img_filename[:-4]), y_pred_img)
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


if __name__ == '__main__':
    model = load_model('models/KerasBaseModel_v.0.1_nopool_pixelwise')
    pixelwise_prediction(model, 'data640x640/lat_27.5,long_-81.455_satellite.png', 'data640x640/lat_27.5,long_-81.455_segmented.png')

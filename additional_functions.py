import numpy as np
from keras.models import model_from_json
from PIL import Image, ImageOps
from scipy.misc import imread, imsave
from keras.utils import np_utils
import cv2


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


def pixelwise_prediction(model, X_test_img_filename, y_test_img_filename, sub_im_width=64):
    usable_width = 640
    usable_height = 380
    offset = int(sub_im_width/2.)
    X_test_img = imread(X_test_img_filename)
    y_test_img = imread(y_test_img_filename)
    X_with_border = cv2.copyMakeBorder(X_test_img, 
                                       offset, offset, offset, offset, 
                                       cv2.BORDER_CONSTANT, value=(0,0,0))
    y_with_border = cv2.copyMakeBorder(y_test_img, 
                                       offset, offset, offset, offset,
                                       cv2.BORDER_CONSTANT, value=(0,0,0))
    y_pred_img = np.zeros((400, usable_width, 3))
    h_start_pxs = np.arange(0, usable_width)
    w_start_pxs = np.arange(0, usable_height)
    y_true = np.zeros((usable_width*usable_height))
    color_to_class = {(0, 0, 0): 0, (0, 0, 255): 1, (0, 255, 0): 2}  # 0: background; 1: water; 2: road
    class_to_color = {0: (0, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
    idx_to_write = 0
    classwise_correct = {i: 0 for i in range(len(color_to_class))}
    for h_start_px in h_start_pxs:
        for w_start_px in w_start_pxs:
            print idx_to_write
            h_end_px = h_start_px + sub_im_width
            w_end_px = w_start_px + sub_im_width
            im_subset = X_with_border[h_start_px:h_end_px, w_start_px:w_end_px]
            im_subset = im_subset.reshape(1, 3, sub_im_width, sub_im_width)
            im_subset_rgb = y_with_border[h_start_px+offset, w_start_px+offset]
            y_true_temp = color_to_class.get(tuple(im_subset_rgb))
            y_pred = model.predict_classes(im_subset, verbose=0)
            if y_true_temp == y_pred:
                classwise_correct[y_true_temp] += 1
            y_true[idx_to_write] = y_true_temp
            color_to_write = class_to_color[y_pred[0]]
            y_pred_img[h_start_px, w_start_px] = color_to_write
            idx_to_write +=1
    imsave('{}_pred.png'.format(y_test_img_filename[:-4]), y_pred_img)
    classwise_accs = {i: (classwise_correct[i]/float(len(y_true==1)))
                      for i in range(len(class_to_color))}
    return X_with_border, y_with_border, y_pred_img, classwise_accs

if __name__ == '__main__':
    model = load_model('models/KerasBaseModel_v.0.1')
    Xti, yti, ypi, classwise_accs = pixelwise_prediction(model, 'data/lat_28.26229,long_-81.3301_satellite.png', 'data/lat_28.26229,long_-81.3301_segmented.png')

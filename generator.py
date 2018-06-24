__author__ = 'Sargam Modak'

import json, numpy as np
from keras.engine.training import _make_batches
from skimage.io import imread
from augmentation import resizeImage,\
    randomcropImage, augmentData
from keras.utils import to_categorical

def image_generator(json_file, batch_size=32, num_classes=2, is_training=True):
    """
    
    :param json_file: file containing the path of the images going to be used for training or validation
    :param batch_size: number of images in one batch for training or validation
    :param num_classes: number of classes to classify the images into
    :param is_training: whether the generator is being used for training or not
    :return:
    """
    image_files_path = json.load(open(json_file))
    len_image_files_path = len(image_files_path)
    print len_image_files_path
    image_files_path = np.array(image_files_path)
    while True:
        batches = _make_batches(size=len_image_files_path,
                                batch_size=batch_size)
        for start, end in batches:
            arr = []
            Y = []
            cur_batch = image_files_path[start:end]
            print "current Batch size", len(cur_batch)
            for image_path in cur_batch:
                print image_path
                img = imread(fname=image_path)
                
                print img.shape
                if len(img.shape)!=3:
                    continue
                img = resizeImage(image=img,
                                  newsize=256)
                if is_training:
                    img = randomcropImage(image=img,
                                          width=224,
                                          height=224)
                    img = augmentData(image=img,
                                      operations=['horizontal_flip',
                                                  'noise',
                                                  'brightness',
                                                  'contrast',
                                                  'translation'])
                else:
                    h, w, c = img.shape
                    center_h = h / 2
                    center_w = w / 2
                    new_x1 = center_w - 112
                    new_y1 = center_h - 112
                    new_x2 = center_w + 112
                    new_y2 = center_h + 112
                    img = img[new_y1:new_y2, new_x1:new_x2]
                arr.append(img)
                folder_name = image_path.split('/')[2]
                cls = folder_name.split('_')[-1]
                if cls=='fog':
                    Y.append(1)
                else:
                    Y.append(0)
            arr = np.array(arr)
            arr.astype('float32')
            arr /= 255.
            arr -= 0.5
            arr *= 2.
            Y = to_categorical(y=Y,
                               num_classes=num_classes)
            print "arr len = ", arr.shape, "Y len = ", Y.shape
            yield (arr, Y)



model = Sequential()
model.add(Dense(50, input_dim=max_words))
# model.add(BatchNormalization())
model.add(Activation('tanh'))

model.add(Dropout(0.4))
model.add(Dense(25, kernel_initializer='uniform'))
# model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(12))
# model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dense(num_classes))
# model.add(Activation('softmax))
model.summary()
input_lyr=Input(shape=(max_words,))
sequential_model_out = model(input_lyr)
actual_out=Activation('softmax', name='actual_out')(sequential_model_out)
# acept_rej_layer=Dense(num_classes)(sequential_model_out)
acept_rej_out=Dense(1,activation='sigmoid', name='acc_rej_out')(sequential_model_out)
ARJ_model=Model(input=input_lyr,outputs=[actual_out, acept_rej_out])
#ARJ_model=Model(input=input_lyr,outputs=actual_out
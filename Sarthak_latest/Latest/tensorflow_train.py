from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, InputLayer, Lambda
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.activations import sigmoid,softmax
from tensorflow.python.keras.losses import categorical_crossentropy,binary_crossentropy
from skimage.io import imread
from tensorflow.python.framework import graph_util
import pandas as pd

from sklearn.metrics import accuracy_score
from data import DataGenerator
from arch.inception_resnet_v1 import inception_resnet_v1
import facenet
import argparse


seed = 128
rng = np.random.RandomState(seed)

slim = tf.contrib.slim
image_size = 160
num_samples = 13143

# initial_learning_rate = 0.001
learning_rate_decay_factor = 0.9
decay_steps = 4000

with open("/media/sarthak11/DATA-2/Datasets_2/lfw/Alter/Imagepaths_Final.txt") as f:
    imagepaths=f.readlines()
imagepaths=[x.strip() for x in imagepaths]
output_classes=[3,3,2,4,2,3,2,2,3,3,2,3]
for i in range(41):
    output_classes.append(1)

train_list_IDs=[]
for i in range(0,10000):
    train_list_IDs.append(i)
val_list_IDs=[]
for i in range(10000,13144):
    val_list_IDs.append(i)

params = {'height':160,
          'width':160,
          'batch_size': 2,
          'n_channels': 3,
          'shuffle': True,
          'output_classes':output_classes,
          'imagepaths':imagepaths }

training_generator = DataGenerator(train_list_IDs,**params)
validation_generator = DataGenerator(val_list_IDs,**params)
model_exp = '/media/sarthak11/DATA-2/Datasets_2/lfw/Alter/20180402-114759/20180402-114759.pb'

#sess=tf.Session()
#saver = tf.train.import_meta_graph('20180402-114759/model-20180402-114759.meta')
#saver.restore(sess, '20180402-114759/model-20180402-114759.ckpt-275.data-00000-of-00001')
# print("loaded graph and meta")


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnetV1') or node.name.startswith('embeddings') or
                node.name.startswith('phase_train') or node.name.startswith('Bottleneck') or node.name.startswith(
                    'Logits')):
            whitelist_names.append(node.name)
    #print(whitelist_names)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def



def get_labels():
    df = pd.read_csv("Classes.csv", delimiter=",", index_col=False)
    Y = []
    for output_class in output_classes:
        t = np.empty(self.batch_size, output_class, dtype=int)
        Y.append(t)
    last = 0
    #for row in df.itertuples(index=False, name='Pandas'):
    #   row_read = np.asarray(row)
    for j in range(0, len(output_classes)):
        Y[j][i] = df.iloc[:, last:last + output_classes[j]]
        last += output_classes[j]
    return Y

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed=args.seed)

            dataset = facenet.get_dataset(args.data_dir)
            for cls in dataset:
                assert 'len(cls.image_paths)>0', 'There must be at least one image for each class in the dataset'

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            facenet.load_model(args.model)

            #with gfile.FastGFile(model_exp, 'rb') as f:
            #   graph_def = tf.GraphDef()
            #   graph_def.ParseFromString(f.read())
            #tf.import_graph_def(graph_def, name='')
            #print("loaded model")

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            #print ("image placeholder: {}".format(images_placeholder))
            #print (images_placeholder.get_shape()[1])
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            labels=get_labels()
            labels_placeholder = tf.placeholder(tf.string, shape=73)

            embedding_size = embeddings.get_shape()[1]
            #print(embedding_size)


            X = []
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False , labels_placeholder:labels }
                #print(feed_dict)
                dense = tf.nn.relu(embeddings)
                # emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                emb_array[start_index:end_index, :] = sess.run(dense, feed_dict=feed_dict)



                #res=sess.run(embeddings, feed_dict=feed_dict)
                #print(res)
                #X[i]=emb_array[start_index:end_index, :]
                #print(X[3])
                print(emb_array[start_index:end_index,:])
                print(emb_array[start_index].shape)

            input_graph_def = sess.graph.as_graph_def()
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings')
            #print(output_graph_def)

# model = tf.keras.models.Sequential()
# input_lyr=Input(shape=(160,160,3))

# net, endpoints = inception_resnet_v1(inputs=(input_lyr))
# prelogit = endpoints['Mixed_8b']
#print(model)
#print("yes")

# def func(prelogit):
#     x = K.flatten(prelogit)
#     return x
# dense = Dense(10)(prelogit)
# dense = tf.keras.layers.Dense(10)(prelogit)
# print(type(dense))
# model1 = tf.keras.models.Model(inputs=input_lyr,outputs=dense)
#summ=tf.keras.models.Model.summary()
# tf.keras.models.Model.summary()
# model1.summary()
            #print(summ)
output_list = list()
losses = list()
#input_lyr = Input(shape=(32,))
#sequential_model_out = model(input_lyr)



'''''

# 12 classes for softmax
#race_out=Dense(3,activation='softmax', name='race_out')(sequential_model_out)
#age1_out=Dense(3,activation='softmax', name ='age1_out')(sequential_model_out)
age2_out=Dense(2,activation='softmax', name ='age1_out')(sequential_model_out)
colour_hair_out=Dense(4,activation='softmax', name ='colour_hair_out')(sequential_model_out)
type_hair_out=Dense(2,activation='softmax', name ='type_hair_out')(sequential_model_out)
eyewear_out=Dense(3,activation='softmax', name ='eyewear_out')(sequential_model_out)
face_exp_out=Dense(2,activation='softmax', name ='face_exp_out')(sequential_model_out)
lighting_out=Dense(2,activation='softmax', name ='lighting_out')(sequential_model_out)
forehead_out=Dense(3,activation='softmax', name ='forehead_out')(sequential_model_out)
mouth_out=Dense(3,activation='softmax', name ='mouth_out')(sequential_model_out)
beard_out=Dense(2,activation='softmax', name ='beard_hair_out')(sequential_model_out)
face_out=Dense(3,activation='softmax', name ='face_out')(sequential_model_out)



# 41 classes for sigmoid
male_out=Dense(1,activation='sigmoid', name ='male_out')(sequential_model_out)
lips_out=Dense(1,activation='sigmoid', name ='lips_out')(sequential_model_out)
round_jaw_out=Dense(1,activation='sigmoid', name ='round_jaw_out')(sequential_model_out)
double_chin=Dense(1,activation='sigmoid', name ='double_chin_out')(sequential_model_out)
wearing_hat_out=Dense(1,activation='sigmoid', name ='wearing_hat_out')(sequential_model_out)
bag_under_eyes_out=Dense(1,activation='sigmoid', name ='bag_under_eyes_out')(sequential_model_out)
fiveoclock_shadow_out=Dense(1,activation='sigmoid', name ='fiveoclock_shadow_out')(sequential_model_out)
strong_nose_mouth_lines_out=Dense(1,activation='sigmoid', name ='stong_nose_mouth_lines_out')(sequential_model_out)
wearing_lipstick_out=Dense(1,activation='sigmoid', name ='wearing_lipstick_out')(sequential_model_out)
flushed_face_out=Dense(1,activation='sigmoid', name ='flushed_face_out')(sequential_model_out)
high_cheekbones_out=Dense(1,activation='sigmoid', name ='high_cheekbones_out')(sequential_model_out)
wearing_earrings_out=Dense(1,activation='sigmoid', name ='wearing_earrings_out')(sequential_model_out)
indian_out=Dense(1,activation='sigmoid', name ='indian_out')(sequential_model_out)
bald_out=Dense(1,activation='sigmoid', name ='bald_out')(sequential_model_out)
wavy_hair_out=Dense(1,activation='sigmoid', name ='wavy_hair_out')(sequential_model_out)
hairline_out=Dense(1,activation='sigmoid', name ='hairline_out')(sequential_model_out)
bangs_out=Dense(1,activation='sigmoid', name ='bangs_out')(sequential_model_out)
sideburns_out=Dense(1,activation='sigmoid', name ='sideburns_out')(sequential_model_out)
blurry_out=Dense(1,activation='sigmoid', name ='blurry_out')(sequential_model_out)
flash_out=Dense(1,activation='sigmoid', name ='flash_out')(sequential_model_out)
outdoor_out=Dense(1,activation='sigmoid', name ='outdoor_out')(sequential_model_out)
bushy_eyebrows_out=Dense(1,activation='sigmoid', name ='bushy_eyebrows_out')(sequential_model_out)
arched_eyebrows_out=Dense(1,activation='sigmoid', name ='arched_eyebrows_out')(sequential_model_out)
narrow_eyes_out=Dense(1,activation='sigmoid', name ='narrow_eyes_out')(sequential_model_out)
eyes_open_out=Dense(1,activation='sigmoid', name ='eyes_open_out')(sequential_model_out)
brown_eyes_out=Dense(1,activation='sigmoid', name ='brown_eyes_out')(sequential_model_out)
big_nose_out=Dense(1,activation='sigmoid', name ='big_nose_out')(sequential_model_out)
pointy_nose_out=Dense(1,activation='sigmoid', name ='pointy_nose_out')(sequential_model_out)
teeth_not_visible_out=Dense(1,activation='sigmoid', name ='teeth_not_visible_out')(sequential_model_out)
mustache_out=Dense(1,activation='sigmoid', name ='mustache_out')(sequential_model_out)
colour_photo_out=Dense(1,activation='sigmoid', name ='colour_photo_out')(sequential_model_out)
posed_photo_out=Dense(1,activation='sigmoid', name ='posed_photo_out')(sequential_model_out)
attractive_man_out=Dense(1,activation='sigmoid', name ='attractive_man_out')(sequential_model_out)
attractive_woman_out=Dense(1,activation='sigmoid', name ='attractive_woman_out')(sequential_model_out)
chubby_out=Dense(1,activation='sigmoid', name ='chubby_out')(sequential_model_out)
heavy_makeup_out=Dense(1,activation='sigmoid', name ='heavy_makeup_out')(sequential_model_out)
rosy_cheeks_out=Dense(1,activation='sigmoid', name ='rosy_cheeks_out')(sequential_model_out)
shiny_skin_out=Dense(1,activation='sigmoid', name ='shiny_skin_out')(sequential_model_out)
pale_skin_out=Dense(1,activation='sigmoid', name ='pale_skin_out')(sequential_model_out)
wearing_necktie_out=Dense(1,activation='sigmoid', name ='wearing_necktie_out')(sequential_model_out)
wearing_necklace_out=Dense(1,activation='sigmoid', name ='wearing_necklace_out')(sequential_model_out)


output_list=[race_out,age1_out,age2_out,colour_hair_out,type_hair_out,eyewear_out,face_exp_out,
              lighting_out,forehead_out,mouth_out,beard_out,face_out,
              male_out,lips_out,round_jaw_out,double_chin,wearing_hat_out,bag_under_eyes_out,
              fiveoclock_shadow_out,strong_nose_mouth_lines_out,wearing_lipstick_out,flushed_face_out,
              high_cheekbones_out,wearing_earrings_out,indian_out,bald_out,wavy_hair_out,
              hairline_out,bangs_out,sideburns_out,blurry_out,flash_out,outdoor_out,bushy_eyebrows_out,
              arched_eyebrows_out,narrow_eyes_out,eyes_open_out,brown_eyes_out,big_nose_out,pointy_nose_out,
              teeth_not_visible_out,mustache_out,colour_photo_out,posed_photo_out,attractive_man_out,
              attractive_woman_out,chubby_out,heavy_makeup_out,rosy_cheeks_out,shiny_skin_out,pale_skin_out,
              wearing_necktie_out,wearing_necklace_out]

losses=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy','categorical_crossentropy',
        'categorical_crossentropy','categorical_crossentropy','categorical_crossentropy','categorical_crossentropy',
        'categorical_crossentropy','categorical_crossentropy','categorical_crossentropy','categorical_crossentropy',
        'binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy',
        'binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy',
        'binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy',
        'binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy',
        'binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy',
        'binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy',
        'binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy',
        'binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy',
        'binary_crossentropy']

'''''

#Model.fit_generator(generator=training_generator,validation_data=validation_generator, use_multiprocessing=True,workers=6)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    # parser.add_argument('classifier_filename',
    # help='Classifier model file name as a pickle (.pkl) file. ' +
    # 'For training this is the output and for classification this is an input.')
    #parser.add_argument('--use_split_dataset',
    #                   help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
    #                         'Otherwise a separate test set can be specified using the test_data_dir option.',
    #                    action='store_true')
    #parser.add_argument('--test_data_dir', type=str,
     #                   help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=10)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))




























''''

def run(model_name, project_dir, initial_learning_rate, batch_size, num_epoch):
    # ================= TRAINING INFORMATION ==================
    num_batches_per_epoch = int(num_samples / batch_size)
    num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed

    # ================ DATASET INFORMATION ======================
    #data_dir = project_dir + 'data/'
    #data_file = data_dir + 'train_aug.tfrecords'

    # pre-trained checkpoint
    model_dir = project_dir + '20180402-114759'
    pretrained_checkpoint = model_dir + 'model-20180402-114759.ckpt-275.data-00000-of-00001'

    # Create the log directory here. Must be done here otherwise import will activate this unneededly.

    # State where your log file is at. If it doesn't exist, create it.
    log_dir = project_dir + 'logs/' + model_name

    checkpoint_prefix = log_dir + '/model_iters'
    final_checkpoint_file = model_dir + model_name + '_final'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    graph = tf.Graph()
    with graph.as_default():
        with tf.gfile.FastGFile('20180402-114759.pb', 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            tf.import_graph_def(graph_def, name='')


    with tf.Session() as sess:
        # step = sess.run(global_step)

        saver = tf.train.Saver(max_to_keep=20)
        ckpt = tf.train.get_checkpoint_state(log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and continue training!")

        else:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            input_saver = tf.train.Saver(variables_to_restore)

            sess.run(init_op)
            input_saver.restore(sess, pretrained_checkpoint)
            print("restore pre-trained parameters!")

'''


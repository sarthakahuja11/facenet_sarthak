from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
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
#import tensorflow.nn

embed_tf1=globals()
output_list=globals()

from sklearn.metrics import accuracy_score
from data import DataGenerator
from arch.inception_resnet_v1 import inception_resnet_v1
import facenet
import argparse

df = pd.read_csv("Classes.csv", delimiter=",", index_col=False)

with open('Classes.csv','rb') as f:
    reader=csv.reader(f, delimiter = ',')
    rows=[r for r in reader]

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




''''def get_labels(args):
    Y = []
    for output_class in output_classes:
        t = np.empty([int(args.batch_size),len(output_classes)],dtype=int)
        #print(t)
        Y.append(t)
        #print(Y)
    last = 0
    i=0
    #for row in df.itertuples(index=False, name='Pandas'):
    #   row_read = np.asarray(row)
    for j in range(0, len(output_classes)):
        print("**")
        Y[j][i] = df.iloc[:, last:last + output_classes[j]]
        i+=1
        #print(Y[j][i])
        last += output_classes[j]
        #print(last)
        print("**")
    return Y

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

'''''
def get_labels(args):
#    Y = []
    print("init")
    for i in range (1,args.batch_size):
        #print(rows[i])
        print(i)
    return rows[i]


#return Y


#loss_genders = losses(logits_gender, genders)
#loss_races = losses(logits_race, races)

losses_list =['race_out_loss','age1_out_loss','age2_out_loss','colour_hair_out_loss','type_hair_out_loss','eyewear_out_loss','face_exp_out_loss',
             'lighting_out_loss','forehead_out_loss','mouth_out_loss','beard_out_loss','face_out_loss',
              'male_out_loss','lips_out_loss','round_jaw_out_loss','double_chin','wearing_hat_out_loss','bag_under_eyes_out_loss',
              'fiveoclock_shadow_out_loss','strong_nose_mouth_lines_out_loss','wearing_lipstick_out_loss','flushed_face_out_loss',
              'high_cheekbones_out_loss','wearing_earrings_out_loss','indian_out_loss','bald_out_loss','wavy_hair_out_loss',
              'hairline_out_loss','bangs_out_loss','sideburns_out_loss','blurry_out_loss','flash_out_loss','outdoor_out_loss','bushy_eyebrows_out_loss',
              'arched_eyebrows_out_loss','narrow_eyes_out_loss','eyes_open_out_loss','brown_eyes_out_loss','big_nose_out_loss','pointy_nose_out_loss',
              'teeth_not_visible_out_loss','mustache_out_loss','colour_photo_out_loss','posed_photo_out_loss','attractive_man_out_loss',
              'attractive_woman_out_loss','chubby_out_loss','heavy_makeup_out_loss','rosy_cheeks_out_loss','shiny_skin_out_loss','pale_skin_out_loss',
              'wearing_necktie_out_loss','wearing_necklace_out_loss']


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

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            #print ("image placeholder: {}".format(images_placeholder))
            #print (images_placeholder.get_shape()[1])
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            labels=get_labels(args)
            #print(labels)
            labels_placeholder = tf.placeholder(tf.int32, shape=[73])


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
                #dense = tf.nn.relu(embeddings)
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                #emb_array[start_index:end_index, :] = sess.run(dense, feed_dict=feed_dict)
                #res=sess.run(embeddings, feed_dict=feed_dict)
                #print(res)
                #X[i]=emb_array[start_index:end_index, :]
                #print(X[3])
                print(emb_array[start_index:end_index,:])
                print(emb_array[start_index].shape)

                input_graph_def = sess.graph.as_graph_def()
                output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings')

                embed=emb_array[start_index:end_index, :]
                embed_np = np.asarray(embed, np.float32)
                embed_tf = tf.convert_to_tensor(embed_np, np.float32)
                print(embed_tf)
                embed_tf1=embed_tf
            #print(output_graph_def)

#model = tf.keras.models.Sequential()
#input_lyr=Input(shape=(160,160,3))

#net, endpoints = inception_resnet_v1(inputs=(input_lyr))
#prelogit = endpoints['Mixed_8b']
#print(model)
#print("yes")

#def func(prelogit):
#x = K.flatten(prelogit)
#return x
#dense = Dense(10)(prelogit)
#dense = tf.keras.layers.Dense(10)(prelogit)
#print(type(dense))
#model1 = tf.keras.models.Model(inputs=input_lyr,outputs=dense)
#summ=tf.keras.models.Model.summary()
#tf.keras.models.Model.summary()
#model1.summary()
#print(summ)
#input_lyr = Input(shape=(32,))
#sequential_model_out = model(input_lyr)


                output_list = list()
                losses = list()
                # 12 classes for softmax
                race_out = tf.layers.dense(inputs=embed_tf1, units=3, activation=tf.nn.softmax)
                age1_out = tf.layers.dense(inputs=embed_tf1, units=3, activation=tf.nn.softmax)
                age2_out = tf.layers.dense(inputs=embed_tf1, units=2, activation=tf.nn.softmax)
                colour_hair_out = tf.layers.dense(inputs=embed_tf1, units=4, activation=tf.nn.softmax)
                type_hair_out = tf.layers.dense(inputs=embed_tf1, units=2, activation=tf.nn.softmax)
                eyewear_out = tf.layers.dense(inputs=embed_tf1, units=3, activation=tf.nn.softmax)
                face_exp_out = tf.layers.dense(inputs=embed_tf1, units=2, activation=tf.nn.softmax)
                lighting_out = tf.layers.dense(inputs=embed_tf1, units=2, activation=tf.nn.softmax)
                forehead_out = tf.layers.dense(inputs=embed_tf1, units=3, activation=tf.nn.softmax)
                mouth_out = tf.layers.dense(inputs=embed_tf1, units=3, activation=tf.nn.softmax)
                beard_out = tf.layers.dense(inputs=embed_tf1, units=2, activation=tf.nn.softmax)
                face_out = tf.layers.dense(inputs=embed_tf1, units=3, activation=tf.nn.softmax)
                print(face_out)


                # 41 classes for sigmoid
                male_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                lips_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                round_jaw_out =  tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                double_chin = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                wearing_hat_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                bag_under_eyes_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                fiveoclock_shadow_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                strong_nose_mouth_lines_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                wearing_lipstick_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                flushed_face_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                high_cheekbones_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                wearing_earrings_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                indian_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                bald_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                wavy_hair_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                hairline_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                bangs_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                sideburns_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                blurry_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                flash_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                outdoor_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                bushy_eyebrows_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                arched_eyebrows_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                narrow_eyes_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                eyes_open_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                brown_eyes_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                big_nose_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                pointy_nose_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                teeth_not_visible_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                mustache_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                colour_photo_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                posed_photo_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                attractive_man_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                attractive_woman_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                chubby_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                heavy_makeup_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                rosy_cheeks_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                shiny_skin_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                pale_skin_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                wearing_necktie_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                wearing_necklace_out = tf.layers.dense(inputs=embed_tf1, units=1, activation=tf.nn.sigmoid)
                #print(wearing_earrings_out)


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

                mean_loss1=[]
                def loss1():
                    last=0
                    l=0
                    for k in range(0,12):
                        logits=output_list[k]
                        #print(logits)
                        #print("**")
                        #print(i)
                        # start_index = i*args.batch_size
                        loss1 = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=df.iloc[start_index:end_index, last:last + output_classes[k]])
                        #print(loss1)
                        mean_loss1 = tf.reduce_mean(loss1)
                        l+=1
                        print(mean_loss1)

                    print(l)
                    return mean_loss1

                #loss = tf.add_n([loss_genders, loss_races / 5])
                #mean_loss1=loss1()
                #print(mean_loss1)


                def loss2():
                    last=32
                    for i in range(12, 52):
                        logits = output_list[i]
                        loss2 = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=  df.iloc[start_index:end_index:, last:last+1])
                        last+=1
                        mean_loss2 = tf.reduce_mean(loss2)

                    #print(mean_loss2)
                    return mean_loss2

                for i in range(0,12):
                        #print(i)
                        #print("*")
                        losses2=loss1()

                print(losses2)

                for j in range(12,53):
                        #print(j)
                        losses2=loss2()



losses=['tf.losses.softmax_cross_entropy','tf.losses.softmax_cross_entropy','tf.losses.softmax_cross_entropy','tf.losses.softmax_cross_entropy',
        'tf.losses.softmax_cross_entropy','tf.losses.softmax_cross_entropy','tf.losses.softmax_cross_entropy','tf.losses.softmax_cross_entropy',
        'tf.losses.softmax_cross_entropy','tf.losses.softmax_cross_entropy','tf.losses.softmax_cross_entropy','tf.losses.softmax_cross_entropy',
        'tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy',
        'tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy',
        'tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy',
        'tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy',
        'tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy',
        'tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy',
        'tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy',
        'tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy','tf.losses.sigmoid_cross_entropy',
        'tf.losses.sigmoid_cross_entropy']





#race_out_loss=loss1(race_out, df)
#age1_out_loss=loss1(age1_out,
#age2_out_loss=loss1
#colour_hair_out_loss=loss1
#type_hair_out_loss=loss1
#eyewear_out_loss
#face_exp_out_loss
#lighting_out_loss
#forehead_out_loss
#mouth_out_loss
#beard_out_loss
#face_out_loss


#for i in range(0,52):
#cost = tf.reduce_mean((losses[i](logits=output_list[i], labels=labels[i])
#for i in range(12,52)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#init = tf.global_variables_initializer()

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


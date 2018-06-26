import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential

from arch.inception_resnet_v1 import inception_resnet_v1
from data import DataGenerator

#from data import DataGenerator

#partition={'train':[],'validation':[]}
#for i in range(10000):
 # partition['train'].append(i)
#for i in range(10000,13143):
#  partition['validation'].append(i)

#seed = 7
#np.random.seed(seed)

with open("imagepaths.txt") as f:
    imagepaths=f.readlines()
imagepaths=[x.strip() for x in imagepaths]
output_classes=[3,3,2,4,2,3,2,2,3,3,2,3]
for i in range(41):
    output_classes.append(1)

train_list_IDs=[]
for i in range(0,10000):
    train_list_IDs.append(i)
for i in range(10000,13144):
    val_list_IDs.append(i)

# Parameters
params = {'dim': (250,250),
          'batch_size': 32,
          'n_classes': 53
          'n_channels': 3,
          'shuffle': True }

img,label = generator_facenet("Classes.csv",2,53)#,partition)
# Datasets
df = pd.read_csv("Classes.csv", delimiter=",")
# split into input (X) and output (Y) variables
array = df.values
# separate array into input and output components
X = array[:,0:72]
Y = array[:,0]

#partition = # IDs
#labels = # Labels

# Generators
#training_generator = DataGenerator(partition['train'], labels, **params)
#validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
model= inception_resnet_v1()
#[...] # Architecture
#FACENET MODEL
#Extract embeddings
#keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',input_tensor=None, input_shape=None, pooling=None, classes=1000)

model.load_weights('/home/udayakumar97/sarthak_project/facenet_sarthak/20180402-114759/20180402-114759.pb')

for layer in model.layers[:15]:
    layer.trainable = False

model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
#model.add(Dense(num_class, activation='softmax'))



output_list=list()
losses=list()
input_lyr=Input(shape=(32,))
sequential_model_out = model(input_lyr)

#actual_out=Activation('softmax', name='actual_out')(sequential_model_out)
#acept_rej_layer=Dense(num_classes)(sequential_model_out)


# 12 classes for softmax
race_out=Dense(3,activation='softmax', name='race_out')(sequential_model_out)
age1_out=Dense(3,activation='softmax', name ='age1_out')(sequential_model_out)
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


Final_model=Model(input=input_lyr,outputs=output_list)#[actual_out, acept_rej_out])
#ARJ_model=Model(input=input_lyr,outputs=actual_out

model.compile(loss=losses,optimizer='adam', metrics=['accuracy'])


model.summary()
# Train model on dataset
model.fit_generator(generator=generator_facenet,use_multiprocessing=True,workers=6)


#OR USE predict

# evaluate the model
scores = model.evaluate(img, label)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



#to calculate predictions

#predictions = model.predict(X)
# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)



#alter


#model.fit(X, epochs=10, batch_size=10)

#preds = model.predict(X)
#preds[preds>=0.5] = 1
#preds[preds<0.5] = 0


'''
model.add(Dense(num_categories, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


model.add(Dense(y_train.shape[1], activation='sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd)


# Last Inception Module 
x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(9 + i))


# Fully Connected Softmax Layer
x_fc = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
x_fc = Flatten(name='flatten')(x_fc)
x_fc = Dense(1000, activation='softmax', name='predictions')(x_fc)

# Create model
model = Model(img_input, x_fc)

#Load ImageNet pre-trained data
#model.load_weights('cache/inception_v3_weights_th_dim_ordering_th_kernels.h5')

# Truncate and replace softmax layer for transfer learning
# Cannot use model.layers.pop() since model is not of Sequential() type
# The method below works since pre-trained weights are stored in layers but not in the model
x_newfc = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
x_newfc = Flatten(name='flatten')(x_newfc)

# Create another model with our customized softmax
model = Model(img_input, x_newfc)

# Learning rate is changed to 0.001
#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model = inception_v3_model(img_rows, img_cols, channel, num_class)

#losses = {
#  "category_output": "categorical_crossentropy",
# "color_output": "categorical_crossentropy",
#}
#lossWeights = {"category_output": 1.0, "color_output": 1.0}



'''            

import numpy as np
import pandas as pd
from keras.engine.training import _make_batches
#from skimage.io import imread
import cv2
import glob


#class (keras.utils.Sequence):

def generator_facenet(csv_file, batch_size=2, num_classes=53, is_training=True):#,partition):


    folders = glob.glob('/media/sarthak11/DATA-2/Datasets_2/lfw_funneled/*')
    imagenames_list = []
    for folder in folders:
        for f in glob.glob(folder+'/*.jpg'):
            imagenames_list.append(f)

        #print(imagenames_list)
    read_images = []                

    for image in imagenames_list:
        read_images.append(cv2.imread(image))#, cv2.IMREAD_GRAYSCALE))

        #print(read_images)
        #plt.imshow(read_images[1])

    #with open('folders.txt', 'r') as f1 :
    #for line in f1:   
    #name=line
    #for line in f:
    #image_files_path = csv.load(open(csv_file))
    #len_image_files_path = len(image_files_path)
    #print len_image_files_path
    image_files_path = np.array(imagenames_list)
    len_image_files_path = len(image_files_path)
     #df  = pd.read_csv("Classes.csv",index=false)  #TO DO
    df = pd.read_csv(csv_file, delimiter=",", index_col=False)


    while True:
        batches = _make_batches(size=len_image_files_path,batch_size=2)
        for start, end in batches:
            arr = []
            Y = []
            X = []
            cur_batch = image_files_path[start:end]
            #print "current Batch size", len(cur_batch)
            for image_path in cur_batch:
                #print image_path
                img = imread(fname=image_path)
                
           
            for row in df.itertuples(index=False, name='Pandas'):
                row_read=np.asarray(row)

                race_out=df.iloc[:,0:3]  
                age1_out=df.iloc[:,3:6]
                age2_out=df.iloc[:,6:8]
                colour_hair_out=df.iloc[:,8:12]
                type_hair_out=df.iloc[:,12:14]
                eyewear_out=df.iloc[:,14:17]
                face_exp_out=df.iloc[:,17:19]
                lighting_out=df.iloc[:,19:21]
                forehead_out=df.iloc[:,21:24]
                mouth_out=df.iloc[:,24:27]
                beard_out=df.iloc[:,27:29]
                face_out=df.iloc[:,29:32]
                             
                male_out=df.iloc[:,32]
                lips_out=df.iloc[:,33]
                round_jaw_out=df.iloc[:,34]
                double_chin=df.iloc[:,35]
                wearing_hat_out=df.iloc[:,36]
                bag_under_eyes_out=df.iloc[:,37]
                fiveoclock_shadow_out=df.iloc[:,38]
                strong_nose_mouth_lines_out=df.iloc[:,39]
                wearing_lipstick_out=df.iloc[:,40]
                flushed_face_out=df.iloc[:,41]
                high_cheekbones_out=df.iloc[:,42]
                wearing_earrings_out=df.iloc[:,43]
                indian_out=df.iloc[:,44]
                bald_out=df.iloc[:,45]
                wavy_hair_out=df.iloc[:,46]
                hairline_out=df.iloc[:,47]
                bangs_out=df.iloc[:,48]
                sideburns_out=df.iloc[:,49]
                blurry_out=df.iloc[:,50]
                flash_out=df.iloc[:,51]
                outdoor_out=df.iloc[:,52]
                bushy_eyebrows_out=df.iloc[:,53]
                arched_eyebrows_out=df.iloc[:,54]
                narrow_eyes_out=df.iloc[:,55]
                eyes_open_out=df.iloc[:,56]
                brown_eyes_out=df.iloc[:,57]
                big_nose_out=df.iloc[:,58]
                pointy_nose_out=df.iloc[:,59]
                teeth_not_visible_out=df.iloc[:,60]
                mustache_out=df.iloc[:,61]
                colour_photo_out=df.iloc[:,62]
                posed_photo_out=df.iloc[:,63]
                attractive_man_out=df.iloc[:,64]
                attractive_woman_out=df.iloc[:,65]
                chubby_out=df.iloc[:,66]
                heavy_makeup_out=df.iloc[:,67]
                rosy_cheeks_out=df.iloc[:,68]
                shiny_skin_out=df.iloc[:,69]
                pale_skin_out=df.iloc[:,70]
                wearing_necktie__out=df.iloc[:,71]
                wearing_necklace_out=df.iloc[:,72]

                    
                arr.append(img)
                arr = np.array(arr)
                arr.astype('float32')
                arr /= 255

                X=[batch_size*arr]
                Y=[row_read]


    return X,Y
                
                #folder_name = image_path.split('/')[2]
                #cls = folder_name.split('_')[-1]
                #arr -= 0.5
                #arr *= 2.
            #yield (arr, Y)
            

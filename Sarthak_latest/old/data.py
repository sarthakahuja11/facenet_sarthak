import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(250,250), n_channels=3,
                 n_classes=53, shuffle=True,imagepaths):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.imagepaths=imagepaths

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y=[]
        t=np.empty((self.batch_size),3,dtype=int)
        y.append(t)
        t=np.empty((self.batch_size),3,dtype=int)
        y.append(t)
        t=np.empty((self.batch_size),2,dtype=int)
        y.append(t)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #Read entire file with ith element as each line
            folder_name=imagepaths[i][:-10] #get image path
            image_id=int(imagepaths[i][-8:-4])
            img = imread(fname=image_path)

            X[i,] = np.array(img)

            # Store class
            y[0][i]=df.iloc[:,0:3]
            y[1][i] = df.iloc[:,3:6]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
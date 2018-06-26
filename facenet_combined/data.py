import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, batch_size=32, dim=(250,250), n_channels=3,
				 output_classes, shuffle=True, imagepaths):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.output_classes = output_classes
		self.shuffle = shuffle
		self.imagepaths=imagepaths
		self.on_epoch_end()

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
		Y = []
		for output_class in output_classes:
			t=np.empty((self.batch_size),output_class,dtype=int)
			Y.append(t)
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			#Read entire file with ith element as each line
			folder_name=imagepaths[i][:-9] #get image path
			image_id=int(imagepaths[i][-8:-4])
			image_path=('../lfw_funneled/'+folder_name+'/'+imagepaths[i])
			img = imread(fname=image_path)

			X[i,] = np.array(img)
			last=0
			for j in range(0,len(output_classes)):
				Y[j][i]=df.iloc[:,last:last+output_classes[j]]
				last+=output_classes[j]

		return X, Y
import os
import math

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from dataset import DataSet


class Bert_Basic_Model(object):
	
	def __init__(self, embedding_dim=None, model_path=None):
		"""
		embedding_dim: BERT sentence embedding dimension (768 or 1024)
		"""

		if model_path:
			self.model_ = keras.models.load_model(model_path)
			return
		
		# construct model
		A_encoded = keras.Input(shape=[embedding_dim], dtype='float32', name='A_encoded')
		B_encoded = keras.Input(shape=[embedding_dim], dtype='float32', name='B_encoded')

		x = layers.concatenate([A_encoded, B_encoded])
		# x = layers.Dense(200, activation='selu')(x)
		# x = layers.Dropout(0.5)(x)
		# x = layers.Dense(50)(x)
		# x = layers.BatchNormalization()(x)
		# x = layers.Activation(keras.activations.relu)(x)
		#x = layers.Dense(10, activation='selu')(x)

		#### bert_large: good train avd validation accuracy (0.9457, 0.9640)
		# x = layers.Dense(200)(x)
		# x = layers.BatchNormalization()(x)
		# x = layers.Activation(keras.activations.relu)(x)
		####

		# x = layers.Dense(200, activation='selu')(x)
		# x = layers.AlphaDropout(0.5)(x)
		# x = layers.Dense(20, activation='selu')(x)

		# x = layers.Dense(200, activation='relu')(x)
		# x = layers.Dropout(0.3)(x)
		# x = layers.Dense(200, activation='relu')(x)

		# x = layers.Dense(200)(x)
		# x = layers.BatchNormalization()(x)
		# x = layers.Activation(keras.activations.relu)(x)
		
		x = layers.Dense(50)(x)
		x = layers.BatchNormalization()(x)
		x = layers.Activation(keras.activations.relu)(x)

		class_output = layers.Dense(3, activation='softmax', name='class_output')(x)

		self.model_ = keras.Model(
			inputs = [A_encoded, B_encoded],
			outputs = [class_output]
		)
		
		self.model_.compile(
			optimizer = keras.optimizers.Adam(lr=0.001, amsgrad=True),
			loss = 'sparse_categorical_crossentropy',
			metrics = ['sparse_categorical_accuracy']
		)


	def fit(self, 
		data, 
		batch_size, 
		max_epochs, 
		model_checkpoint_dir,
		validation_split=0.2,
		weighted=True,
		dataset_buffer_size=None,
		num_parallel_reads=None,
		shuffle=False, 
		shuffle_buffer_size=None,
		seed=None):
		"""
		data: DataSet
		"""

		dataset = tf.data.TFRecordDataset(
			filenames = data.tfrecords['train'],
			buffer_size = dataset_buffer_size,
			num_parallel_reads = num_parallel_reads
		)
		dataset = dataset.map(data.get_example_parser(mode='train', weighted=weighted))
		
		if shuffle:
			dataset = dataset.shuffle(
				buffer_size = shuffle_buffer_size if shuffle_buffer_size is not None else data.size['train'],
    			seed = seed
			)

		dvalid_size = math.floor(data.size['train'] * validation_split)
		dtrain_size = data.size['train'] - dvalid_size

		steps_per_epoch = math.ceil(dtrain_size / batch_size)
		validation_steps = math.ceil(dvalid_size / batch_size)

		dtrain = dataset.take(dtrain_size)
		dvalid = dataset.skip(dtrain_size)

		dtrain = dtrain.batch(batch_size).repeat(max_epochs)
		dvalid = dvalid.batch(batch_size).repeat(max_epochs)

		callbacks = [
			keras.callbacks.EarlyStopping(
				monitor = 'val_loss',
				min_delta = 1e-2,
				patience = 3,
				verbose = 1,
				restore_best_weights = True
			),
			keras.callbacks.ModelCheckpoint(
				filepath = os.path.join(model_checkpoint_dir, 'model_{epoch}.h5'),
				verbose = 0
			)
		]

		self.model_.fit(
			dtrain,
			validation_data = dvalid,
			epochs = max_epochs,
			steps_per_epoch = steps_per_epoch,
			validation_steps = validation_steps,
			callbacks = callbacks,
			verbose = 1
		)


	def predict(self, 
		data,
		batch_size,
		dataset_buffer_size=None,
		num_parallel_reads=None):
		"""
		data: DataSet
		"""
		
		dtest = tf.data.TFRecordDataset(
			filenames = data.tfrecords['test'],
			buffer_size = dataset_buffer_size,
			num_parallel_reads = num_parallel_reads
		)

		sample_id = dtest.map(data.get_example_parser(mode='id_only')).make_one_shot_iterator().get_next() 
		ids = []
		with tf.Session() as sess:
			try:
				while True:
					ids.append(sess.run(sample_id))
			except tf.errors.OutOfRangeError:
				pass
			

		dtest = dtest.map(data.get_example_parser(mode='test'))
		dtest = dtest.batch(batch_size)

		class_probs = self.model_.predict(
			dtest,
			steps = math.ceil(data.size['test'] / batch_size)
		)

		return ids, np.argmax(class_probs, axis=1)

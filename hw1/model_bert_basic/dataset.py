import os
import json

import tensorflow as tf


class DataSet(object):
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'config.json')) as f:
            config = json.load(f)
        
        self.tfrecords = {
            'train': [ os.path.join(data_dir, filename) for filename in config['train']['files']],
            'test': [ os.path.join(data_dir, filename) for filename in config['test']['files']]
        }

        self.size = {
            'train': config['train']['size'],
            'test': config['test']['size']
        }

        self.embedding_dim = config['embedding_dim']
        self.label = config['label']

    def get_example_parser(self, mode, weighted=True):
        """
        mode: 'train', 'test', 'id_only'
        """

        common_features = {
            'A_encoded': tf.FixedLenFeature([self.embedding_dim], dtype=tf.float32),
            'B_encoded': tf.FixedLenFeature([self.embedding_dim], dtype=tf.float32)
        }
        
        def example_parser_train(serialized):
            features = common_features
            features['label'] = tf.FixedLenFeature([], dtype=tf.int64)
            features['class_weight'] = tf.FixedLenFeature([], dtype=tf.float32)

            parsed = tf.parse_single_example(
                serialized=serialized,
                features=features
            )

            if weighted:
                return {'A_encoded': parsed['A_encoded'], 'B_encoded': parsed['B_encoded']}, parsed['label'], parsed['class_weight']
            else:
                return {'A_encoded': parsed['A_encoded'], 'B_encoded': parsed['B_encoded']}, parsed['label']

        def example_parser_test(serialized):
            features = common_features
            features['id'] = tf.FixedLenFeature([], dtype=tf.int64)

            parsed = tf.parse_single_example(
                serialized=serialized,
                features=features
            )

            return {'A_encoded': parsed['A_encoded'], 'B_encoded': parsed['B_encoded']}


        def example_parser_id_only(serialized):
            parsed = tf.parse_single_example(
                serialized=serialized,
                features={
                    'id': tf.FixedLenFeature([], dtype=tf.int64)
                }
            )

            return parsed['id']


        if mode == 'train':
            return example_parser_train
        elif mode == 'test':
            return example_parser_test
        elif mode == 'id_only':
            return example_parser_id_only
        else:
            raise ValueError(f"Unknown mode: {mode}")
import os
import json
import argparse
import math

from tqdm import tqdm
import pandas as pd

import tensorflow as tf
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features
from tensorflow.train import Example
from tensorflow.io import TFRecordWriter

from bert_serving.client import BertClient


def create_examples(data, bert_client, training=True, label2int=None, class_weight=None):
    """
    data: pd.DataFrame
    label2int: dict
    class_weight: list

    yield examples
    """
    idx_start = data.index[0]

    A_encoded = bert_client.encode(data['title1_en'].tolist())
    B_encoded = bert_client.encode(data['title2_en'].tolist())

    for i in range(len(data)):
        feature = {
            'A_encoded': Feature(float_list=FloatList(value=A_encoded[i])),
            'B_encoded': Feature(float_list=FloatList(value=B_encoded[i]))
        }
        if training:
            label = label2int[ data.loc[idx_start+i, 'label'] ]
            feature['label'] = Feature(int64_list=Int64List(value=[ label ]))  
            feature['class_weight'] = Feature(float_list=FloatList(value=[ class_weight[label] ]))
        else:
            feature['id'] = Feature(int64_list=Int64List(value=[ data.loc[idx_start+i, 'id'] ]))

        yield Example(features=Features(feature=feature))


def make_dataset(data_path, output_dir, name, bert_client, training=True, label2int=None, class_weight=None, n_split=1):
    """
    data_path: path to the data (csv)
    label2int: dict
    class_weight: list
    n_split: Save the dataset to `n_split` seperated files

    Write dataset to ${output_dir}/${name}_${seq}.tfrecord (seq = 0 ~ n_split-1)
    
    Return file names of the created datasets (list), size of the dataset
    """
    
    data = pd.read_csv(data_path)

    # replace empty titles with 'none'
    data['title1_en'] = data['title1_en'].apply(lambda x: 'none' if x.strip() == '' else x)
    data['title2_en'] = data['title2_en'].apply(lambda x: 'none' if x.strip() == '' else x)

    n_samples = math.ceil(len(data) / n_split)
    filenames = []

    with tqdm(total=len(data)) as pbar:

        for i in range(n_split):
            filenames.append(f"{name}_{i}.tfrecord")
            with TFRecordWriter(os.path.join(output_dir, filenames[-1])) as writer:
                examples = create_examples(
                    data = data[i*n_samples : (i+1)*n_samples],
                    bert_client = bert_client,
                    training = training,
                    label2int = label2int,
                    class_weight = class_weight
                )
                for example in examples:
                    writer.write(example.SerializeToString())
                    pbar.update()

    return filenames, len(data)


def main(args):

    config = {
        'train': {},
        'test': {},
        'embedding_dim': 1024,
        'label': ['agreed', 'disagreed', 'unrelated']
    }
    label2int = {
        'agreed': 0,
        'disagreed': 1,
        'unrelated': 2
    }
    class_weight = [1/15, 1/5, 1/16]
    
    with BertClient(port=5555, port_out=5556, show_server_config=True, check_length=False) as bert_client:
        
        print('Generating training datasets ...')
        config['train']['files'], config['train']['size'] = make_dataset(
            data_path=args.dtrain,
            output_dir=args.output_dir,
            name='train',
            bert_client=bert_client,
            training=True,
            label2int=label2int,
            class_weight=class_weight,
            n_split=args.n_split
        )

        print('Generating testing datasets ...')
        config['test']['files'], config['test']['size'] = make_dataset(
            data_path=args.dtest,
            output_dir=args.output_dir,
            name='test',
            bert_client=bert_client,
            training=False,
            n_split=args.n_split
        )

    print('Writing config ...')
    with open(os.path.join(args.output_dir, 'config.json'), mode='w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    print('Datasets generation complete.')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, dest='dtrain',
                        help='path to training data')
    parser.add_argument('--test', type=str, required=True, dest='dtest',
                        help='path to testing data')
    parser.add_argument('-o', type=str, required=True, dest='output_dir',
                        help='output directory of preprocessed datasets')               
    parser.add_argument('-s', '--splits', type=int, default=1, dest='n_split',
                        help='split the dataset into multiple files')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
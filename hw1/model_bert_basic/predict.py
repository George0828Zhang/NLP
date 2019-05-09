import os
import argparse

from dataset import DataSet
from model_bert_basic import Bert_Basic_Model

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'


def main(args):
    
    dataset = DataSet(args.data_dir)

    if args.model_epoch:
        model = Bert_Basic_Model(model_path = os.path.join(args.model_save_dir, f"model_{args.model_epoch}.h5"))
    else:
        model = Bert_Basic_Model(embedding_dim = dataset.embedding_dim)
        model.fit(
            data = dataset, 
            batch_size = 64, 
            max_epochs = 30, 
            model_checkpoint_dir = args.model_save_dir,
            validation_split = 0.2, 
            weighted = True,
            num_parallel_reads = 2,
            shuffle = True, 
            shuffle_buffer_size = None
        )

    sample_id, prediction = model.predict(
        data = dataset,
		batch_size = 1024,
		num_parallel_reads = 1
    )

    with open(args.output_path, mode='w', encoding='utf-8') as f:
        f.write('Id,Category\n')
        for _id, class_id in zip(sample_id, prediction):
            f.write(f"{_id},{dataset.label[class_id]}\n")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, dest='data_dir',
                        help='data directory')
    parser.add_argument('-s', type=str, required=True, dest='model_save_dir',
                        help='model save directory')
    parser.add_argument('-o', type=str, required=True, dest='output_path',
                        help='prediction output path')
    parser.add_argument('-n', type=int, dest='model_epoch',
                        help='load model trained at epoch n')              
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
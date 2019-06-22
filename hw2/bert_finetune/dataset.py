import os, sys
import csv

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bert', 'src'))

import tokenization

import tensorflow as tf
from tensorflow.io import TFRecordWriter


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text or text pair of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text or text pair of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                input_ids,
                input_mask,
                segment_ids,
                label_id=None,
                is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class OLID_Processor(DataProcessor):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_train_examples(self):
        return self._create_examples(os.path.join(self.data_dir, 'train.tsv'), testing=False)

    def get_dev_examples(self):
        return self._create_examples(os.path.join(self.data_dir, 'eval.tsv'), testing=False)

    def get_test_examples(self):
        return self._create_examples(os.path.join(self.data_dir, 'test.tsv'), testing=True)

    def get_labels(self):
        raise NotImplementedError()
      
    @classmethod
    def _create_examples(cls, file_path, testing=False):
        examples = []
        
        with open(file_path, encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, line in enumerate(reader):
                if i == 0:
                    continue

                text_a = tokenization.convert_to_unicode(line[1])
                label = None
                if not testing:
                    label = tokenization.convert_to_unicode(line[2])
                
                examples.append(
                    InputExample(guid=line[0], text_a=text_a, text_b=None, label=label)
                )
        
        return examples


class Olid_A_Processor(OLID_Processor):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_labels(self):
        return ['NOT', 'OFF']


class Olid_B_Processor(OLID_Processor):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_labels(self):
        return ['UNT', 'TIN']


class Olid_C_Processor(OLID_Processor):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_labels(self):
        return ['IND', 'GRP', 'OTH']



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while len(tokens_a) + len(tokens_b) > max_length:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _tokenize(text_a, text_b, max_seq_length, tokenizer):
    """
    text_a: string
    text_b: string or None
    """
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b) if text_b is not None else None

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length-2:
            tokens_a = tokens_a[0: max_seq_length-2]

    
    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segment_ids = [0] * (len(tokens_a)+2)

    if tokens_b:
        tokens += tokens_b + ['[SEP]']
        segment_ids += [1] * (len(tokens_b)+1)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    if len(input_ids) < max_seq_length:
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

    return input_ids, input_mask, segment_ids


def _convert_single_example(example, label_map, max_seq_length, tokenizer, is_testing):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids = [0] * max_seq_length,
            input_mask = [0] * max_seq_length,
            segment_ids = [0] * max_seq_length,
            label_id = 0,
            is_real_example = False
        )

    input_ids, input_mask, segment_ids = _tokenize(example.text_a, example.text_b, max_seq_length, tokenizer)
    label_id = label_map[example.label] if not is_testing else 0

    return InputFeatures(
        input_ids = input_ids,
        input_mask = input_mask,
        segment_ids = segment_ids,
        label_id = label_id,
        is_real_example = True
    )



def write_examples_to_tfrecord(examples, label_list, max_seq_length, tokenizer, output_file, is_testing, pbar_desc=None):
    """Write a set of `InputExample`s to a TFRecord file."""

    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    label_map = { label:i for i, label in enumerate(label_list) }

    with TFRecordWriter(output_file) as writer:
        for example in tqdm(examples, desc=pbar_desc):
            feature = _convert_single_example(
                example=example, 
                label_map=label_map, 
                max_seq_length=max_seq_length, 
                tokenizer=tokenizer,
                is_testing=is_testing
            )

            tf_features = {
                'input_ids': create_int_feature(feature.input_ids),
                'input_mask': create_int_feature(feature.input_mask),
                'segment_ids': create_int_feature(feature.segment_ids),
                'label_id': create_int_feature([ feature.label_id ]),
                'is_real_example': create_int_feature([int(feature.is_real_example)])
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=tf_features))
            writer.write(tf_example.SerializeToString())
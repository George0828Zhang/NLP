# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os, sys
import json
import re
import math

import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bert', 'src'))

import modeling, optimization, tokenization
from dataset import write_examples_to_tfrecord, PaddingInputExample
from dataset import Olid_A_Processor, Olid_B_Processor, Olid_C_Processor


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    'data_dir', None,
    'The data directory that stores tfrecord files. (GCS is supported)'
)

flags.DEFINE_string(
    'local_data_dir', None,
    'The directory where the data processor will load data from. Dataset config will also be stored here.'
)

flags.DEFINE_string(
    'bert_config_file', None,
    'The config json file corresponding to the pre-trained BERT model.'
)

flags.DEFINE_string(
    'task_name', None, 
    'The name of the task to train.'
)

flags.DEFINE_string(
    'vocab_file', None,
    'The vocabulary file that the BERT model was trained on.'
)

flags.DEFINE_string(
    'output_dir', None,
    'The output directory where the model checkpoints will be written.'
)

## Other parameters

flags.DEFINE_bool(
    'make_dataset_only', False,
    'Whether to only generate tfrecord files.'
)

flags.DEFINE_bool(
    'reuse_dataset', False,
    'Whether to use existing dataset.'
)

flags.DEFINE_string(
    'init_checkpoint', None,
    'Initial checkpoint (usually from a pre-trained BERT model).'
)

flags.DEFINE_bool(
    'do_lower_case', True,
    'Whether to lower case the input text. Should be True for uncased models and False for cased models.'
)

flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter than this will be padded.'
)

flags.DEFINE_bool(
    'do_train', False, 
    'Whether to run training.'
)

flags.DEFINE_bool(
    'do_eval', False, 
    'Whether to run eval on the dev set.'
)

flags.DEFINE_bool(
    'do_predict', False,
    'Whether to run the model in inference mode on the test set.'
)

flags.DEFINE_integer(
    'train_batch_size', 32, 
    'Batch size for training.'
)

flags.DEFINE_integer(
    'eval_batch_size', 8, 
    'Batch size for eval.'
)

flags.DEFINE_integer(
    'predict_batch_size', 8, 
    'Batch size for predict.'
)

flags.DEFINE_float(
    'learning_rate', 5e-5, 
    'The initial learning rate for Adam.'
)

flags.DEFINE_integer(
    'num_train_epochs', 3,
    'Total number of training epochs to perform.'
)

flags.DEFINE_float(
    'warmup_proportion', 0.1,
    'Proportion of training to perform linear learning rate warmup for. '
    'E.g., 0.1 = 10% of training.'
)

flags.DEFINE_integer(
    'save_checkpoints_steps', 1000,
    'How often to save the model checkpoint.'
)

flags.DEFINE_integer(
    'iterations_per_loop', 1000,
    'How many steps to make in each estimator call.'
)

flags.DEFINE_bool(
    'use_tpu', False, 
    'Whether to use TPU or GPU/CPU.'
)

tf.flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.'
)

tf.flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from metadata.'
)

tf.flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from metadata.'
)

tf.flags.DEFINE_string(
    'master', None, 
    '[Optional] TensorFlow master URL.'
)

flags.DEFINE_integer(
    'num_tpu_cores', 8,
    'Only used if `use_tpu` is True. Total number of TPU cores to use.'
)



def _get_example_parser(seq_length, is_testing):
    
    name_to_features = {
        'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.FixedLenFeature([seq_length], tf.int64),
        'segment_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'label_id': tf.FixedLenFeature([], tf.int64),
        'is_real_example': tf.FixedLenFeature([], tf.int64)
    }

    def _example_parser(serialized):
        
        parsed = tf.parse_single_example(
            serialized=serialized,
            features=name_to_features
        )

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        features = {}
        for name, feature in parsed.items():
            if name == 'label_id':
                continue

            features[name] = tf.cast(feature, tf.int32)

        if not is_testing:
            return features, parsed['label_id']
        else:
            return features

    return _example_parser



def file_based_input_fn_builder(input_file, seq_length, mode, drop_remainder, shuffle_buffer_size=100):
    """
    Creates an `input_fn` closure to be passed to TPUEstimator.
    
    mode: 'train', 'eval', 'test'
    """

    def input_fn(params):
        
        batch_size = params['batch_size']

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.

        d = tf.data.TFRecordDataset(input_file, buffer_size = 2**23)
        if mode == 'train':
            d = d.shuffle(buffer_size=shuffle_buffer_size).repeat()

        d = d.apply(
            tf.data.experimental.map_and_batch(
                _get_example_parser(seq_length, mode == 'test'),
                batch_size = batch_size,
                drop_remainder = drop_remainder
            )
        )
        d = d.prefetch(tf.contrib.data.AUTOTUNE)

        return d

    return input_fn


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """
    Compute the union of the current variables and checkpoint variables.
    """
    
    assignment_map = {}
    initialized_variable_names = set()

    name_to_variable = {}
    for var in tvars:
        name = var.name
        m = re.match(r'^(.*):\d+$', name)
        if m is not None:
            name = m.group(1)
        
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)
    for name, _ in init_vars:
        if name in name_to_variable:
            assignment_map[name] = name_to_variable[name]
            initialized_variable_names.add(name)
            initialized_variable_names.add(name+':0')

        # elif 'BERT/'+name in name_to_variable:
        #     assignment_map[name] = name_to_variable['BERT/'+name]
        #     initialized_variable_names.add('BERT/'+name)
        #     initialized_variable_names.add('BERT/'+name+':0')

    return assignment_map, initialized_variable_names


def mul_mask(x, m):
    return x * tf.expand_dims(m, axis=-1)

def masked_reduce_mean(x, m):
    m = tf.cast(m, tf.float32)
    return tf.reduce_sum(mul_mask(x, m), axis=1) / (tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)


def create_model(bert_config, 
                mode, 
                features, 
                labels, 
                num_labels,
                use_one_hot_embeddings):

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config = bert_config,
        is_training = is_training,
        input_ids = features['input_ids'],
        input_mask = features['input_mask'],
        token_type_ids = features['segment_ids'],
        use_one_hot_embeddings = use_one_hot_embeddings,
        scope='bert'
    )

    with tf.variable_scope('fine_tune'):
    
        x = model.get_pooled_output()
        # x = model.get_sequence_output()
        # x = masked_reduce_mean(x, features['input_mask'])

        if is_training:
            x = tf.nn.dropout(x, rate=0.1)

        logits = tf.keras.layers.Dense(
            num_labels, 
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
            bias_initializer = 'zeros',
            name = 'fc1'
        )(x)

        probs = tf.nn.softmax(logits)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return probs
        
        else:
            with tf.variable_scope('loss'):
                log_probs = tf.nn.log_softmax(logits)
                one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                loss = tf.reduce_mean(per_example_loss)

            return loss, per_example_loss, probs


def model_fn_builder(bert_config, 
                    output_dir,
                    num_labels,
                    init_checkpoint, 
                    learning_rate,
                    num_train_steps, 
                    num_warmup_steps, 
                    use_tpu,
                    use_one_hot_embeddings):

    """
    Returns `model_fn` closure for TPUEstimator.
    """

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

        tf.logging.info('*** Features ***')
        for name in features:
            tf.logging.info(f"  name = {name}, shape = {features[name].shape}")
        
        is_real_example = tf.cast(features['is_real_example'], dtype=tf.float32)

        results = create_model(
            bert_config = bert_config, 
            mode = mode, 
            features = features, 
            labels = labels, 
            num_labels = num_labels,
            use_one_hot_embeddings = use_one_hot_embeddings
        )

        if mode == tf.estimator.ModeKeys.PREDICT:
            probs = results
        else:
            total_loss, per_example_loss, probs = results
        
        scaffold_fn = None
        if mode == tf.estimator.ModeKeys.TRAIN and init_checkpoint:
            
            tvars = tf.trainable_variables()
            initialized_variable_names = set()

            assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info('**** Trainable Variables ****')
            for var in tvars:
                init_string = ''
                if var.name in initialized_variable_names:
                    init_string = ', *INIT_FROM_CKPT*'
                
                tf.logging.info(f"  name = {var.name}, shape = {var.shape}{init_string}")

        
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            def host_call_fn(global_step, loss):
                with tf.contrib.summary.create_file_writer(logdir=output_dir).as_default():
                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar('loss', loss[0], step=global_step[0])

                        return tf.contrib.summary.all_summary_ops()

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                host_call=(host_call_fn, [ tf.reshape(tf.train.get_global_step(), [1]), tf.reshape(total_loss, [1]) ])
            )
        
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probs, is_real_example):
                predictions = tf.argmax(probs, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, 
                    predictions=predictions, 
                    weights=is_real_example
                )
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)

                return {
                    'eval_accuracy': accuracy,
                    'eval_loss': loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, labels, probs, is_real_example])
            
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics
            )

        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={'probabilities': probs}
            )
        
        return output_spec

    return model_fn


def _dataset_path(filename):
    if FLAGS.data_dir.startswith('gs://'):
        return FLAGS.data_dir + '/' + filename
    else:
        return os.path.join(FLAGS.data_dir, filename)


def make_datasets(processor):
    """
    Only make required tfrecord files.
    Only generate required tfrecord files that already exist in `data_dir` if `FLAGS.reuse_dataset` is false.
    """

    def _isfile(path):
        return tf.gfile.Exists(path) and not tf.gfile.IsDirectory(path)


    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file = FLAGS.vocab_file, 
        do_lower_case = FLAGS.do_lower_case
    )

    config_path = os.path.join(FLAGS.local_data_dir, 'config.json')

    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    if FLAGS.do_train:
        
        train_tfrecord_path = _dataset_path('train.tfrecord')
        if not FLAGS.reuse_dataset or not _isfile(train_tfrecord_path):

            tf.logging.info('***** Writing training examples to tfrecord file *****')

            examples = processor.get_train_examples()
            write_examples_to_tfrecord(
                examples = examples, 
                label_list = label_list, 
                max_seq_length = FLAGS.max_seq_length, 
                tokenizer = tokenizer,
                output_file = train_tfrecord_path,
                is_testing = False,
                pbar_desc = 'train.tfrecord' 
            )
            config['num_train_examples'] = len(examples)
            config['train_file'] = 'train.tfrecord'

    if FLAGS.do_eval:

        eval_tfrecord_path = _dataset_path('eval.tfrecord')
        if not FLAGS.reuse_dataset or not _isfile(eval_tfrecord_path):

            tf.logging.info('***** Writing eval examples to tfrecord file *****')

            examples = processor.get_dev_examples()
            config['num_eval_examples'] = len(examples)
            
            if FLAGS.use_tpu:
                # TPU requires a fixed batch size for all batches, therefore the number
                # of examples must be a multiple of the batch size, or else examples
                # will get dropped. So we pad with fake examples which are ignored
                # later on. These do NOT count towards the metric (all tf.metrics
                # support a per-instance weight, and these get a weight of 0.0).
                while len(examples) % FLAGS.eval_batch_size != 0:
                    examples.append(PaddingInputExample())
            
            write_examples_to_tfrecord(
                examples = examples, 
                label_list = label_list, 
                max_seq_length = FLAGS.max_seq_length, 
                tokenizer = tokenizer,
                output_file = eval_tfrecord_path,
                is_testing = False,
                pbar_desc = 'eval.tfrecord' 
            )

            config['num_eval_examples_plus_padding'] = len(examples)
            config['eval_file'] = 'eval.tfrecord'

    if FLAGS.do_predict:

        test_tfrecord_path = _dataset_path('test.tfrecord')
        if not FLAGS.reuse_dataset or not _isfile(test_tfrecord_path):

            tf.logging.info('***** Writing testing examples to tfrecord file *****')

            examples = processor.get_test_examples()
            config['num_test_examples'] = len(examples)

            if FLAGS.use_tpu:
                # TPU requires a fixed batch size for all batches, therefore the number
                # of examples must be a multiple of the batch size, or else examples
                # will get dropped. So we pad with fake examples which are ignored
                # later on.
                while len(examples) % FLAGS.predict_batch_size != 0:
                    examples.append(PaddingInputExample())
            
            write_examples_to_tfrecord(
                examples = examples, 
                label_list = label_list, 
                max_seq_length = FLAGS.max_seq_length, 
                tokenizer = tokenizer,
                output_file = test_tfrecord_path,
                is_testing = True,
                pbar_desc = 'test.tfrecord' 
            )

            config['num_test_examples_plus_padding'] = len(examples)
            config['test_file'] = 'test.tfrecord'


    with open(config_path, mode='w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    return config



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        'olid_a': Olid_A_Processor,
        'olid_b': Olid_B_Processor,
        'olid_c': Olid_C_Processor
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.make_dataset_only:
        raise ValueError("At least one of `do_train`, `do_eval`, `do_predict' or `make_dataset_only` must be True.")

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            f"Cannot use sequence length {FLAGS.max_seq_length} because the BERT model "
            f"was only trained up to sequence length {bert_config.max_position_embeddings}"
        )

    task_name = FLAGS.task_name
    if task_name not in processors:
        raise ValueError(f"Task not found: {task_name}")

    processor = processors[task_name](FLAGS.local_data_dir)
    label_list = processor.get_labels()
    dataset_config = make_datasets(processor)

    if FLAGS.make_dataset_only:
        sys.exit(0)


    tf.gfile.MakeDirs(FLAGS.output_dir)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, 
            zone = FLAGS.tpu_zone, 
            project = FLAGS.gcp_project
        )

    
    if FLAGS.do_train:
        steps_per_epoch = math.ceil(dataset_config['num_train_examples'] / FLAGS.train_batch_size)
        save_checkpoints_steps = steps_per_epoch
        iterations_per_loop = steps_per_epoch
    else:
        save_checkpoints_steps = FLAGS.save_checkpoints_steps
        iterations_per_loop = FLAGS.iterations_per_loop

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster = tpu_cluster_resolver,
        master = FLAGS.master,
        model_dir = FLAGS.output_dir,
        save_checkpoints_steps = save_checkpoints_steps,
        keep_checkpoint_max = 10,
        tpu_config = tf.contrib.tpu.TPUConfig(
            iterations_per_loop = iterations_per_loop,
            num_shards = FLAGS.num_tpu_cores,
            per_host_input_for_training = is_per_host
        )
    )


    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        num_train_steps = steps_per_epoch * FLAGS.num_train_epochs
        num_warmup_steps = round(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config = bert_config,
        output_dir = FLAGS.output_dir,
        num_labels = len(label_list),
        init_checkpoint = FLAGS.init_checkpoint,
        learning_rate = FLAGS.learning_rate,
        num_train_steps = num_train_steps,
        num_warmup_steps = num_warmup_steps,
        use_tpu = FLAGS.use_tpu,
        use_one_hot_embeddings = FLAGS.use_tpu
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu = FLAGS.use_tpu,
        model_fn = model_fn,
        config = run_config,
        train_batch_size = FLAGS.train_batch_size,
        eval_batch_size = FLAGS.eval_batch_size,
        predict_batch_size = FLAGS.predict_batch_size
    )

    if FLAGS.do_train:
        train_tfrecord = _dataset_path(dataset_config['train_file'])
        
        tf.logging.info('***** Running training *****')
        tf.logging.info(f"  Num examples = {dataset_config['num_train_examples']}")
        tf.logging.info(f"  Batch size = {FLAGS.train_batch_size}")
        tf.logging.info(f"  Num steps = {num_train_steps}")

        train_input_fn = file_based_input_fn_builder(
            input_file = train_tfrecord,
            seq_length = FLAGS.max_seq_length,
            mode = 'train',
            drop_remainder = True,
            shuffle_buffer_size = dataset_config['num_train_examples']
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


    if FLAGS.do_eval:
        eval_file = _dataset_path(dataset_config['eval_file'])

        tf.logging.info('***** Running evaluation *****')
        tf.logging.info(f"  Num examples = {dataset_config['num_eval_examples']}")  
        tf.logging.info(f"  Batch size = {FLAGS.eval_batch_size}")

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the number of steps.
        if FLAGS.use_tpu:
            assert dataset_config['num_eval_examples_plus_padding'] % FLAGS.eval_batch_size == 0
            eval_steps = dataset_config['num_eval_examples_plus_padding'] // FLAGS.eval_batch_size

        eval_input_fn = file_based_input_fn_builder(
            input_file = eval_file,
            seq_length = FLAGS.max_seq_length,
            mode = 'eval',
            drop_remainder = FLAGS.use_tpu
        )
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        with tf.gfile.GFile(os.path.join(FLAGS.output_dir, 'eval_results.txt'), 'w') as writer:
        
            tf.logging.info('***** Eval results *****')
            for key in sorted(result.keys()):
                tf.logging.info(f"  {key} = {result[key]}")
                writer.write(f"{key} = {result[key]}")

    if FLAGS.do_predict:
        test_file = _dataset_path(dataset_config['test_file'])

        tf.logging.info('***** Running prediction *****')
        tf.logging.info(f"  Num examples = {dataset_config['num_test_examples']}")
        tf.logging.info(f"  Batch size = {FLAGS.predict_batch_size}")

        predict_input_fn = file_based_input_fn_builder(
            input_file = test_file,
            seq_length = FLAGS.max_seq_length,
            mode = 'test',
            drop_remainder = FLAGS.use_tpu
        )
        result = estimator.predict(input_fn=predict_input_fn)

        with tf.gfile.GFile(os.path.join(FLAGS.output_dir, 'test_results.tsv'), 'w') as writer:
            
            tf.logging.info('***** Predict results *****')

            num_written_lines = 0
            for i, prediction in enumerate(result):
                if i >= dataset_config['num_test_examples']:
                    break

                probabilities = prediction['probabilities']
                
                output_line = '\t'.join( str(class_probability) for class_probability in probabilities) + '\n'
                writer.write(output_line)
                num_written_lines += 1

            assert num_written_lines == dataset_config['num_test_examples']



if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('local_data_dir')
    flags.mark_flag_as_required('task_name')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('bert_config_file')
    flags.mark_flag_as_required('output_dir')
    tf.app.run()

# NLP
### hw1
##### Required packages
- tensorflow-gpu 1.12
- jieba
- tqdm
- jupyter notebook
- tensorflow-gpu 1.13.1 (model_bert_basic)
- bert-as-service

##### Preprocessing
- Use `Preprocessing.ipynb` to tokenize sentences and turn them into sequences of int values.
- Use `WordEmbedding.ipynb` to pretrain word vectors.
- Use `model_bert_basic/preprocessing.py` to generate datasets.

##### Training
- Use `RNN_basic.ipynb` to train a simple model w/o attention mechanism.
- Use `RNN_complex.ipynb` to train a attentional model.
- Use `model_bert_basic/predict.py` to train a model based on BERT.

##### Experiment
- Add Convolution Layer
	- Use `Var1_Simple_CNN_network.ipynb` to train a CNN model w/o attention
	- Use `Var2_Convolution_before_RNN.ipynb` to train a RNN_basic model w/ additional Convolution Layer
	- Use `Var3_Convolution_Layer_after_Attentional_bi-LSTM_network.ipynb` to train a RNN_complex mode w/ additional Convolution Layer

##### Data & Submission
https://www.kaggle.com/c/fake-news-pair-classification-challenge/

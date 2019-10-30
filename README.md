# Lexical-AT 
This is the codes of our paper: LexicalAT: Lexical-Based Adversarial Reinforcement Trainingfor Robust Sentiment Classification.
![The architecture of our Lexical-AT](https://github.com/lancopku/LexicalAT/blob/master/model.jpg)
# Requirements
* Ubuntu 16.0.4
* Python 3.6
* Tensorflow  1.12
* NLTK 3.4.3
# Training Examples
## For CNN or LSTM
```
cd cnn_lstm
CUDA_VISIBLE_DEVIVES=0 bash sst_adv_cnn.sh
```
## For Bert
Before executing the command below, you need to download the pretrained model and vocabulary of bert. The download url and other detaild information can be found in [bert](https://github.com/google-research/bert).
```
cd bert 
CUDA_VISIBLE_DEVICES=0 bash rt_adv_bert.sh
```

# Note
- Before running the code, you need to download wordnet via nltk.
- The code is currently non-deterministic due to various GPU ops, so you are likely to end up with a slightly better or worse evaluation.
# Citation
If you use the above codes for your research, please cite our paper.
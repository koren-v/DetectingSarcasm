# Detecting Sarcasm in News Headlines

### Description

Can you identify sarcastic sentences? Can you distinguish between fake news and legitimate news?
In this repository, you can find an implementation of several RNNs for detecting sarcasm in news headlines and also using pre-trained BERT for this task. There is one important notice before looking at comparison table that RNNs are not pre-trained they just use pre-trained GloVe vectors. 
Dataset was taken from [Kaggle](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection) 

### Models

* [Usual LSTM](https://github.com/koren-v/DetectingSarcasm/blob/master/Models/LSTM.py)
* [Bidirectional LSTM](https://github.com/koren-v/DetectingSarcasm/blob/master/Models/BidirectionalLSTM.py)
* [LSTM with Attantion](https://github.com/koren-v/DetectingSarcasm/blob/master/Models/AttantionLSTM.py)
* [LSTM with 2D MaxPooling layer](https://github.com/koren-v/DetectingSarcasm/blob/master/Models/LSTM2DMaxPool.py)
* [BERT](https://github.com/koren-v/DetectingSarcasm/blob/master/Notebooks/BERT.ipynb)

### Comparison

Obviously the pre-trained model got better performance, therefore BERT showed the highest accuracy. LSTM with 2D MaxPooling layer, Bidirectional LSTM, LSTM with Attention showed approximately the same result, while usual LSTM was little bit worse.

|  |  Loss  | Validation Accuracy | Recall | Precision | F1 |
| --- | --- | --- | --- | --- | --- |
| Usual LSTM                     | 0.6616 |        0.8660       | 0.8190 | 0.8890 | 0.8526 |
| Bidirectional LSTM             | 0.6558 |        0.8697       | 0.8944 | 0.8432 | 0.8681 |
| LSTM with Attantion            | 0.6555 |        0.8698       | 0.9143 | 0.8188 | 0.8639 |
| LSTM with 2D MaxPooling leyer  | 0.6515 |        0.8719       | 0.8712 | 0.8649 | 0.8680 |
| BERT                           | 0.3999 |        0.9092       | 0.8739 | 0.9113 | 0.8922 |

Model's weights you can find [here](https://github.com/koren-v/DetectingSarcasm/tree/master/Models/Weights)

For searching hyperparameters such as learning rate, weight decay and hidden size of LSTMs was used HyperOpt Algorithm from Tune library. You can find more detail in [this](https://github.com/koren-v/DetectingSarcasm/blob/master/Notebooks/Experements/TuneLSTMs.ipynb) notebook.

Sometimes, it's important to see examples, where your model makes mistakes. It was implemented in [ModelMistakes notebook](https://github.com/koren-v/DetectingSarcasm/blob/master/Notebooks/Experements/ModelMistakes.ipynb) and you can find some examples there. 

### Data Preprocessing

After an attempt to do data cleanings such as lowercasing, noise removal, lemmatization, and stop-words removal was found that it made worse result. So it can mean that the dataset is good enough, and we can skip this step to save more information. This experiment you can find [here](https://github.com/koren-v/DetectingSarcasm/blob/master/Notebooks/Experements/CleanedDataLSTMs.ipynb)

### References

* Text Classification Improved by Integrating Bidirectional LSTM
with Two-dimensional Max Pooling [paper](https://www.aclweb.org/anthology/C16-1329.pdf)
* BERT Classifier: Just Another Pytorch Model [article](https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784)
* Algorithms for Hyper-Parameter Optimization [paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
* Tune Search Algorithms [docs](https://ray.readthedocs.io/en/latest/tune-searchalg.html)
* Text-Classification-Pytorch [repository](https://github.com/prakashpandey9/Text-Classification-Pytorch#license)

# Detecting Sarcasm in News Headlines

### Description

Can you identify sarcastic sentences? Can you distinguish between fake news and legitimate news?
In this repository you can find implementation of several RNNs for detecting sarcasm in news headline and also using of pre-trained BERT for this task.
Dataset was taken from [Kaggle](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection) 

### Models

* [Usual LSTM](https://github.com/sqrt420/DetectingSarcasm/blob/master/LSTM.py)
* [Bidirectional LSTM](https://github.com/sqrt420/DetectingSarcasm/blob/master/BidirectionalLSTM.py)
* [LSTM with Attantion](https://github.com/sqrt420/DetectingSarcasm/blob/master/AttantionLSTM.py)
* [LSTM with 2D MaxPooling leyer](https://github.com/sqrt420/DetectingSarcasm/blob/master/LSTM2DMaxPool.py)
* [BERT](https://github.com/sqrt420/DetectingSarcasm/blob/master/BERT.ipynb)

### Comparison

Obviusly pre-trained model got better performance, therefore BERT showed the highest accuracy. LSTM with 2D MaxPooling leyer, Bidirectional LSTM, LSTM with Attantion showed approximately the same result, while usual LSTM was litle bit worse.

|  |  Loss  | Validation Accuracy |
| --- | --- | --- |
| Usual LSTM                     | 0.5056 |        0.8557       |
| Bidirectional LSTM             | 0.4967 |        0.8669       |
| LSTM with Attantion            | 0.4983 |        0.8650       |
| LSTM with 2D MaxPooling leyer  | 0.4977 |        0.8627       |
| BERT                           | 0.3999 |        0.9092       |

Model's weights you can find [here](https://github.com/sqrt420/DetectingSarcasm/tree/master/ModelWeights)

For searching hyperparameters such as learning rate, weight decay and hidden size of LSTMs was used HyperOpt Algorithm from Tune library. You can find more detail in [this](https://github.com/sqrt420/DetectingSarcasm/blob/master/TuneLSTMs.ipynb) notebook.

### Data Preprocesing

After [attempt](https://github.com/sqrt420/DetectingSarcasm/blob/master/CleanedDataLSTMs.ipynb) to do data cleaning such as lowercasing, noise removal, lemmatization and stop-words removal wos found that it made worse result. So it can mean that dataset is good anought, and we can skip this step to save more information.

### References

* Text Classification Improved by Integrating Bidirectional LSTM
with Two-dimensional Max Pooling [[paper](https://www.aclweb.org/anthology/C16-1329.pdf)]
* BERT Classifier: Just Another Pytorch Model [[article](https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784)]
* Algorithms for Hyper-Parameter Optimization [[paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)]
* Tune Search Algorithms [[docs](https://ray.readthedocs.io/en/latest/tune-searchalg.html)]
* Text-Classification-Pytorch [[repository](https://github.com/prakashpandey9/Text-Classification-Pytorch#license)]



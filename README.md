# RAD_Version_Python
Implementation of Robust PCA and Robust Deep Autoencoder over Time Series for detection of Outliers.


# Description

This repository contains code of Robust PCA  and Robust Deep Autoencoder. Inspired by the [Surus Project](https://github.com/Netflix/Surus) (Netflix), I made a version of Robust PCA for Time Series in order to compare the efficiency for the detection of outliers compared to Robust Deep Autoencoder (in Time Series).

# Functions

The models are in two functions:

 *  AnomalyDetection_RPCA: implementaion of Robust PCA similar of the Netflix's propuest.
 *  AnomalyDetection_AUTOENCODER: implementation and adaptation of paper "Anomaly Detection with Robust Deep Auto-encoders" Chong Zhou;Randy Paffenroth.

# Result

 * [Robust Deep Autoencoder:](http://nbviewer.jupyter.org/github/dlegor/RAD_Version_Python/blob/master/Notebook/Examples_and_Tests-Autoencoder.ipynb)
 
 * [Robust PCA](http://nbviewer.jupyter.org/github/dlegor/RAD_Version_Python/blob/master/Notebook/Examples_and_Tests-rPCA.ipynb)

# References:

* [Stable Principal Component Pursuit](https://arxiv.org/abs/1001.2363)
* [Robust Principal Component Analysis?](http://statweb.stanford.edu/~candes/papers/RobustPCA.pdf)
* [https://www.kdd.org/kdd2017/papers/view/anomaly-detection-with-robust-deep-auto-encoders](https://www.kdd.org/kdd2017/papers/view/anomaly-detection-with-robust-deep-auto-encoders)

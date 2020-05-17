====
RAD
====

Implementation of Robust PCA and Robust Deep Autoencoder for Time Series to detect outliers.

-----------
Description
-----------

This repository contains code of Robust PCA  and Robust Deep Autoencoder. Inspired by the `Surus Project <https://github.com/Netflix/Surus>`_ ( from Netflix ), I made a version of Robust PCA for Time Series in order to compare the efficiency for the detection of outliers compared to Robust Deep Autoencoder (for Time Series).

---------
Functions
---------

The models are in two functions:

- AnomalyDetection_RPCA: implementaion of Robust PCA similar of the Netflix's propuest.
- (**DEPRECATED**) AnomalyDetection_AUTOENCODER: implementation and adaptation of paper "Anomaly Detection with Robust Deep Auto-encoders" Chong Zhou;Randy Paffenroth.

--------
Examples
--------

- `Robust Deep Autoencoder <http://nbviewer.jupyter.org/github/dlegor/RAD_Version_Python/blob/master/Notebook/Examples_and_Tests-Autoencoder.ipynb>`_ (DEPRECATED)
 
- `Robust PCA <http://nbviewer.jupyter.org/github/dlegor/rad/blob/master/notebooks/Examples_and_Tests-rPCA.ipynb>`_

----------
Installing 
----------
RAD depends upon ``scikit-learn`` and ``numba`` .

Requirements:

* Python 3.6 or greater
* numpy
* scipy
* scikit-learn
* numba
* pandas

For a manual install get this package:

.. code:: bash

    wget https://github.com/dlegor/rad/archive/master.zip
    unzip master.zip
    rm master.zip
    cd rad

Install the requirements

.. code:: bash

    sudo pip install -r requirements.txt

or

.. code:: bash

    conda install --file requirements.txt

Install the package

.. code:: bash

    python setup.py install

***********
References:
***********

* `Stable Principal Component Pursuit <https://arxiv.org/abs/1001.2363>`_
* `Robust Principal Component Analysis? <http://statweb.stanford.edu/~candes/papers/RobustPCA.pdf>`_
* `Anomaly Detection with Robust Deep Auto-encoders <https://www.kdd.org/kdd2017/papers/view/anomaly-detection-with-robust-deep-auto-encoders>`_

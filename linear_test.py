# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 13:59:34 2018

@author: eobad
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import csv

from tensorflow.contrib.learn.python.learn.datasets import base


def load_csv(filename):
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        data,target = [],[]
        for row in data_file:
            target.append(row.pop(19))
            for element in row:
                if type(element) is not str:
                    data.append(np.asarray(row, dtype=feature_type))
                    break
    target = np.array(target, dtype=str)
    data = np.array(data)
    return base.Dataset(data=data, target = target)

# Data files
TRAIN = "./data/ipo_data.csv"
TEST = "./data/ipo_test.csv"

# Load datasets.
training_set = load_csv(TRAIN) 
'''base.load_csv_with_header(filename=TRAIN,
                                         features_dtype=np.float32,
                                         target_dtype=np.str)'''
test_set = load_csv(TEST) 
'''base.load_csv_with_header(filename=TEST,
                                     features_dtype=np.float32,
                                     target_dtype=np.str)'''

# Specify that all features have real-value data
feature_name = "ipo_features"
feature_columns = [tf.feature_column.numeric_column(feature_name, 
                                                    shape=[4])]
classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    n_classes=2,
    model_dir="/tmp/test_model")

def input_fn(dataset):
    def _fn():
        features = {feature_name: tf.constant(dataset.data)}
        label = tf.constant(dataset.target)
        return features, label
    return _fn

# Fit model.
classifier.train(input_fn=input_fn(training_set),
               steps=1000)
print('fit done')

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=input_fn(test_set), 
                                     steps=100)["accuracy"]
print('\nAccuracy: {0:f}'.format(accuracy_score))

# Export the model for serving
feature_spec = {'flower_features': tf.FixedLenFeature(shape=[4], dtype=np.float32)}

serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

classifier.export_savedmodel(export_dir_base='/tmp/iris_model' + '/export', 
                            serving_input_receiver_fn=serving_fn)
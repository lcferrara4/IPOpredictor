# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys
import matplotlib.pyplot as plt
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.arg_parsers import parsers
from official.utils.logs import hooks_helper
from official.utils.misc import model_helpers

def plot(results, model_type, epoch_iter):
  epoch_list = []
  print(epoch_iter)
  for i in range(epoch_iter):
    epoch_list.append(i)
  for key in results.keys():
    if key == 'accuracy' or key == 'auc_precision_recall' or key == 'prediction/mean':
      plt.plot(epoch_list, results[key])
      plt.title('Graph of ' + str(key) + ' for ' +str(model_type))
      plt.xlabel('epoch')
      plt.ylabel(key)
      plt.show()
        

_CSV_COLUMNS = [
    'tx_month', 'tx_day', 'tx_year', 'exchange', 'tx_value', 'price_per_share',
    'offer_month', 'offer_day', 'offer_year', 'days_between', 'num_shares', 
    'overallotment', 'day1_return', 'month1_return', 'month3_return',
    'percent_to_company', 'percent_to_shareholders', 'country', 'industry', 'class'
]

_CSV_COLUMN_DEFAULTS = [[1], [1], [2000], [''], [0.0], [0.0], [1], [1], [2000], [0],
                        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [''], [''], ['']]

_NUM_EXAMPLES = {
    'train': 3755,
    'validation': 418,
}


LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}



def build_model_columns(export_flag):
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  tx_month = tf.feature_column.numeric_column('tx_month')
  tx_day = tf.feature_column.numeric_column('tx_day')
  tx_year = tf.feature_column.numeric_column('tx_year')
  tx_value = tf.feature_column.numeric_column('tx_value')
  price_per_share = tf.feature_column.numeric_column('price_per_share')
  offer_month = tf.feature_column.numeric_column('offer_month')
  offer_day = tf.feature_column.numeric_column('offer_day')
  offer_year = tf.feature_column.numeric_column('offer_year')
  days_between = tf.feature_column.numeric_column('days_between')
  num_shares = tf.feature_column.numeric_column('num_shares')
  overallotment = tf.feature_column.numeric_column('overallotment')
  day1_return = tf.feature_column.numeric_column('day1_return')
  month1_return = tf.feature_column.numeric_column('month1_return')
  month3_return = tf.feature_column.numeric_column('month3_return')
  percent_to_company = tf.feature_column.numeric_column('percent_to_company')
  percent_to_shareholders = tf.feature_column.numeric_column('percent_to_shareholders')
 

  exchange = tf.feature_column.categorical_column_with_vocabulary_list(
      'exchange', [
          'OB', 'NZSE', 'NasdaqGM', 'ENXTPA', 'SEHK', 'TSEC', 'WSE', 'BIT', 
          'NYSE', 'OTCPK', 'LSE', 'MutualFund', 'ISE', 'AMEX', 'TSE', 
          'NasdaqCM', 'IBSE', 'BMV', 'SHSE', 'SNSE', 'TSX', 'BOVESPA', 
          'TASE', 'KOSE', 'BASE', 'NasdaqGS'])

  country = tf.feature_column.categorical_column_with_vocabulary_list(
      'country', [
          'Israel', 'India', 'Denmark', 'Poland', 'Turkey', 'Cyprus', 
          'Mexico', 'Jordan', 'South Korea', 'Mongolia', 'United Arab Emirates', 
          'Germany', 'Jersey', 'Cayman Islands', 'Spain', 'Norway', 'Italy', 
          'Singapore', 'Belgium', 'China', 'Sweden', 'Monaco', 'Hong Kong', 
          'Greece', 'Chile', 'Brazil', 'Luxembourg', 'Indonesia', 'France', 
          'Uruguay', 'New Zealand', 'Argentina', 'Netherlands', 'Bahamas', 
          'Russia', 'Ireland', 'Taiwan', 'Canada', 'Bulgaria', 'Panama', 'Bermuda', 
          'Switzerland', 'United Kingdom', 'United States', 'Japan', 
          'British Virgin Islands'])

  industry = tf.feature_column.categorical_column_with_vocabulary_list(
      'industry', [
          'Other Diversified Financial Services', 'Consumer Finance', 'Oil and Gas Refining and Marketing', 
          'Food Retail', 'Diversified Chemicals', 'Brewers', 'Advertising', 'Application Software', 
          'Diversified Capital Markets', 'Residential REITs', 'Consumer Electronics', 
          'Real Estate Operating Companies', 'Construction Machinery and Heavy Trucks', 'Aerospace and Defense', 
          'Oil and Gas Storage and Transportation', 'Leisure Facilities', 'Heavy Electrical Equipment', 'Broadcasting', 
          'Air Freight and Logistics', 'Distillers and Vintners', 'Electrical Components and Equipment', 
          'Healthcare Equipment', 'Drug Retail', 'IT Consulting and Other Services', 'Steel', 'Asset Management and Custody Banks', 
          'Healthcare Services', 'Healthcare Distributors', 'Life Sciences Tools and Services', 'Personal Products', 
          'Integrated Telecommunication Services', 'Forest Products', 'Oil and Gas Exploration and Production', 
          'Specialized REITs', 'Reinsurance', 'Electronic Manufacturing Services', 'Tobacco', 'Water Utilities', 
          'Thrifts and Mortgage Finance', 'Diversified Support Services', 'Diversified Metals and Mining', 
          'Environmental and Facilities Services', 'Renewable Electricity', 'Specialized Consumer Services', 'Diversified Banks', 
          'Regional Banks', 'Healthcare REITs', 'Security and Alarm Services', 'Household Products', 'Soft Drinks', 
          'Mortgage REITs', 'General Merchandise Stores', 'Footwear', 'Life and Health Insurance', 
          'Data Processing and Outsourced Services', 'Multi-Sector Holdings', 'Specialty Chemicals', 
          'Internet and Direct Marketing Retail', 'Human Resource and Employment Services', 'Hotels, Resorts and Cruise Lines', 
          'Semiconductor Equipment', 'Internet Software and Services', 'Packaged Foods and Meats', 'Specialty Stores', 
          'Investment Banking and Brokerage', 'Construction and Engineering', 'Aluminum', 'Gas Utilities', 
          'Electric Utilities', 'Diversified Real Estate Activities', 'Movies and Entertainment', 'Health Care Technology', 
          'Electronic Components', 'Property and Casualty Insurance', 'Gold', 'Distributors', 'Pharmaceuticals', 
          'Oil and Gas Drilling', 'Semiconductors', 'Automotive Retail', 'Restaurants', 'Home Furnishings', 'Commercial Printing', 
          'Agricultural and Farm Machinery', 'Communications Equipment', 'Financial Exchanges and Data', 'Systems Software', 
          'Trading Companies and Distributors', 'Technology Distributors', 'Household Appliances', 'Construction Materials', 
          'Airport Services', 'Oil and Gas Equipment and Services', 'Diversified REITs', 'Publishing', 'Coal and Consumable Fuels', 
          'Independent Power Producers and Energy Traders', 'Education Services', 'Automobile Manufacturers', 'Homebuilding', 
          'Paper Packaging', 'Specialized Finance', 'Alternative Carriers', 'Leisure Products', 'Home Improvement Retail', 
          'Fertilizers and Agricultural Chemicals', 'Retail REITs', 'Food Distributors', 'Metal and Glass Containers', 
          'Wireless Telecommunication Services', 'Apparel, Accessories and Luxury Goods', 'Electronic Equipment and Instruments', 
          'Commodity Chemicals', 'Research and Consulting Services', 'Healthcare Facilities', 'Industrial Machinery', 
          'Real Estate Services', 'Building Products', 'Agricultural Products', 'Auto Parts and Equipment', 'Tires and Rubber', 
          'Biotechnology', 'Railroads', 'Office REITs', 'Textiles', 'Airlines', 'Home Furnishing Retail', 'Real Estate Development', 
          'Trucking', 'Industrial REITs', 'Integrated Oil and Gas', 'Multi-line Insurance', 'Insurance Brokers', 'Apparel Retail', 
          'Office Services and Supplies', 'Technology Hardware, Storage and Peripherals', 'Managed Healthcare', 'Healthcare Supplies', 
          'Home Entertainment Software', 'Marine Ports and Services', 'Cable and Satellite', 'Housewares and Specialties', 'Paper Products', 
          'Computer and Electronics Retail', 'Marine', 'Hotel and Resort REITs', 'Casinos and Gaming'])

  '''
  # To show an example of hashing:
  occupation = tf.feature_column.categorical_column_with_hash_bucket(
      'occupation', hash_bucket_size=1000)

  # Transformations.
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
  '''

  # Wide columns and deep columns.
  base_columns = [
      tx_month, tx_day, tx_year, exchange, tx_value, price_per_share,
      offer_month, offer_day, offer_year, days_between, num_shares, 
      overallotment, day1_return, month1_return, month3_return,
      percent_to_company, percent_to_shareholders, country, industry
  ]

  crossed_columns = [
      tf.feature_column.crossed_column(
          ['country', 'exchange'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          ['country', 'industry'], hash_bucket_size=1000),
  ]

  wide_columns = base_columns + crossed_columns

  deep_columns = [
      tx_month, tx_day, tx_year, tx_value, price_per_share,
      offer_month, offer_day, offer_year, days_between, num_shares, 
      overallotment, day1_return, month1_return, month3_return,
      percent_to_company, percent_to_shareholders,
      tf.feature_column.indicator_column(country),
      tf.feature_column.indicator_column(exchange),
      tf.feature_column.indicator_column(industry),
      # To show an example of embedding
      #tf.feature_column.embedding_column(occupation, dimension=8),
  ]
  if export_flag:
      return crossed_columns + deep_columns
  else:
      return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns(False)
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run data_download.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('class')
    return features, tf.equal(labels, 'increase')

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset

def export_model(model, model_type, export_dir):
    feature_columns = build_model_columns(True)
    #wide_columns, deep_columns = build_model_columns()
    #if model_type == 'wide':
    #    columns=wide_columns
    #elif model_type == 'deep':
    ##    columns=deep_columns
    #else:
    #    columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_fun = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    servable_model_path = model.export_savedmodel('/tmp/wide_deep', serving_input_fun)
    servable_model_path

def main(argv):
  parser = WideDeepArgParser()
  flags = parser.parse_args(args=argv[1:])

  # Clean up the model directory if present
  shutil.rmtree(flags.model_dir, ignore_errors=True)
  model = build_estimator(flags.model_dir, flags.model_type)

  train_file = os.path.join(flags.data_dir, 'ipo_data.csv')
  test_file = os.path.join(flags.data_dir, 'ipo_test.csv')

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return input_fn(
        train_file, flags.epochs_between_evals, True, flags.batch_size)

  def eval_input_fn():
    return input_fn(test_file, 1, False, flags.batch_size)

  loss_prefix = LOSS_PREFIX.get(flags.model_type, '')
  train_hooks = hooks_helper.get_train_hooks(
      flags.hooks, batch_size=flags.batch_size,
      tensors_to_log={'average_loss': loss_prefix + 'head/truediv',
                      'loss': loss_prefix + 'head/weighted_loss/Sum'})

  with tf.Session() as sess:
  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    output = {}  
    epoch_count = 0
    for n in range(flags.train_epochs // flags.epochs_between_evals):
      model.train(input_fn=train_input_fn, hooks=train_hooks)
      results = model.evaluate(input_fn=eval_input_fn)

    # Display evaluation metrics
      print('Results at epoch', (n + 1) * flags.epochs_between_evals)
      print('-' * 60)

      for key in sorted(results):
          if key in output.keys():
              output[key].append(results[key])
          else:
              output[key] = [results[key]]
          print('%s: %s' % (key, results[key]))
      epoch_count+=1
      if model_helpers.past_stop_threshold(flags.stop_threshold, results['accuracy']):
        break
        

  export_model(model, flags.model_type, '/tmp/wide_deep')
  meta_graph_def = tf.train.export_meta_graph(filename='/tmp/wide_deep/my-model.meta')
  
  plot(output, flags.model_type, epoch_count)

class WideDeepArgParser(argparse.ArgumentParser):
  """Argument parser for running the wide deep model."""

  def __init__(self):
    super(WideDeepArgParser, self).__init__(parents=[parsers.BaseParser()])
    self.add_argument(
        '--model_type', '-mt', type=str, default='wide_deep',
        choices=['wide', 'deep', 'wide_deep'],
        help='[default %(default)s] Valid model types: wide, deep, wide_deep.',
        metavar='<MT>')
    self.set_defaults(
        data_dir='./data',
        model_dir='./data_model',
        train_epochs=40,
        epochs_between_evals=2,
        batch_size=40)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)


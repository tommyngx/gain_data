'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from data_loader import data_loader
from gain import gain
from utils import rmse_loss
from missingpy import MissForest
from sklearn import metrics
from math import sqrt
from impyute.imputation.cs import mice
import pandas as pd
from autoimpute.imputations import MiceImputer, SingleImputer, MultipleImputer
from autoimpute.analysis import MiLinearRegression


def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
  
  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters)
  
  # Report the RMSE performance
  rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
  print()
  mi_data = miss_data_x.astype(float)
  no, dim = imputed_data_x.shape
  miss_data = np.reshape(mi_data,(no,dim))
  np.savetxt("data/missing_data.csv",mi_data,delimiter=',',fmt='%1.2f')
  print( 'Shape of miss data: ',miss_data.shape)
  print( 'Save results in missing_data.csv')
  
  print()
  print('=== GAIN RMSE ===')
  print('RMSE Performance: ' + str(np.round(rmse, 6)))
  #print('Kích thước của file đầu ra: ', imputed_data_x.shape)
  np.savetxt("data/imputed_data.csv",imputed_data_x, delimiter=',',  fmt='%d')
  print( 'Save results in Imputed_data.csv')
  
  # MissForest

  print()
  print('=== MissForest RMSE ===')
  data = miss_data_x
  imp_mean = MissForest(max_iter = 5)
  miss_f = imp_mean.fit_transform(data)
  #miss_f = pd.DataFrame(imputed_train_df)
  rmse_MF = rmse_loss (ori_data_x, miss_f, data_m)
  print('RMSE Performance: ' + str(np.round(rmse_MF, 6)))
  np.savetxt("data/imputed_data_MF.csv",miss_f, delimiter=',',  fmt='%d')
  print( 'Save results in Imputed_data_MF.csv')

  # MICE From Auto Impute
  print()
  print('=== MICE of Auto Impute RMSE ===')
  data_mice = pd.DataFrame(miss_data_x)
  mi = MiceImputer(k=1, imp_kwgs=None, n=1, predictors='all', return_list=True,
        seed=None, strategy='default predictive', visit='default')
  mice_out = mi.fit_transform(data_mice)
  c = [list(x) for x in mice_out]
  c1= c[0]
  c2=c1[1]
  c3=np.asarray(c2)
  mice_x=c3
  #print('here :', mice_x, miss_f, miss_f.shape)
  rmse_MICE = rmse_loss (ori_data_x, mice_x, data_m)
  print('=== MICE of Auto Impute RMSE ===')
  print('RMSE Performance: ' + str(np.round(rmse_MICE, 6)))
  np.savetxt("data/imputed_data_MICE.csv",mice_x, delimiter=',',  fmt='%d')
  print( 'Save results in Imputed_data_MICE.csv')


  return imputed_data_x, rmse


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam', 'breast', 'credit', 'news'],
      default='spam',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse = main(args)

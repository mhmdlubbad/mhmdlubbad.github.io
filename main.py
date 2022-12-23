import pandas as pd
import numpy as np
import datetime
from pandas._libs.tslibs.timestamps import Timestamp

DEBUG = True
def log(*args):
    if DEBUG:
        print(args)

class Table():
  def __init__(self, csv_path, table_name = None):
    ''' Initiation the table '''
    # define table name
    self.table_name = table_name if table_name is not None else csv_path
    
    # read the csv file
    try:
      self.dataframe = pd.read_csv(csv_path, 
                                   on_bad_lines='warn')
      log('[TRACE] CSV file is loaded successfully')
    except:
      self.dataframe = pd.DataFrame({})
      log('[ERROR] Error while loading the csv file, try another file')
    
    # convert possible date columns
    self.convert_possible_date_columns()
    # self.observe_redundant_datetime_columns()

    # prepare the dataframe metadata; data types, uniqueness, and null values
    self.prepare_dataframe_metadata()

    # calculate the time distrubtion; by day, week, year, etc.
    self.get_best_time_distribution()

    # 

    # print the metadata
    self.print_metadata()

    

  def convert_possible_date_columns(self):
    ''' converts string columns to dates '''
    # for each column, try to convert non-numeric to dates
    # and numeric to numeric values try to convert it to a date
    for column in self.dataframe.columns:
      if not pd.api.types.is_numeric_dtype(self.dataframe[column].dtype):
        try:
          self.dataframe[column] = pd.to_datetime(self.dataframe[column]).\
          astype(dtype='datetime64[ns]')
        except:
          pass
      else:
        try:
          self.dataframe[column] = pd.to_numeric(self.dataframe[column])
        except:
          pass 
    log('[TRACE] possible columns are converted to dates')

  def prepare_dataframe_metadata(self):
    '''prepare the metadata of the dataframe; such as data types,
    uniqueness, and null values'''

    self.metadata = pd.DataFrame({})

    # check the type of each element in the dataframe
    self.metadata['Type'] = self.dataframe.apply(
        lambda y:
        list(set([type(x) for x in filter(lambda v: v==v, y)]))[0],
        raw=True
    )

    # compute the uniqueness of columns' values
    self.metadata['# Unique'] = self.dataframe.apply(
        lambda y:
        len(set([x for x in filter(lambda v: v==v, y)])),
        raw=True
    )

    # compute columns' non-null values
    self.metadata['# Non-null'] = self.dataframe.apply(
        lambda y:
        len([x for x in filter(lambda v: v==v, y)]),
        raw=True
    )

    log('[TRACE] dataframe summary is generated')

  
  def get_best_time_distribution(self):
    ''' to compute the time distribution for date columns,
    e.g # categories by month, day or seconds 
    this function is not mature yet'''

    self.date_formats_distribution = pd.DataFrame({'date-formats':['%Y', '%Y-%q', '%Y-%m', '%Y-%U', '%Y-%m-%d', '%Y-%m-%d %h', '%Y-%m-%d %h:%m', '%Y-%m-%d %h:%m:%s'],
                                    'description': ['by year', 'by quarter', 'by month', 'by week', 'by day', 'by hour', 'by minute', 'by second']})

    for column in self.dataframe.columns:
      if self.dataframe[column].dtype == '<M8[ns]':
        self.date_formats_distribution[column] = self.date_formats_distribution['date-formats'].apply(lambda x: len(set(self.dataframe[column].dt.strftime(x))))

    self.date_formats_distribution
    log('[TRACE] datetime columns are distributed')

  def observe_redundant_datetime_columns(self):
    ''' to observe redundant columns, such as: year, month, etc.'''
    # get datetime columns
    datetime_correlator = self.dataframe[[col for col in self.dataframe.columns
                                               if self.dataframe[col].dtype == '<M8[ns]']]
    time_formats = ['%Y', '%m', '%U', '%d', '%H', '%M', '%s']
   
    for datetime_column in datetime_correlator.columns:
      for time_format in time_formats:
        # represent date in different timeformat
        timeformat_column = datetime_correlator[datetime_column].apply(lambda t: int(t.strftime(time_format)))
        # check if duplicate -> drop the duplicate columns
        for column in self.dataframe.columns:
          if timeformat_column.equals(self.dataframe[column]):
            self.dataframe = self.dataframe.drop(column, axis=1)
            log(f'{column} is dropped because of duplication with {datetime_column}_{time_format}')
      # represent date in quarter format
      timeformat_column = datetime_correlator[datetime_column].dt.quarter
      time_format = '%q'
      for column in self.dataframe.columns:
        if timeformat_column.equals(self.dataframe[column]):
          self.dataframe = self.dataframe.drop(column, axis=1)
          log(f'{column} is dropped because of duplication with {datetime_column}_{time_format}')



  def print_metadata(self):
    ''' print the metadata of dataframe'''
    # order of types: string -> int -> float -> dates
    self.metadata['Type'] = pd.Categorical(self.metadata['Type'],
                                          categories=[bool, str, int, float, Timestamp],#[bool, category, object, int, float64, <M8[ns]],
                                          ordered=True)
    # get most unique
    self.most_unique = self.metadata.sort_values(['# Unique', 'Type'], ascending = [False, True])
    # log(f"{self.most_unique.loc[0,'# Non-null']} == {len(self.dataframe)} {self.most_unique.loc[0,'# Non-null'] == len(self.dataframe)}")
    # if self.most_unique.loc[0,'# Non-null'] == len(self.dataframe):
    #   self.most_unique = self.most_unique[[self.most_unique.index[0]]]
      
    log('suggestion for values', self.metadata.sort_values(['# Unique', 'Type'], ascending = [False, True]))
    # get least unique
    self.least_unique = self.metadata.sort_values(['# Unique', 'Type'], ascending = [True, True])
    self.least_unique['area_of_attention'] = self.least_unique[['# Unique']].cumprod()
    log('----\nsuggestion for columns and indices\n----\n', self.least_unique)
    log('getting choices')
    self.choices = self.most_unique.index
    log(self.choices)



                                                                                              
#  `7MMF'     A     `7MF'`7MM"""Mq.       db     `7MN.   `7MF' .g8"""bgd `7MMF'      `7MM"""YMM   # 
#   `MA     ,MA     ,V    MM   `MM.     ;MM:      MMN.    M .dP'     `M   MM          MM    `7    #
#    VM:   ,VVM:   ,V     MM   ,M9     ,V^MM.     M YMb   M dM'       `   MM          MM   d      #
#     MM.  M' MM.  M'     MMmmdM9     ,M  `MM     M  `MN. M MM            MM          MMmmMM      #
#     `MM A'  `MM A'      MM  YM.     AbmmmqMA    M   `MM.M MM.    `7MMF' MM      ,   MM   Y  ,   #
#      :MM;    :MM;       MM   `Mb.  A'     VML   M     YMM `Mb.     MM   MM     ,M   MM     ,M   #
#       VF      VF      .JMML. .JMM.AMA.   .AMMA.JML.    YM   `"bmmmdPY .JMMmmmmMMM .JMMmmmmMMM   #             
#|-----------------------------------------------------------------------------------------------|#
#|-----------------------------------------------------------------------------------------------|#
# IMPORTS
from env import host, username, password, get_db_url
import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
#|-----------------------------------------------------------------------------------------------|#
#|-----------------------------------------------------------------------------------------------|#
def acquire_zillow_data(use_cache=True):
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('zillow.csv')
    print('Acquiring data from SQL database')
    df = pd.read_sql('''
                    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,
                    taxvaluedollarcnt, yearbuilt, taxamount, fips
                        FROM properties_2017
                        JOIN propertylandusetype USING(propertylandusetypeid)
                        WHERE propertylandusetypeid = 261
                     '''
                    , get_db_url('zillow'))
    df.to_csv('zillow.csv', index=False)
    
    
    return df
#|-----------------------------------------------------------------------------------------------|#
def remove_outliers(df, k, col_list):
    for col in col_list:
        
        q1, q3 = df[col].quantile([.25, .75])
        
        iqr = q3 - q1
        
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df
#|-----------------------------------------------------------------------------------------------|#
def prepare_zillow(df):
    #just in case there are blanks
    df = df.replace(r'^\s*$', np.NaN, regex=True)

    # drop all nulls, for an affect of .00586 on data
    df.dropna(axis=0, how='any', inplace=True)

    # modify two columns
    df['fips'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
    df['yearbuilt'] = df['yearbuilt'].astype(int)
    df.yearbuilt = df.yearbuilt.astype(object)

    #create a new column named 'age', which is the difference of yearbuilt and 2017
    df['age'] = 2017-df['yearbuilt']

    df = df.rename(columns={
                        'calculatedfinishedsquarefeet': 'area',
                        'bathroomcnt': 'baths',
                        'bedroomcnt': 'beds',
                        'taxvaluedollarcnt':'tax_value',
                        'taxamount': 'tax_amount'}
              )
    # remove outliers.
    remove_outliers(df, 1.5, ['beds', 'baths', 'area', 'tax_value', 'tax_amount'])
    
    # create dummy vars of fips id
    dummy_df = pd.get_dummies(df.fips, drop_first=True)
    # rename columns by actual county name
    dummy_df.columns = ['Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, dummy_df], axis = 1)
    # drop fips columns
    df = df_dummies.drop(columns = ['fips'])
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test
#|-----------------------------------------------------------------------------------------------|#
def wrangled_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(acquire_zillow_data())
    
    return train, validate, test
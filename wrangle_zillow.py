from env import host, username, password, get_db_url
import os
import pandas as pd 
import numpy as np

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


def prepare_zillow(df):
    #just in case there are blanks
    df = df.replace(r'^\s*$', np.NaN, regex=True)

    # drop all nulls, for an affect of .00586 on data
    df.dropna(axis=0, how='any', inplace=True)

    # modify two columns
    df['fips'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
    df['yearbuilt'] = df['yearbuilt'].astype(int)

    #create a new column named 'age', which is the difference of yearbuilt and 2017
    df['age'] = 2017-df['yearbuilt']

    df = df.rename(columns={
                        'calculatedfinishedsquarefeet': 'sqft',
                        'bathroomcnt': 'baths',
                        'bedroomcnt': 'beds',
                        'taxvaluedollarcnt':'tax_value'}
              )

    return df


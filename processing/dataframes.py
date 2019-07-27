
import pandas as pd
import os

# returns unified DataFrame with path, id_code and diagnosis columns
def get_train_df ():
	train_df_new = get_train_df_new()
	train_df_old = get_train_df_old()
	return pd.concat([train_df_new, train_df_old])

def get_train_df_new ():
	train_df_new = pd.read_csv('data/new.csv')
	train_df_new['path'] = 'data/new/' + train_df_new.id_code + '.png'
	return train_df_new

def get_train_df_old ():
	train_df_old = pd.read_csv('data/old.csv')
	train_df_old['id_code'] = train_df_old.image
	train_df_old['diagnosis'] = train_df_old.level
	train_df_old['path'] = 'data/old/' + train_df_old.id_code + '.jpeg'
	train_df_old.drop(['Unnamed: 0', 'Unnamed: 0.1', 'image', 'level'],inplace=True,axis=1)
	train_df_old = train_df_old[[os.path.isfile(fname) for fname in train_df_old.path]]
	return train_df_old

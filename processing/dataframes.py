
import pandas as pd

# returns unified DataFrame with path, id_code and diagnosis columns
def get_train_df ():
	train_df_1 = pd.read_csv('data/new.csv')
	train_df_1['path'] = 'data/new/' + train_df_1.id_code + '.png'

	train_df_2 = pd.read_csv('data/old.csv')
	train_df_2['id_code'] = train_df_2.image
	train_df_2['diagnosis'] = train_df_2.level
	train_df_2['path'] = 'data/old/' + train_df_2.id_code + '.jpeg'
	train_df_2.drop(['Unnamed: 0', 'Unnamed: 0.1', 'image', 'level'],inplace=True,axis=1)
	train_df_2 = train_df_2[[os.path.isfile(fname) for fname in train_df_2.path]]

	return pd.concat([train_df_2, train_df_1])

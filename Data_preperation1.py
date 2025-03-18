import pandas as pd
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import numpy as np


#with zipfile.ZipFile(r'C:\Users\OMER\Downloads\archive.zip') as zip_:
    #zip_.extractall()

#Display all columns
pd.set_option('display.max_columns',None)





######################## Spotify ##############################
#with zipfile.ZipFile(r'C:\Users\OMER\Downloads\TMDB_tv_dataset_v3.csv.zip') as zip_:
    #zip_.extractall()

tv_db = pd.read_csv('TMDB_tv_dataset_v3.csv')

#print(spotify.head())
print(tv_db.info())

tv_db_copy = tv_db.copy()
#Removing columns with more than 55% nulls
tv_db_copy = tv_db_copy.loc[:,tv_db_copy.isnull().mean()<=0.55]

print(tv_db_copy.info())
missing_present = (tv_db_copy.isnull().sum() / len(tv_db_copy)) * 100
print(missing_present)
# Reamoving unrelevent collumns

unrealevent_colls = ['backdrop_path','poster_path',]

tv = tv_db_copy.drop(columns= unrealevent_colls)
print(tv.info())

tv[['name','original_name']] = tv[['name','original_name']].astype(str)
# Adding a column if the name of the show changed and removing the original_name column
tv['changed_name'] = tv.apply(lambda x: False if x['name']==x['original_name'] else True, axis=1)
tv = tv.drop(columns='original_name')

print(tv.info())


print(tv['origin_country'].nunique())

# handling large category columns
cols = ['origin_country','original_language']
def group_origin_con(df,cols):
    df_copy = df.copy()
    for col in cols:
# assigning 'other' for shows that has less than 1000 counts
        series_count = df_copy.value_counts(col).sort_values(ascending=False)
        series_lst = [val for val in series_count.index if series_count[val] <= 1000]
        df_copy[col] = df_copy[col].astype('string')
        df_copy[col] = df_copy[col].map(lambda x : 'other' if str(x) in series_lst else str(x))
#Converting to dummies after grouping values
        #df_copy[col] = df_copy[col].astype('category')
        #df_copy[col] = df_copy[col].cat.codes
    return df_copy
# Apply to the main DB:
tv = group_origin_con(tv,cols)

print(tv.info())
#print(tv['origin_country'].nunique())

def check_columns(): # check how many unique values in each column
    for col in tv.columns :
        print(f'{col} : {tv[col].nunique()} , {tv[col].dtypes}')

#bins =range(tv['number_of_episodes'].min(),tv['number_of_episodes'].max()+100, 800)
#bins2 =range(tv['number_of_seasons'].min(),tv['number_of_seasons'].max()+10, 10)
#plt.figure(figsize=(10, 5))
#tv_filtered = tv.loc[tv['number_of_episodes'] <= 100,: ]

#displaying the data
#sns.histplot(tv_filtered,x='number_of_seasons', kde = True,discrete=True)#,bins = (1,2,3,4,5))
#sns.distplot(tv_filtered,x='number_of_episodes', kde = True, bins=bins)#,discrete=True,)
#sns.displot(tv_filtered['number_of_episodes'], kde = True, bins = 5)#), bins=bins)
#sns.scatterplot(tv,x=tv['number_of_seasons'].index, y=tv['number_of_seasons'],alpha = 0.8)
#sns.boxplot(tv,x='popularity')
#plt.show()

print(tv['number_of_seasons'].quantile(0.5))
print(tv['number_of_episodes'].quantile(0.5))
print(tv['popularity'].quantile(0.99))

# showing correlation between numeric columns
cols = tv.select_dtypes(['int64', 'float64']).columns
tv_for_cor = tv[cols]
#print(tv_for_cor.corr())

print(tv.value_counts('original_language').sort_values(ascending=False).head(20))
# setting a columns for number of languges
##################################################
#before hundling large category columns and reducing tham i want to creat new columns based on the original columns 
tv_for_features = tv.copy()
lst = ['languages','spoken_languages','production_countries','networks']
def languages(df,lst):
    for col in lst :
        df[col] = df[col].astype('string')
        df[f'{col}_num'] = df[col].map(lambda x : len(str(x).split(",")) if len(str(x).split(",")) >1 and pd.notna(x) else (np.nan if pd.isna(x) else 1))
    #removing the 2 language cols
    df = df.drop(columns=lst)
    return df

test = 'asddsa'
n= test.split(",")
print(n)
t = test in n
print(t)

###############################
print('hello')
print(tv.value_counts('production_countries').sort_values(ascending=False).head(25).values.sum())
tv_for_features = languages(tv_for_features,lst)
print(tv_for_features.head())
pd.to_pickle(tv_for_features,'tv_for_features.pkl')

print(tv.info())

#print(tv.isna().sum())

# date time : convert to numericals
tv['first_air_date'] = pd.to_datetime(tv['first_air_date'], errors='coerce')
tv['last_air_date'] = pd.to_datetime(tv['last_air_date'], errors='coerce')
# Extract year and month
tv['year_start'] = tv['first_air_date'].dt.year
tv['month_start'] = tv['first_air_date'].dt.month


tv['year_end'] = tv['last_air_date'].dt.year
tv['month_end'] = tv['last_air_date'].dt.month

#print(tv[['first_air_date','last_air_date']])

#print(tv.info())

print(tv.value_counts('genres').sort_values(ascending=False).head(15))
above = (tv['popularity'] > 80).sum()
print(above)
col_to_check = tv.select_dtypes(['float64','int64','int8']).columns.tolist()
col_to_check.append('popularity')

# grop the genres x&y genres and y&x geners are the same:
def group_geners(genre):
    if pd.isna(genre):
        return genre
    generes = genre.split(", ")
    generes = sorted(generes)
    return '&'.join(generes)




tv['group_genere'] = tv['genres'].apply(group_geners)

tv_cop = tv.copy()

print(check_columns())
geners_count = tv.value_counts('group_genere').sort_values(ascending=False)
geners_count_other = geners_count[geners_count <= 100].index.tolist()
tv_cop['group_genere'] = tv_cop['group_genere'].astype('string')
tv_cop['group_genere'] = tv_cop['group_genere'].apply(lambda x : 'other' if pd.notna(x) and x in geners_count_other else x)
#print(tv_cop.value_counts('group_genere').sort_values(ascending=False).head(20))

tv['group_genere'] = tv_cop['group_genere']
########

print(tv.value_counts('type').sort_values(ascending=False).head(20))

tv['overview'] = tv['overview'].astype('string')


#handling large category production_contries

print(tv['production_countries'].head(20))
production_con_count = tv['production_countries'].value_counts().sort_values(ascending=False)

prod_count_to_other =production_con_count[production_con_count >= 400].index.tolist()

print(tv.value_counts('production_countries').sort_values(ascending=False).values.sum())
tv_cop['production_countries'] = tv_cop['production_countries'].astype('string')

def replace_low_count_countries(value):
    if pd.isna(value):  # Handle NaN values
        return value
    countries = value.split(',')  # Split if multiple countries exist
    updated_countries = [country if country in prod_count_to_other else 'Other' for country in countries]
    return ', '.join(updated_countries)  # Join back into a single string

# Apply the function
tv_cop['production_countries'] = tv_cop['production_countries'].apply(replace_low_count_countries)


print(tv_cop.value_counts('production_countries').sort_values(ascending=False).head(20))
#apply to the original DB:
tv['production_countries'] = tv_cop['production_countries']

# hundle networks column reduce the number of networks - convert sum to 'other'

net_counts = tv.value_counts('networks').sort_values(ascending=False)
net_counts_other = net_counts[net_counts >= 50].index.tolist()
print(net_counts)
tv_cop['networks'] = tv_cop['networks'].astype('string')
tv_cop['networks'] = tv_cop['networks'].apply(lambda x : 'other' if pd.notna(x) and x not in net_counts_other else x)
#print(tv_cop.value_counts('networks').sort_values(ascending=False).head(20))
tv['networks'] = tv_cop['networks']

print(tv.info())

# handle spoken_languages column:
spoken_lang_count = tv.value_counts('spoken_languages').sort_values(ascending=False)
#print(spoken_lang_count.head(20))
spoken_lang_count_other = spoken_lang_count[spoken_lang_count <= 250].index.tolist()
tv_cop['spoken_languages'] = tv_cop['spoken_languages'].astype('string')
tv_cop['spoken_languages'] = tv_cop['spoken_languages'].apply(lambda x : 'other' if pd.notna(x) and x in spoken_lang_count_other else x)
print(tv_cop.value_counts('spoken_languages').sort_values(ascending=False).head(20))



tv = tv.drop(columns='name')


#sns.heatmap(tv.corr(numeric_only=True))
#plt.show()


tv.to_pickle('tv_show.pkl')

import sys
print(sys.executable)



#tv_= pd.read_pickle('tv_show.pkl')


#import sys
#print(sys.version)

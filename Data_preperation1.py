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

########### handle overviews column ##########
tv['overview'] = tv['overview'].astype('string')
tv['overview'] = tv['overview'].apply(lambda x :x.lower() if pd.notna(x) else x) 


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
# production_countries column
print(tv['production_countries'].head(20))
production_con_count = tv['production_countries'].value_counts().sort_values(ascending=False)

#prod_count_to_other =production_con_count[production_con_count >= 400].index.tolist()
prod_count_to_other = production_con_count.head(22).index.tolist()

print(tv.value_counts('production_countries').sort_values(ascending=False).values.sum())
tv_cop['production_countries'] = tv_cop['production_countries'].astype('string')

def replace_low_count_countries(value):
    if pd.isna(value):  # Handle NaN values
        return value
    countries = value.split(',')  # Split if multiple countries exist
    if any(country in prod_count_to_other for country in countries):
        if len(countries) > 1:
            return 'multiple top_20'
        else:
            return value
    return 'Other'
    #updated_countries = [country if country in prod_count_to_other else 'Other' for country in countries]
    #return ', '.join(updated_countries)  # Join back into a single string

# Apply the function
tv_cop['production_countries'] = tv_cop['production_countries'].apply(replace_low_count_countries)


print(tv_cop.value_counts('production_countries').sort_values(ascending=False).head(20))
#apply to the original DB:
tv['production_countries'] = tv_cop['production_countries']

# hundle networks column reduce the number of networks - convert sum to 'other'

top_20_net = tv.value_counts('networks').sort_values(ascending=False).head(20)
net_counts = tv.value_counts('networks').sort_values(ascending=False)
#net_counts_other = net_counts[net_counts >= 100].index.tolist()
net_counts_other  = top_20_net.index.tolist()

def replace_low_count_networks(value):
    if pd.isna(value):  # Handle NaN values
        return value
    
    networks = value.split(', ')  # Split into list of networks
    
    # Check if at least one network is in net_counts_other
    if any(net in net_counts_other for net in networks):
        return value  # Keep original value if at least one network is valid
    
    # Otherwise, replace all with 'Other'
    return 'Other'

# function to creat a replace column , I am adding a column for now because after i would like to creat a new 
# column based on networks that specified the number of networks that participated in the production

def replace_with_dominant(value):
    if pd.isna(value):
        return value
    networks = value.lower().split(', ')
    if len(networks) >1 and any(net in top_20_net.index for net in networks):
            return 'coop top_20'
    #for net in networks:
     #   if net in top_20_net.index:
            #return net
        #continue
    else:
        return value.lower()
        



#print(net_counts)
tv_cop['networks'] = tv_cop['networks'].astype('string')
tv_cop['networks'] = tv_cop['networks'].apply(replace_low_count_networks)
#apllying
tv_cop['_networks_'] = tv_cop['networks'].apply(replace_with_dominant)
#Apply to the original DB 
#tv['networks'] = tv_cop['_networks_']
print(tv_cop['_networks_'].nunique())
print('before the change')
print(tv_cop['networks'].nunique())
print(tv_cop.value_counts('networks').sort_values(ascending=False).head(20))
print('_networks_')
print(tv_cop.value_counts('_networks_').sort_values(ascending=False).head(20))
tv['networks'] = tv_cop['_networks_']

print(tv.info())

# handle spoken_languages column:
spoken_lang_count = tv.value_counts('spoken_languages').sort_values(ascending=False)
#print(spoken_lang_count.head(20))
#spoken_lang_count_other = spoken_lang_count[spoken_lang_count >= 250].index.tolist()
spoken_lang_count_other = spoken_lang_count.head(20).index.tolist()

def replace_low_count_spoken(value):
    if pd.isna(value):  # Handle NaN values
        return value
    
    spoken = value.split(', ')  # Split into list of spoken languages
    
    # Check if at least one language is in net_counts_other
    if any(spok in spoken_lang_count_other for spok in spoken):
        if len(spoken) > 1:
            return 'multiple top_20'
        else:
            return value # Keep original value if at least one language is valid
            
    # Otherwise, replace all with 'Other'
    return 'Other'

tv_cop['spoken_languages'] = tv_cop['spoken_languages'].astype('string')
tv_cop['spoken_languages'] = tv_cop['spoken_languages'].apply(replace_low_count_spoken)
print(tv_cop.value_counts('spoken_languages').sort_values(ascending=False).head(20))
#print(tv_cop['spoken_languages'].nunique())
# Apply to the original DB:

tv['spoken_languages'] = tv_cop['spoken_languages'] 

#######
lan_count =tv.value_counts('languages').sort_values(ascending=False)
#lan_count_other = lan_count[lan_count >= 350].index.tolist()
lan_count_other = lan_count.head(20).index.tolist()

def replace_low_count_lang(value):
    if pd.isna(value):  # Handle NaN values
        return value
    
    lang = value.split(', ')  # Split into list of spoken languages
    
    # Check if at least one language is in net_counts_other
    if any(lan in lan_count_other for lan in lang):
        if len(lang) > 1:
            return 'multiple top_20'
        else:
            return value  # Keep original value if at least one language is valid
    
    # Otherwise, replace all with 'Other'
    return 'Other'


print(tv['languages'].nunique())
tv['languages'] = tv['languages'].apply(replace_low_count_lang)
lang_count =tv.value_counts('languages').sort_values(ascending=False)
print(lang_count.head(20))
print(tv['languages'].nunique())



tv = tv.drop(columns='name')


#sns.heatmap(tv.corr(numeric_only=True))
#plt.show()


tv.to_pickle('tv_show.pkl')

import sys
print(sys.executable)



#tv_= pd.read_pickle('tv_show.pkl')


#import sys
#print(sys.version)

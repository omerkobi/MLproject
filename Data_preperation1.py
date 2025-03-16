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


#costumer = pd.read_csv('customer_details.csv',index_col=1)
#product = pd.read_csv('product_details.csv',index_col=1)
#ecommerce =pd.read_csv('E-commerece sales data 2024.csv',index_col=1)

#print(costumer.info())
#print(product.info())
#print(ecommerce.info())

#merge_e_p = pd.merge(ecommerce,product,left_on='product id', right_on='Uniqe Id')
#merge = pd.merge(merge_e_p,costumer, left_on='user id',right_on='Customer ID')
#print(merge.info())
#print(merge_e_p.info())

#print(merge.corr())
#merge_no_nan = merge.loc[:, merge.isnull().mean()<=0.6]


#print(merge_no_nan.head(5))
#print(merge_no_nan.info())
#print(merge_no_nan[['Selling Price','Purchase Amount (USD)']].sort_values(['Purchase Amount (USD)','Selling Price'],ascending=False).head(10))
############################### option 2 ###########################

#with zipfile.ZipFile(r'C:\Users\OMER\Downloads\athletes.csv.zip') as zip_cross :
#    zip_cross.extractall()

#cross = pd.read_csv('athletes.csv')
#print(cross.info())
### clean more than 60% nulls :
#cross = cross.loc[:,cross.isnull().mean() <= 0.6]
# remove rows with more than n nulls :
#cross_after = cross[cross.isnull().sum(axis=1)> 5]

#cross_cleaned = cross[cross.isnull().sum(axis=1)<= 7]

#print(cross_after.info())
#print(cross_cleaned.info())
#flo_col = [col for col in cross_cleaned.select_dtypes("float64")]
#print(cross_cleaned[flo_col].corr())


######################## Spotify ##############################
with zipfile.ZipFile(r'C:\Users\OMER\Downloads\TMDB_tv_dataset_v3.csv.zip') as zip_:
    zip_.extractall()

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
        df_copy[col] = df_copy[col].astype('category')
        df_copy[col] = df_copy[col].cat.codes
    return df_copy

tv = group_origin_con(tv,cols)
print(tv.info())
print(tv['origin_country'].nunique())

def check_columns():
    for col in tv.columns :
        print(f'{col} : {tv[col].nunique()} , {tv[col].dtypes}')

bins =range(tv['number_of_episodes'].min(),tv['number_of_episodes'].max()+100, 800)
bins2 =range(tv['number_of_seasons'].min(),tv['number_of_seasons'].max()+10, 10)
plt.figure(figsize=(10, 5))
tv_filtered = tv.loc[tv['number_of_episodes'] <= 100,: ]

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
lst = ['languages','spoken_languages','production_countries']
def languages(df,lst):
    for col in lst :
        df[col] = df[col].astype('string')
        df[f'{col}_num'] = df[col].map(lambda x : len(str(x).split(",")) if len(str(x).split(",")) >1 and pd.notna(x) else (np.nan if pd.isna(x) else 1))
    #removing the 2 language cols
    df = df.drop(columns=lst)
    return df

###############################
tv = languages(tv,lst)
print(tv.info())

print(tv.isna().sum())

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
#chack on jupyter
#sns.pairplot(tv, vars=col_to_check)
#plt.show()

def group_geners(genre):
    if pd.isna(genre):
        return genre
    generes = genre.split(", ")
    generes = sorted(generes)
    return '&'.join(generes)

tv_cop = tv.copy()

tv['group_genere'] = tv['genres'].apply(group_geners)
print(check_columns())
print(tv.value_counts('group_genere').sort_values(ascending=False).head(20))

print(tv.value_counts('type').sort_values(ascending=False).head(20))

tv['overview'] = tv['overview'].astype('string')
print(tv.info())


tv = tv.drop(columns='name')

#print(tv[['year_end','year_start']])
sns.heatmap(tv.corr(numeric_only=True))
plt.show()


tv.to_pickle('tv_show.pkl')

import sys
print(sys.executable)



#tv_= pd.read_pickle('tv_show.pkl')


#import sys
#print(sys.version)

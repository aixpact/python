import numpy as np
import pandas as pd

df
df = df.set_index([df.index, 'Name'])
df.index.names = ['Location', 'Name']
df.append({'Item Purchased': 'Kitty Food', 'Cost': 3.00}, name=['Store 2', 'Kevyn'])

# Hierarchical indexing
# Adding data to df
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])


df = df.set_index([df.index, 'Name'])
df.index.names = ['Location', 'Name']
df = df.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, name=('Store 2', 'Kevyn')))
df

#### HW2

import pandas as pd

df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index)
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
df.head()

def answer_two():
    return abs(df['Gold']-df['Gold.1']).argmax()
answer_two()

df.set_index("Gold").sort_index()
df.set_index("Gold.1").sort_index()

df['diff'] = abs(df['Gold'] - df['Gold.1'])
df.set_index("diff").sort_index()

(abs(df[df['Gold']>0]-df[df['Gold.1']>0])/df['Gold.2']).argmax()
masks = df['Gold']>0
maskw = df['Gold.1']>0
df[masks & maskw]
df2 = df[(df['Gold']>0) & (df['Gold.1']>0)]
df3 = abs(df2['Gold'] - df2['Gold.1'])/df2['Gold.2']
df3.argmax()

dfc = df[(df['Gold']>0) & (df['Gold.1']>0)]
dfc = abs(dfc['Gold'] - dfc['Gold.1'])/dfc['Gold.2']
dfc.argmax()
df.head()

df['Points'] = df['Gold.1']*3 + df['Silver.1']*2 + df['Bronze.1']
df['Points'].head(146)

### Part 2
census_df = pd.read_csv('census.csv')
census_df['STNAME'].head(70)

census_df[census_df['SUMLEV'] == 40][['STNAME', 'CTYNAME']]

census_dfc = census_df[census_df['SUMLEV'] == 50]
census_dfc.set_index(['STNAME', 'CTYNAME']).head()
census_dfc['count'] = pd.groupby(by=census_dfc['STNAME'], level=census_dfc['CTYNAME'], agg=sum)

census_df[census_df['SUMLEV'] == 50].groupby(['STNAME'])['CTYNAME'].count().idxmax()

(census_df[census_df['SUMLEV'] == 40]
    .groupby(['STNAME'])['CENSUS2010POP']
    .sum()
    .sort_values(ascending=False)
    .head(3))

census_df['Max'] = census_df[max(census_df.iloc[:, 9:15])]
census_df['Min'] = census_df[min(census_df.iloc[:, 9:15])]
census_df['Growth'] = census_df['Max'] - census_df['Min']

census_df[census_df['SUMLEV'] == 40].groupby(['CTYNAME'])['Growth'].max().idxmax()
# census_df[census_df['CTYNAME'] == 'Texas']


# Create a query that finds the counties that belong to regions 1 or 2,
# whose name starts with 'Washington',
#     and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.
census_df['REGION'].unique()
dfc = (census_df[(census_df['SUMLEV'] == 50)
          & (census_df['REGION'] < 3)
          & (census_df['CTYNAME'].str.contains('Washington'))
          & (census_df['POPESTIMATE2015'] > census_df['POPESTIMATE2014']) ]
)
dfc.loc[:, ('STNAME', 'CTYNAME')]

len('Washington')

def answer_seven():
    dfc = census_df.iloc[:, [0, 5, 6, 9, 10, 11, 12, 13, 14]][census_df['SUMLEV'] == 40]
    max = dfc.loc[:, 'POPESTIMATE2010':].max(axis=1)
    min = dfc.iloc[:, 9:15].min(axis=1)
    dfc.loc['Growth'] = max - min
    return dfc[:].groupby(['CTYNAME'])['Growth'].max().idxmax()
answer_seven()



def answer_six():
    dfc = (census_df[census_df['SUMLEV'] == 50]
            .sort_values(['STNAME', 'CENSUS2010POP'], ascending=[1, 0])
            .groupby(['STNAME'])['STNAME', 'CTYNAME', 'CENSUS2010POP'].head(3)
            .groupby(['STNAME'])['CENSUS2010POP'].sum()
            .sort_values(ascending=False).head(3))
    return list(dfc.index)
answer_six()

def answer_six():
    dfc = (census_df[census_df['SUMLEV'] == 40]
            .groupby(['STNAME'])['CENSUS2010POP']
            .sum()
            .sort_values(ascending=False)
            .head(3))
    return list(dfc.index)

print(df.drop(df[df['Quantity'] == 0].index).rename(columns={'Weight': 'Weight (oz.)'}))

# Sub total and other aggragate functions
print(df.groupby('Category').apply(lambda df,a,b: sum(df[a] * df[b]), 'Weight (oz.)', 'Quantity'))
#or
def totalweight(df, w, q):
       return sum(df[w] * df[q])

print(df.groupby('Category').apply(totalweight, 'Weight (oz.)', 'Quantity'))
# or
df['Total'] = df['Quantity']*df['Weight (oz.)']
print(df.groupby('Category').agg({'Total': 'sum'}))

s = pd.Series(['Low', 'Low', 'High', 'Medium', 'Low', 'High', 'Low'])

s.astype('category', categories=['Low', 'Medium', 'High'], ordered=True)  # Ordinal

#
s = pd.Series([168, 180, 174, 190, 170, 185, 179, 181, 175, 169, 182, 177, 180, 171])
pd.cut(s, 3, labels=['Small', 'Medium', 'Large'])


print(pd.pivot_table(Bikes, index=['Manufacturer','Bike Type']))

###
# Question 1 (20%)
# Load the energy data from the file Energy Indicators.xls,
# which is a list of indicators of energy supply and renewable electricity production from the United Nations for the year 2013,
# and should be put into a DataFrame with the variable name of energy.
# Keep in mind that this is an Excel file, and not a comma separated values file.
# Also, make sure to exclude the footer and header information from the datafile.
# The first two columns are unneccessary, so you should get rid of them,
energy = pd.read_excel('Energy Indicators.xls', skiprows=16, skip_footer=37, na_values=np.NaN).iloc[1:, 2:]
# energy.head(-1)
# and you should change the column labels so that the columns are:
# ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
# For all countries which have missing data (e.g. data with "...") make sure this is reflected as np.NaN values.
energy[energy.iloc[:, :] == '...'] = np.NaN
# Convert Energy Supply to gigajoules (there are 1,000,000 gigajoules in a petajoule).
energy['Energy Supply'] *= 1000000

# There are also several countries with numbers and/or parenthesis in their name. Be sure to remove these,
# e.g.
# 'Bolivia (Plurinational State of)' should be 'Bolivia',
# 'Switzerland17' should be 'Switzerland'.
energy['Country'] = energy['Country'].str.replace('\d+', '')
energy['Country'] = energy['Country'].str.replace('\(.+\)', '')
energy['Country'] = energy['Country'].str.strip()
energy.columns
# Rename the following list of countries (for use in later questions):
# "Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"
ctry = {"Republic of Korea": "South Korea",
"United States of America": "United States",
"United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
"China, Hong Kong Special Administrative Region": "Hong Kong"}
energy['Country'].replace(ctry, inplace=True)


# Next, load the GDP data from the file world_bank.csv, which is a csv containing countries' GDP from 1960 to 2015 from World Bank. Call this DataFrame GDP.
# Make sure to skip the header, and rename the following list of countries:
# "Korea, Rep.": "South Korea",
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"
#
#!cat world_bank.csv

GDP = pd.read_csv('world_bank.csv', skiprows=4)
# GDP.head(-1)
GDP.rename(columns={'Country Name': 'Country'}, inplace=True)


GDP.columns

GDP['Country'] = GDP['Country'].str.replace('\d+', '')
GDP['Country'] = GDP['Country'].str.replace('\(.+\)', '')
GDP['Country'] = GDP['Country'].str.strip()
ctry = {
        "Korea, Rep.": "South Korea",
        "Iran, Islamic Rep.": "Iran",
        "Hong Kong SAR, China": "Hong Kong"
        }
GDP['Country'].replace(ctry, inplace=True)
# Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15).
GDP.drop(GDP.columns[[range(1, 50)]], axis=1, inplace=True)
#GDP.drop(GDP.columns[[1, 2, 3]], axis=1, inplace=True)
GDP.columns


#!cat scimagojr-3.xlsx
# Finally, load the Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology from the file scimagojr-3.xlsx,
# which ranks countries based on their journal contributions in the aforementioned area. Call this DataFrame ScimEn.

scimEn = pd.read_excel('scimagojr-3.xlsx', na_values=np.NaN)
scimEn.head(-1)

dfi = pd.merge(scimEn, GDP, how='inner', left_on='Country', right_on='Country')
dfi = pd.merge(energy, dfi, how='inner', left_on='Country', right_on='Country')
dfi.shape  # (162, 21)
dfo = pd.merge(scimEn, GDP, how='outer', left_on='Country', right_on='Country')
dfo = pd.merge(energy, dfi, how='outer', left_on='Country', right_on='Country')
dfo.shape  # (228, 24)
dfo.shape[0] - dfi.shape[0]


# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names).
df = pd.merge(scimEn[scimEn['Rank']<16], GDP, how='inner', left_on='Country', right_on='Country')
df = pd.merge(energy, df, how='inner', left_on='Country', right_on='Country')

# The index of this DataFrame should be the name of the country,
df = df.set_index('Country')
df.columns
df.shape  # (15, 20)
df.head()

# and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations',
# 'Self-citations', 'Citations per document', 'H index', 'Energy Supply',
# 'Energy Supply per Capita', '% Renewable',
# '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
# This function should return a DataFrame with 20 columns and 15 entries.

# What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)
# This function should return a Series named avgGDP with 15 countries and their average GDP sorted in descending order.
avgGDP = df.T[-10:].mean().T
avgGDP

#
avg = df.T[-10:].mean().T
avgGDP = df.T[-10:].T
avgGDP['Avg'] = avg
dfc = avgGDP.sort_values(['Avg'], axis=0, ascending=False)
growth = dfc.iloc[5, -2] - dfc.iloc[5, 0]
growth

#
df.iloc[:, 1].mean()

# What country has the maximum % Renewable and what is the percentage?
df.loc[df.loc[:, '% Renewable'].fillna(0).idxmax(), '% Renewable']



def answer_one():
    #
    energy = pd.read_excel('Energy Indicators.xls', skiprows=16, skip_footer=37, na_values=np.NaN).iloc[1:, 2:]
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy[energy.iloc[:, :] == '...'] = np.NaN
    #energy = energy.infer_objects()  # change type from general object to more specific eg. 'float' or 'int'
    numeric_cols = ['Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy[numeric_cols] = energy[numeric_cols].astype('float64', errors='ignore')
    #energy.dtypes
    energy['Energy Supply'] *= 1000000
    energy['Country'] = energy['Country'].str.replace('\d+', '')
    energy['Country'] = energy['Country'].str.replace('\(.+\)', '')
    energy['Country'] = energy['Country'].str.strip()
    ctry = {
        "Republic of Korea": "South Korea",
        "United States of America": "United States",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "China, Hong Kong Special Administrative Region": "Hong Kong"
    }
    energy['Country'].replace(ctry, inplace=True)

    #
    GDP = pd.read_csv('world_bank.csv', skiprows=4)
    GDP.rename(columns={'Country Name': 'Country'}, inplace=True)
    GDP['Country'] = GDP['Country'].str.replace('\d+', '')
    GDP['Country'] = GDP['Country'].str.replace('\(.+\)', '')
    GDP['Country'] = GDP['Country'].str.strip()
    ctry = {
        "Korea, Rep.": "South Korea",
        "Iran, Islamic Rep.": "Iran",
        "Hong Kong SAR, China": "Hong Kong"
    }
    GDP['Country'].replace(ctry, inplace=True)
    GDP.drop(GDP.columns[[range(1, 50)]], axis=1, inplace=True)

    #
    scimEn = pd.read_excel('scimagojr-3.xlsx', na_values=np.NaN)

    # Join the three datasets
    df = pd.merge(scimEn[scimEn['Rank'] < 16], GDP, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(energy, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df = df.infer_objects()
    return df
answer_one()

def answer_two():
    dfi = pd.merge(scimEn, GDP, how='inner', left_on='Country', right_on='Country')
    dfi = pd.merge(energy, dfi, how='inner', left_on='Country', right_on='Country')
    dfi.shape  # (162, 21)
    dfo = pd.merge(scimEn, GDP, how='outer', left_on='Country', right_on='Country')
    dfo = pd.merge(energy, dfi, how='outer', left_on='Country', right_on='Country')
    dfo.shape  # (162, 21)
    return dfo[0] - dfi[0]
answer_two()


def answer_three():
    Top15 = answer_one()
    avgGDP = Top15.T[-10:].mean().T
    return avgGDP.sort_values(ascending=False)
answer_three()

def answer_four():
    Top15 = answer_one()
    avg = answer_three()  # df.T[-10:].mean().T
    avgGDP = Top15.T[-10:].T
    avgGDP['Avg'] = avg
    dfc = avgGDP.sort_values(['Avg'], axis=0, ascending=False)
    growth = dfc.iloc[5, -2] - dfc.iloc[5, 0]
    return growth
answer_four()

def answer_five():
    Top15 = answer_one()
    return Top15.iloc[:, 1].mean()
answer_five()

def answer_six():
    Top15 = answer_one()
    idx = Top15.loc[:, '% Renewable'].fillna(0).idxmax()
    max = Top15.loc[idx, '% Renewable']
    return (idx, max)
answer_six()

# Question 7 (6.6%)
# Create a new column that is the ratio of Self-Citations to Total Citations.
# What is the maximum value for this new column, and what country has the highest ratio?
# This function should return a tuple with the name of the country and the ratio.

df['Ratio'] = df['Self-citations'] / df['Citations']
ctry = df[['Self-citations', 'Citations', 'Ratio']].sort_values('Ratio', ascending=False).index[0]
ratio = df.loc[ctry, 'Ratio']
(ctry, ratio)


def answer_seven():
    Top15 = answer_one()
    Top15['Ratio'] = Top15['Self-citations'] / df['Citations']
    ctry = Top15[['Self-citations', 'Citations', 'Ratio']].sort_values('Ratio', ascending=False).index[0]
    ratio = df.loc[ctry, 'Ratio']
    return (ctry, ratio)
answer_seven()

# Question 8 (6.6%)
# Create a column that estimates the population using Energy Supply and Energy Supply per capita.
# What is the third most populous country according to this estimate?
# This function should return a single string value.
df.columns

Top15 = df
Top15['Population'] = Top15['Energy Supply'] / df['Energy Supply per Capita']
Top15[['Population']].sort_values('Population', ascending=False).index[0]

def answer_eight():
    Top15 = answer_one()
    Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    ctry = Top15[['Population']].sort_values('Population', ascending=False).index[2]
    return ctry
answer_eight()

#Question 9 (6.6%)
# Create a column that estimates the number of citable documents per person.
# What is the correlation between the number of citable documents per capita and
# the energy supply per capita? Use the .corr() method, (Pearson's correlation).
# This function should return a single number.
# (Optional: Use the built-in function plot9() to visualize the relationship
# between Energy Supply per Capita vs. Citable docs per Capita)

Top15 = df.infer_objects()
Top15['Population'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita'])
Top15['Citations per capita'] = (Top15['Citations'] / Top15['Population'])
corr_coef = Top15[['Citations per capita', 'Energy Supply per Capita']].corr().iloc[0, 1]
corr_coef

def answer_nine():
    Top15 = answer_one()
    Top15['Population'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita'])
    Top15['Citations per capita'] = (Top15['Citations'] / Top15['Population'])
    corr_coef = Top15[['Citations per capita', 'Energy Supply per Capita']].corr().iloc[0, 1]
    return corr_coef
answer_nine()
answer_one().columns
def answer_nine():
    Top15 = answer_one()
    Top15['PopEst'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita'])
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    corr_coef = Top15[['Citable docs per Capita', 'Energy Supply per Capita']].corr().iloc[0, 1]
    return corr_coef
answer_nine()

# Question 10 (6.6%)
# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15,
# and a 0 if the country's % Renewable value is below the median.
# This function should return a series named HighRenew whose index is the country name sorted in ascending order of rank.
Top15.columns

def answer_ten():
    Top15 = answer_one()
    median = Top15['% Renewable'].median()
    Top15['High Renewable'] = 0
    Top15['High Renewable'] = Top15['% Renewable'] >= median  # True=1
    HighRenew = Top15[['High Renewable', 'Rank']].sort_values('Rank').astype('i').iloc[:, 0]
    HighRenew
    return HighRenew
answer_ten()

# Question 11 (6.6%)
# Use the following dictionary to group the Countries by Continent,
# then create a dateframe that displays the sample size (the number of countries in each continent bin),
# and the sum, mean, and std deviation for the estimated population of each country.
ContinentDict  = {'China':'Asia',
                  'United States':'North America',
                  'Japan':'Asia',
                  'United Kingdom':'Europe',
                  'Russian Federation':'Europe',
                  'Canada':'North America',
                  'Germany':'Europe',
                  'India':'Asia',
                  'France':'Europe',
                  'South Korea':'Asia',
                  'Italy':'Europe',
                  'Spain':'Europe',
                  'Iran':'Asia',
                  'Australia':'Australia',
                  'Brazil':'South America'}
# This function should return a DataFrame with index named Continent
# ['Asia', 'Australia', 'Europe', 'North America', 'South America'] and columns ['size', 'sum', 'mean', 'std']

df = df.infer_objects()
Top15 = df
Top15['Population'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita'])
Top15['Country'] = Top15.index
Top15['Continent'] = Top15.index
Top15['Continent'].replace(ContinentDict, inplace=True)
Top15['Continent']
Top15 = Top15.set_index('Continent')
Top15 = Top15.loc[:, ['Country', 'Population']]

# Top15 = Top15.groupby(Top15.index)['Country', 'Population']
Top15['size'] = Top15.groupby(Top15.index)['Population'].count()
Top15['sum'] = Top15.groupby(Top15.index)['Population'].sum()
Top15['mean'] = Top15.groupby(Top15.index)['Population'].mean()
Top15['std'] = Top15.groupby(Top15.index)['Population'].std()
Top15 = Top15.drop(['Country', 'Population'], axis=1)
Top15 = Top15.drop_duplicates()
Top15


def answer_eleven():
    Top15 = answer_one()
    Top15['Population'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita'])
    Top15['Country'] = Top15.index
    Top15['Continent'] = Top15.index
    Top15['Continent'].replace(ContinentDict, inplace=True)
    Top15['Continent']
    Top15 = Top15.set_index('Continent')
    Top15 = Top15.loc[:, ['Country', 'Population']]

    Top15['size'] = Top15.groupby(Top15.index)['Population'].count()
    Top15['sum'] = Top15.groupby(Top15.index)['Population'].sum()
    Top15['mean'] = Top15.groupby(Top15.index)['Population'].mean()
    Top15['std'] = Top15.groupby(Top15.index)['Population'].std()
    # Top15 = Top15.drop(columns=['Country', 'Population'])
    # Top15 = Top15.drop_duplicates()
    return Top15.drop(columns=['Country', 'Population']).drop_duplicates()
answer_eleven()

# Question 12 (6.6%)
# Cut % Renewable into 5 bins.
# Group Top15 by the Continent, as well as these new % Renewable bins.
# How many countries are in each of these groups?
# This function should return a Series with a MultiIndex of Continent, then the bins for % Renewable.
# Do not include groups with no countries.
Top15 = df.infer_objects()
Top15['% Renewable'] = pd.cut(Top15['% Renewable'], 5)
Top15['Country'] = Top15.index
Top15['Continent'] = Top15.index
Top15['Continent'].replace(ContinentDict, inplace=True)
Top15 = Top15.loc[:, ['Continent', 'Country', '% Renewable']]
Top15 = Top15.set_index(['Continent', '% Renewable'])
Top15 = Top15.groupby(Top15.index)['Country'].count()
Top15
type(Top15)

def answer_twelve():
    Top15 = answer_one()
    Top15 = df.infer_objects()
    Top15['% Renewable'] = pd.cut(Top15['% Renewable'], 5)
    Top15['Country'] = Top15.index
    Top15['Continent'] = Top15.index
    Top15['Continent'].replace(ContinentDict, inplace=True)
    Top15 = Top15.loc[:, ['Continent', 'Country', '% Renewable']]
    Top15 = Top15.set_index(['Continent', '% Renewable'])
    Top15 = Top15.groupby(Top15.index)['Country'].count()
    return Top15
answer_twelve()


# Question 13 (6.6%)
# Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
# e.g. 317615384.61538464 -> 317,615,384.61538464
# This function should return a Series PopEst whose index is the country name and whose values are the population estimate string.
df = df.infer_objects()
Top15 = df
Top15['PopEst'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita'])
Top15['PopEst'].apply('{0:,f}'.format)

def answer_thirteen():
    Top15 = answer_one()
    Top15['PopEst'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita'])
    # Top15['PopEst'] = Top15['PopEst'].apply('{0:,f}'.format)
    Top15['PopEst'] = Top15['PopEst'].astype(str)
    return Top15['PopEst']
answer_thirteen()

def answer_thirteen2():
    Top15 = answer_one()
    Top15['PopEst'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita'])
    return Top15['PopEst'].apply('{0:,}'.format).astype(str)
answer_thirteen2()


#### HW!.4

#
df_univ_towns = pd.read_csv('university_towns.txt', sep='\n', header=None)
df_univ_towns.head()
df_univ_towns['raw'] = df_univ_towns.iloc[:, 0]
df_univ_towns['State'] = df_univ_towns['raw']
df_univ_towns['RegionName'] = df_univ_towns['State']

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming',
          'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon',
          'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont',
          'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin',
          'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi',
          'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota',
          'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia',
          'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska',
          'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania',
          'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands',
          'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
statesId = {v:k for k, v in states.items()}
df_univ_towns['State ID'] = df_univ_towns['State'].replace(statesId)

#
mask_state = df_univ_towns['State'].str.contains('edit')
mask_region = (mask_state == False)  # reverse boolean

# State
df_univ_towns['State'] = (df_univ_towns['State']
                          .str.replace('\[edit\]', '')
                          .str.strip() * mask_state)
df_univ_towns['State'][mask_region] = np.nan
df_univ_towns['State'].fillna(method='ffill', inplace=True)

# Region
df_univ_towns['RegionName'] = (df_univ_towns['RegionName']
                               .str.replace(r'\(.+?\)', '')
                               .str.replace(r'\[.+\]', '')
                               .str.replace(r'University\s.*\,\s', '')
                               # .str.replace(r'[:;]', '')
                               .str.strip() * mask_region)
df_univ_towns = df_univ_towns[mask_region]
df_univ_towns = df_univ_towns[['State', 'RegionName']]


## ZHVI Home
df_zhvi = pd.read_csv('City_Zhvi_AllHomes.csv')
df_zhvi.head(-1)
df_zhvi.drop(df_zhvi.columns[0], axis=1, inplace=True)
df_zhvi.drop(df_zhvi.columns[2:50], axis=1, inplace=True)

def col_batch(start, seq, step):
    return (seq.iloc[:, pos:pos+step] for pos in range(start, len(seq.columns), step))

for i, cols in enumerate(col_batch(2, df_zhvi, 3)):
    quarter = '{}q{}'.format(i//4+2000, i%4+1)
    df_zhvi[quarter] = cols.mean(axis=1)

mask_cols = df_zhvi.columns.str.extract('(.+-.+)', expand=False).fillna(False) == False
df_zhvi.columns[mask_cols]
df_zhvi = df_zhvi.iloc[:, mask_cols]
df_zhvi['State'] =  df_zhvi['State'].replace(states)
df_zhvi.set_index(['State', 'RegionName'], inplace=True)


##
df_gdplev = pd.read_excel('gdplev.xls', skiprows=5)
df_gdplev_annual = df_gdplev.iloc[3:89, :3]  # annual values are the average quarter values / this df can be neglected
df_gdp = df_gdplev.iloc[212:, [4, 6]]  # start with 2 quarters earlier to use for shift
df_gdp.columns = ['Quarter', 'Chained GDP']
df_gdp['Rolling Min'] = df_gdp['Chained GDP'].rolling(window=2).min()
df_gdp = df_gdp.iloc[2:, :]

# A recession is defined as starting with two consecutive quarters of GDP decline,
# and ending with two consecutive quarters of GDP growth.
# A recession bottom is the quarter within a recession which had the lowest GDP.

# Start: chained < -1 < -2
# End: chained > -1 > -2
df_gdp['Recession'] = ((df_gdp['Chained GDP'] < df_gdp['Chained GDP'].shift(1)) &
                       (df_gdp['Chained GDP'].shift(1) < df_gdp['Chained GDP'].shift(2)))
df_gdp['Start'] = (df_gdp['Recession'].shift(1) == False) & (df_gdp['Recession'] == True)
df_gdp['End'] = (df_gdp['Recession'].shift(-1) == False) & (df_gdp['Recession'] == True)
start_recession = df_gdp[df_gdp['Start']]['Quarter'].values[0]
end_recession = df_gdp[df_gdp['End']]['Quarter'].values[0]

# Bottom:
df_gdp['Bottom'] = ((df_gdp['Rolling Min'].shift(-1) == df_gdp['Rolling Min']) &
                    (df_gdp['Recession']))
bottom_recession = df_gdp[df_gdp['Bottom']]['Quarter'].values[0]

# Last quarter before recession
df_gdp['Quarter Before'] = (df_gdp['Recession'].shift(-1)) & (df_gdp['Recession'] == False)
last_quarter = df_gdp[df_gdp['Quarter Before']]['Quarter'].values[0]

print(start_recession, end_recession, bottom_recession, last_quarter)


# Hypothesis: University towns have their mean housing prices less effected by recessions.
# Run a t-test to compare: (price_ratio=quarter_before_recession/recession_bottom)
# the mean price of houses in university towns the quarter before the recession starts
# compared to the recession bottom.
#

df_zhvi_uni = pd.merge(df_univ_towns, df_zhvi, how='outer', left_on=['State', 'RegionName'], right_index=True)

# Sanity test if merged set has same result for mean
# ratio_all = df_zhvi_uni[last_quarter].mean().values/df_zhvi_uni[bottom_recession].mean().values
# ratio = df_zhvi[last_quarter].mean().values/df_zhvi[bottom_recession].mean().values
# assert ratio_all - ratio < 10E-8 , 'wtf'

df_zhvi_uni['Ratio'] = df_zhvi_uni.loc[:, last_quarter]/df_zhvi_uni.loc[:, bottom_recession]

# Non University Towns
mask_null_uni = pd.isnull(df_zhvi_uni['State ID'])        # mask non University states
mask_null_ratio = pd.isnull(df_zhvi_uni['Ratio'])      # mask NaN ratios
mask_null = ~mask_null_uni | mask_null_ratio
ratio_non_uni = df_zhvi_uni.loc[~mask_null, 'Ratio']   # create ratio

# University Towns
mask_null_uni = pd.isnull(df_zhvi_uni['State ID'])        # mask non University states
mask_null_ratio = pd.isnull(df_zhvi_uni['Ratio'])      # mask NaN ratios
mask_null = mask_null_uni | mask_null_ratio
ratio_uni = df_zhvi_uni.loc[~mask_null, 'Ratio']       # create ratio

from scipy.stats import ttest_ind
p = ttest_ind(ratio_uni, ratio_non_uni)[1]
different = p < 0.01  #
better = ["university town", "non-university town"][ratio_uni.mean()<ratio_non_uni.mean()]
print(different, p, better)


# (df_zhvi_uni.columns == '2000q1').argmax()



'''Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''






# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming',
          'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon',
          'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont',
          'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin',
          'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi',
          'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota',
          'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut',
          'WV': 'West Virginia',
          'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska',
          'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania',
          'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands',
          'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}

##

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ],
    columns=["State", "RegionName"]  )

    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''

    df_univ_towns = pd.read_csv('university_towns.txt', sep='\n', header=None)
    df_univ_towns['raw'] = df_univ_towns.iloc[:, 0]
    df_univ_towns['State'] = df_univ_towns['raw']
    df_univ_towns['RegionName'] = df_univ_towns['State']

    #
    mask_state = df_univ_towns['State'].str.contains('edit')
    mask_region = (mask_state == False)  # reverse boolean

    # State
    df_univ_towns['State'] = (df_univ_towns['State']
                              .str.replace('\[edit\]', '')
                              .str.strip() * mask_state)
    df_univ_towns['State'][mask_region] = np.nan
    df_univ_towns['State'].fillna(method='ffill', inplace=True)

    # Region
    df_univ_towns['RegionName'] = (df_univ_towns['RegionName']
                                   # .str.replace(r'\(.*?\)', '')
                                   .str.replace(r'\(.*', '')
                                   # .str.replace(r'\[.*\]', '')
                                   # .str.replace(r'University\s.*', '')
                                   # .str.replace(',.*', '')
                                   # .str.replace('^The.*', '')
                                   .str.strip() * mask_region)
    df_univ_towns = df_univ_towns[mask_region]
    df_univ_towns = df_univ_towns[df_univ_towns['RegionName'] != '']
    df_univ_towns = df_univ_towns[['State', 'RegionName']]
    return df_univ_towns

towns = get_list_of_university_towns()


def get_recession_start():
    '''Returns the year and quarter of the recession start time as a
    string value in a format such as 2005q3'''

    df_gdplev = pd.read_excel('gdplev.xls', skiprows=5)
    df_gdp = df_gdplev.iloc[212:, [4, 6]]  # start with 2 quarters earlier to use for shift
    df_gdp.columns = ['Quarter', 'Chained GDP']
    df_gdp['Rolling Min'] = df_gdp['Chained GDP'].rolling(window=2).min()
    df_gdp = df_gdp.iloc[2:, :]

    #
    df_gdp['Recession'] = ((df_gdp['Chained GDP'] < df_gdp['Chained GDP'].shift(1)) &
                           (df_gdp['Chained GDP'].shift(1) < df_gdp['Chained GDP'].shift(2)))
    df_gdp['Start'] = (df_gdp['Recession'].shift(1) == False) & (df_gdp['Recession'])
    start_recession = df_gdp[df_gdp['Start']]['Quarter'].values[0]

    return start_recession


get_recession_start()


def get_recession_end():
    '''Returns the year and quarter of the recession end time as a
    string value in a format such as 2005q3'''

    df_gdplev = pd.read_excel('gdplev.xls', skiprows=5)
    df_gdplev_annual = df_gdplev.iloc[3:89,
                       :3]  # annual values are the average quarter values / this df can be neglected
    df_gdp = df_gdplev.iloc[212:, [4, 6]]  # start with 2 quarters earlier to use for shift
    df_gdp.columns = ['Quarter', 'Chained GDP']
    df_gdp['Rolling Min'] = df_gdp['Chained GDP'].rolling(window=2).min()
    df_gdp = df_gdp.iloc[2:, :]

    #
    df_gdp['Recession'] = ((df_gdp['Chained GDP'] < df_gdp['Chained GDP'].shift(1)) &
                           (df_gdp['Chained GDP'].shift(1) < df_gdp['Chained GDP'].shift(2)))
    df_gdp['End'] = (df_gdp['Recession'].shift(-1) == False) & (df_gdp['Recession'])
    end_recession = df_gdp[df_gdp['End']]['Quarter'].values[0]

    return end_recession


get_recession_end()


def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a
    string value in a format such as 2005q3'''

    df_gdplev = pd.read_excel('gdplev.xls', skiprows=5)
    df_gdp = df_gdplev.iloc[212:, [4, 6]]  # start with 2 quarters earlier to use for shift
    df_gdp.columns = ['Quarter', 'Chained GDP']
    df_gdp['Rolling Min'] = df_gdp['Chained GDP'].rolling(window=2).min()
    df_gdp = df_gdp.iloc[2:, :]

    #
    df_gdp['Recession'] = ((df_gdp['Chained GDP'] < df_gdp['Chained GDP'].shift(1)) &
                           (df_gdp['Chained GDP'].shift(1) < df_gdp['Chained GDP'].shift(2)))

    # Bottom:
    df_gdp['Bottom'] = ((df_gdp['Rolling Min'].shift(-1) == df_gdp['Rolling Min']) &
                        (df_gdp['Recession']))
    bottom_recession = df_gdp[df_gdp['Bottom']]['Quarter'].values[0]

    return bottom_recession


get_recession_bottom()


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].

    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.

    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    df_zhvi = pd.read_csv('City_Zhvi_AllHomes.csv')
    df_zhvi.drop(df_zhvi.columns[0], axis=1, inplace=True)
    df_zhvi.drop(df_zhvi.columns[2:50], axis=1, inplace=True)

    def col_batch(start, seq, step):
        return (seq.iloc[:, pos:pos + step] for pos in range(start, len(seq.columns), step))

    for i, cols in enumerate(col_batch(2, df_zhvi, 3)):
        quarter = '{}q{}'.format(i // 4 + 2000, i % 4 + 1)
        df_zhvi[quarter] = cols.mean(axis=1)

    mask_cols = df_zhvi.columns.str.extract('(.+-.+)', expand=False).fillna(False) == False
    df_zhvi.columns[mask_cols]
    df_zhvi = df_zhvi.iloc[:, mask_cols]

    df_zhvi['State'] =  df_zhvi['State'].replace(states)
    df_zhvi.set_index(['State', 'RegionName'], inplace=True)

    return df_zhvi


convert_housing_data_to_quarters()
convert_housing_data_to_quarters().loc["Texas"].loc["Austin"].loc["2010q3"]



def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values,
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence.

    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''

    df_univ_towns = get_list_of_university_towns()

    # statesId = {v: k for k, v in states.items()}
    # df_univ_towns['State ID'] = df_univ_towns['State'].replace(statesId)

    df_housing = convert_housing_data_to_quarters()

    df_zhvi_uni = pd.merge(df_univ_towns,
                           df_housing,
                           how='outer',
                           left_on=['State', 'RegionName'],
                           right_index=True)

    df_zhvi_uni['Ratio'] = df_zhvi_uni.loc[:, last_quarter] / df_zhvi_uni.loc[:, bottom_recession]

    # Non University Towns
    mask_null_uni = pd.isnull(df_zhvi_uni['State'])  # mask non University states
    mask_null_ratio = pd.isnull(df_zhvi_uni['Ratio'])  # mask NaN ratios
    mask_null = ~mask_null_uni | mask_null_ratio
    ratio_non_uni = df_zhvi_uni.loc[~mask_null, 'Ratio']  # create ratio

    # University Towns
    mask_null_uni = pd.isnull(df_zhvi_uni['State'])  # mask non University states
    mask_null_ratio = pd.isnull(df_zhvi_uni['Ratio'])  # mask NaN ratios
    mask_null = mask_null_uni | mask_null_ratio
    ratio_uni = df_zhvi_uni.loc[~mask_null, 'Ratio']  # create ratio

    # from scipy.stats import ttest_ind
    p = ttest_ind(ratio_uni, ratio_non_uni)[1]
    different = p < 0.01  #
    better = ["university town", "non-university town"][ratio_uni.mean() > ratio_non_uni.mean()]

    return (different, p, better)


run_ttest()

####
def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values,
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence.

    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''

    df_housing = convert_housing_data_to_quarters()
    df_univ_towns = get_list_of_university_towns()
    df_univ_towns['University'] = True  # Splitting rule

    # Merge and split df
    df_merge = pd.merge(df_univ_towns, df_housing, how='outer', left_on=['State', 'RegionName'], right_index=True)
    df_univ = df_merge[df_merge['University'] == True]
    df_nonu = df_merge[df_merge['University'] != True]

    df_merge.sort_values(by=['State', 'RegionName'], inplace=True)

    #
    bottom_recession = get_recession_bottom()
    start_recession = get_recession_start()
    last_quarter = df_univ.columns[[ix-1 for ix, c in enumerate(df_univ.columns) if c == start_recession]][0]

    # Ratio distributions
    ratio_univ = df_univ.loc[:, last_quarter] / df_univ.loc[:, bottom_recession]
    df_univ['Ratio'] = ratio_univ
    ratio_nonu = df_nonu.loc[:, last_quarter] / df_nonu.loc[:, bottom_recession]
    df_nonu['Ratio'] = ratio_nonu

    # from scipy.stats import ttest_ind
    p = ttest_ind(ratio_univ, ratio_nonu, nan_policy='omit')[1]
    different = p < 0.01
    better = ["university town", "non-university town"][ratio_univ.mean() > ratio_nonu.mean()]

    return different, p, better  #, ratio_univ.mean(), ratio_nonu.mean(), last_quarter, start_recession, bottom_recession


run_ttest()


#### Course 2
import matplotlib.pyplot as plt
import numpy as np

plt.figure()

languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

plt.bar(pos, popularity, align='center')
# plt.xticks(pos, languages)
plt.ylabel('% Popularity')
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.show()

#
plt.figure()

languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]
# TODO: change the bar colors to be less bright blue
# TODO: make one bar, the python bar, a contrasting color
plt.bar(pos, popularity, align='center')

# TODO: soften all labels by turning grey
plt.xticks(pos, languages)
plt.ylabel('% Popularity')
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow')

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()

#
plt.figure()

languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

# change the bar colors to be less bright blue
bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')
# make one bar, the python bar, a contrasting color
bars[0].set_color('#1F77B4')

# soften all labels by turning grey
plt.xticks(pos, languages, alpha=0.8)
plt.ylabel('% Popularity', alpha=0.8)
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()

#
plt.figure()

languages = ['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

# change the bar colors to be less bright blue
bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')
# make one bar, the python bar, a contrasting color
bars[0].set_color('#1F77B4')

# soften all labels by turning grey
plt.xticks(pos, languages, alpha=0.8)

# TODO: remove the Y label since bars are directly labeled
plt.ylabel('% Popularity', alpha=0.8)
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# TODO: direct label each bar with Y axis values
plt.show()

#
plt.figure()

languages = ['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

# change the bar color to be less bright blue
bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')
# make one bar, the python bar, a contrasting color
bars[0].set_color('#1F77B4')

# soften all labels by turning grey
plt.xticks(pos, languages, alpha=0.8)
# remove the Y label since bars are directly labeled
# plt.ylabel('% Popularity', alpha=0.8)
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# direct label each bar with Y axis values
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, str(int(bar.get_height())) + '%',
                   ha='center', color='w', fontsize=11)
plt.show()
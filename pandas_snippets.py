import numpy as np
import pandas as pd

# List unique values in a DataFrame column
# h/t @makmanalp for the updated syntax!

df = pd.DataFrame()  # TODO

df['Column Name'].unique()

# Convert Series datatype to numeric (will error if column has non-numeric values)
# h/t @makmanalp
pd.to_numeric(df['Column Name'])

# Convert Series datatype to numeric, changing non-numeric values to NaN
# h/t @makmanalp for the updated syntax!
pd.to_numeric(df['Column Name'], errors='coerce')

# Grab DataFrame rows where column has certain values
valuelist = ['value1', 'value2', 'value3']
df = df[df.column.isin(valuelist)]

# Grab DataFrame rows where column doesn't have certain values
valuelist = ['value1', 'value2', 'value3']
df = df[~df.column.isin(valuelist)]

# Delete column from DataFrame
del df['column']

# Select from DataFrame using criteria from multiple columns
# (use `|` instead of `&` to do an OR)
newdf = df[(df['column_one']>2004) & (df['column_two']==9)]

# Rename several DataFrame columns
df = df.rename(columns = {
    'col1 old name':'col1 new name',
    'col2 old name':'col2 new name',
    'col3 old name':'col3 new name',
})

# Lower-case all DataFrame column names
df.columns = map(str.lower, df.columns)

# Even more fancy DataFrame column re-naming
# lower-case all DataFrame column names (for example)
df.rename(columns=lambda x: x.split('.')[-1], inplace=True)

# Loop through rows in a DataFrame
# (if you must)
for index, row in df.iterrows():
    print(index, row['some column'])

# Much faster way to loop through DataFrame rows
# if you can work with tuples
# (h/t hughamacmullaniv)
for row in df.itertuples():
    print(row)

# Next few examples show how to work with text data in Pandas.
# Full list of .str functions: http://pandas.pydata.org/pandas-docs/stable/text.html

# Slice values in a DataFrame column (aka Series)
df.column.str[0:2]

# Lower-case everything in a DataFrame column
df.column_name = df.column_name.str.lower()

# Get length of data in a DataFrame column
df.column_name.str.len()

# Sort dataframe by multiple columns
df = df.sort(['col1','col2','col3'],ascending=[1,1,0])

# Get top n for each group of columns in a sorted dataframe
# (make sure dataframe is sorted first)
top5 = df.groupby(['groupingcol1', 'groupingcol2']).head(5)

# Grab DataFrame rows where specific column is null/notnull
newdf = df[df['column'].isnull()]

# Select from DataFrame using multiple keys of a hierarchical index
df.xs(('index level 1 value','index level 2 value'), level=('level 1','level 2'))

# Change all NaNs to None (useful before
# loading to a db)
df = df.where((pd.notnull(df)), None)

# More pre-db insert cleanup...make a pass through the dataframe, stripping whitespace
# from strings and changing any empty values to None
# (not especially recommended but including here b/c I had to do this in real life one time)
df = df.applymap(lambda x: str(x).strip() if len(str(x).strip()) else None)

# Get quick count of rows in a DataFrame
len(df.index)

# Pivot data (with flexibility about what what
# becomes a column and what stays a row).
# Syntax works on Pandas >= .14
pd.pivot_table(
  df,values='cell_value',
  index=['col1', 'col2', 'col3'], #these stay as columns; will fail silently if any of these cols have null values
  columns=['col4']) #data values in this column become their own column

# Change data type of DataFrame column
df.column_name = df.column_name.astype(np.int64)

# Get rid of non-numeric values throughout a DataFrame:
refunds = df
for col in refunds.columns.values:
  refunds[col] = refunds[col].replace('[^0-9]+.-', '', regex=True)

# Set DataFrame column values based on other column values (h/t: @mlevkov)
some_value, some_other_value, new_value = 0, 99, 999
df.loc[(df['column1'] == some_value) & (df['column2'] == some_other_value), ['column_to_change']] = new_value

# Clean up missing values in multiple DataFrame columns
df = df.fillna({
    'col1': 'missing',
    'col2': '99.999',
    'col3': '999',
    'col4': 'missing',
    'col5': 'missing',
    'col6': '99'
})

# Concatenate two DataFrame columns into a new, single column
# (useful when dealing with composite keys, for example)
# (h/t @makmanalp for improving this one!)
df['newcol'] = df['col1'].astype(str) + df['col2'].astype(str)

# Doing calculations with DataFrame columns that have missing values
# In example below, swap in 0 for df['col1'] cells that contain null
df['new_col'] = np.where(pd.isnull(df['col1']),0,df['col1']) + df['col2']

# Split delimited values in a DataFrame column into two new columns
df['new_col1'], df['new_col2'] = zip(*df['original_col'].apply(lambda x: x.split(': ', 1)))

# Collapse hierarchical column indexes
df.columns = df.columns.get_level_values(0)

# Convert Django queryset to DataFrame
DjangoModelName = None # TODO
qs = DjangoModelName.objects.all()
q = qs.values()
df = pd.DataFrame.from_records(q)

# Create a DataFrame from a Python dictionary
a_dictionary = {} # TODO
df = pd.DataFrame(list(a_dictionary.items()), columns = ['column1', 'column2'])

# Get a report of all duplicate records in a dataframe, based on specific columns
dupes = df[df.duplicated(['col1', 'col2', 'col3'], keep=False)]

# Set up formatting so larger numbers aren't displayed in scientific notation (h/t @thecapacity)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# ---------------------------------------------> List, Dict, Set snippets <-------------------------------------------- #

from collections import Counter, OrderedDict
import operator

d = {'c': 2, 'b': 5, 'a': 2, 'e': 5, 'd': 4, 'f': 2}
d1 = {('a', 2): 2, ('b', 1): 3, ('d', 4): 1, ('c', 5): 5, ('f', 6): 4, ('e', 1): 5}

# Key or value in dictionary
'a' in d
5 in d.values()

# get keys from listed tuples
[i for i in d]                    # get keys
[i for i in d.values()]           # get values
[i for i in d.values()].index(5)  # get index from list
[i for i in d.values()].count(5)  # get count from list

# Create tuples from dict
Counter(d).items()
Counter(d.values())

# Dict to list of tuples and back to dict
tups = Counter(d).items(); tups
{k: v for k, v in sorted(tups)}

# Dictionary to pd.Series, DataFrame
pd.Series(d)
pd.DataFrame(list(d.items()), columns = ['idx', 'value']).set_index('idx')

# Reverse tuples
[(v, k) for k, v in d.items()]

# Reverse dict - deletes duplicate keys!
{v:k for k, v in d.items()}

# Sort by keys
sorted(d, key=operator.itemgetter(0), reverse = True)[:5]  # returns keys
sorted(d.items(), key=lambda i: i[0], reverse = True)[:5]  # returns tuples(k, v)

# Sort by values
[i[0] for i in sorted(d.items(), key=lambda i: i[1], reverse = True)[:5]]  # returns keys
sorted(d.items(), key=lambda i: i[1], reverse = True)[:5]  # returns tuples(k, v)
Counter(d).most_common()                                   # returns tuples(k, v)

# Sort tuple by index
sorted(d1, key=operator.itemgetter(0), reverse = True)[:5]  # returns keys()
sorted(d1, key=operator.itemgetter(1), reverse = True)[:5]  # returns keys()

# Sort tuple by value
Counter(d).most_common()
Counter(d1).most_common()

# Frequencies
[(i, list(d.values()).count(i)) for i in set(d.values())]
Counter(d.values()).most_common()  # returns
Counter(d.values())                # returns counter-dict

# Find frequency/count of arbitrary value (eg. 5)
x = 5
Counter(d.values())[x]
len([k for k, v in d.items() if v == x])

# Find max frequency in list of dicts
# max(frequency of values == diameter per node)
d01 = {'c': 2, 'b': 5, 'a': 2, 'e': 5, 'd': 2, 'f': 2}
d02 = {'c': 2, 'b': 5, 'a': 2, 'e': 5, 'd': 5, 'f': 5}
d03 = {'c': 2, 'b': 2, 'a': 2, 'e': 5, 'd': 4, 'f': 2}
nodes = [d01, d02, d03]
x = 5
max(pd.DataFrame([[node, Counter(node.values())[x]] for node in nodes]).iloc[:, 1])
[(a, b, c) for a, b, c in zip(d01, d02, d03)] # if t == 1]
[[a, b, c] for a, b, c in zip(d01.values(), d02.values(), d03.values()) if c == 2]

#
dictA = {
    'a' : ('duck','duck','goose'),
    'b' : ('goose','goose'),
    'c' : ('duck','duck','duck'),
    'd' : ('goose'),
    'e' : ('duck','duck')
    }
[k for (k, v) in dictA.items() if v.count('duck') > 1]
[i for i in dictA if Counter(dictA[i])['duck'] > 1]



# Distribution
d.values().count_values()
np.bincount(pd.Series(d))/np.bincount(pd.Series(d)).sum()


# Create keyvalue frequency list
Counter(sorted(d.items(), key=lambda t: t[1]))

# Outer - permutations
[(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]

list(zip([1, 2, 3], [3, 1, 4]))


# Flatten a list using a listcomp with two 'for'
vec = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
[num for elem in vec for num in elem]

# Transpose 1
matrix3 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]]
[[row[i] for row in matrix3] for i in range(4)]
list(zip(matrix3[0], matrix3[1], matrix3[2]))
list(zip(*matrix3))

# Transpose 2
matrix2 = [[1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16]]
list(zip(matrix2[0], matrix2[1]))
list(zip(*matrix2))

# Set comprehension
{x for x in 'abracadabra' if x not in 'abc'}





def top_cat(df_, feature, top=10):
    """Replace top 10 most frequent labels with 0-9 and rest with 10"""
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    labels = alphabet[:top]
    other = alphabet[top + 1]
    top_violation_codes = df_.groupby(feature)[feature].count().sort_values(ascending=False).head(
        top).index.tolist()
    map_values = {k: l for k, l in (zip(top_violation_codes, labels))}  # [::-1]
    key_others = set(map_values.keys()) ^ (set(df_.loc[:, feature].values))
    map_others = {k: other for k in key_others}
    map_all = {**map_others, **map_values}
    df_.loc[:, feature] = df_.loc[:, feature].replace(map_all).astype('category')
    return df_

# Default value
a = None or 1
a

# Merge lists
x = [1, 2, 3]
y = [4, 5]
x.append(y); x  # [1, 2, 3, [4, 5]] - nested
x.extend(y); x  # [1, 2, 3, 4, 5] - flat

# Merge dictionaries
x = {'a':1, 'b': 2}
y = {'c':10, 'd': 11}
x.update(y)  # {'a': 1, 'b': 2, 'c': 10, 'd': 11}
z = x.copy()
z.update(y)  # same but copy

# Dict constructor
keys, values = None, None  # TODO
{key: value for (key, value) in zip(keys, values)}



# coding: utf-8
import numpy as np
import pandas as pd
import mplleaflet
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
# from matplotlib.ticker import FixedLocator, LinearLocator, FormatStrFormatter
# import datetime


# Import data
df_GHCN = pd.read_csv('fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')
df_GHCN.head()

# Mask leap day
mask_leap = df_GHCN.loc[:, 'Date'].str.endswith('02-29', na=False)
df_GHCN = df_GHCN[~mask_leap]

# Create filters
is_2015 = df_GHCN.loc[:, 'Date'].str.startswith('2015', na=False)
is_tmin = df_GHCN.loc[:, 'Element'] == 'TMIN'
is_2015_tmin = is_2015 & is_tmin
is_2015_tmax = is_2015 & ~is_tmin
is_2005_tmin = ~is_2015 & is_tmin
is_2005_tmax = ~is_2015 & ~is_tmin

# Select datasets for plot
tmin_2005 = df_GHCN.loc[is_2005_tmin].groupby(df_GHCN.loc[:, 'Date'].str.replace(r'^\d*-', ''))['Data_Value'].min().reset_index()
tmax_2005 = df_GHCN.loc[is_2005_tmax].groupby(df_GHCN.loc[:, 'Date'].str.replace(r'^\d*-', ''))['Data_Value'].max().reset_index()
tmin_2015 = df_GHCN.loc[is_2015_tmin].groupby(df_GHCN.loc[:, 'Date'].str.replace(r'^\d*-', ''))['Data_Value'].min().reset_index()
tmax_2015 = df_GHCN.loc[is_2015_tmax].groupby(df_GHCN.loc[:, 'Date'].str.replace(r'^\d*-', ''))['Data_Value'].max().reset_index()

# Select outside bandwidth
tlow_2015 = tmin_2015[tmin_2015['Data_Value'] < tmin_2005['Data_Value']]
thigh_2015 = tmax_2015[tmax_2015['Data_Value'] > tmax_2005['Data_Value']]


# Setup plot
plt.rcdefaults()  # restore plot defaults
plt.figure(figsize=(20, 6))
fig, ax = plt.gcf(), plt.gca()
rc('mathtext', default='regular')

# Line plots record low and high
plt.plot(tmin_2005.index, tmin_2005.loc[:, 'Data_Value'], '-', c='black', linewidth=.7, label='Extremes 2005-2014')
plt.plot(tmax_2005.index, tmax_2005.loc[:, 'Data_Value'], '-', c='black', linewidth=.7, label='')

# Fill the area between the High and Low temperatures
plt.gca().fill_between(tmin_2005.index, tmin_2005.loc[:, 'Data_Value'], tmax_2005.loc[:, 'Data_Value'],
                       facecolor='gray', alpha=0.25)

# Scatter outside bandwidth
plt.scatter(tlow_2015.index, tlow_2015.loc[:, 'Data_Value'], c='red', s=25, marker = 'o', alpha=.6, label='Extremes 2015')
plt.scatter(thigh_2015.index, thigh_2015.loc[:, 'Data_Value'], c='red', s=25, marker = 'o', alpha=.6, label='')

# Title, axis, labels, legend
plt.title(r'2015 Temperature Extremes - Outside 2005-2014' +
          '\n(Ann Arbor, Michigan, United States)')
plt.suptitle('')
ax.legend(loc=4, frameon=False)

# yticks
ylim = ax.get_ylim()
y_ticks = np.arange((ylim[0]//50 - 1)*50, (ylim[1]//50 + 1)*50, 100)
temperatures = [r'{:3.0f}$^\circ$C'.format(t//10) for t in y_ticks]
plt.yticks(y_ticks, temperatures, rotation=0)

# xticks
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x_ticks = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
plt.xticks(x_ticks, months, ha='left')

# Hide axis spines
[ax.spines[side].set_visible(False) for side in ['top', 'left', 'bottom', 'right']]

plt.show()




#### Alternative Long version ##########

# Convert long DF to wide DF; TMIN and TMAX to columns
df_GHCN_max = df_GHCN[df_GHCN.loc[:, 'Element'] == 'TMAX']
df_GHCN_min = df_GHCN[df_GHCN.loc[:, 'Element'] == 'TMIN']
# df_GHCN_max.head(50)


df_GHCN = pd.merge(df_GHCN_min, df_GHCN_max, how='outer', left_on=['ID','Date'], right_on=['ID','Date'])
df_GHCN.columns = ['ID', 'Date', 'x', 'TMIN', 'y', 'TMAX']
df_GHCN.drop(['x', 'y'], axis=1, inplace=True)
# df_GHCN.head(10)


# Convert dates
# observation_dates = np.arange('2017-01-01', '2017-01-09', dtype='datetime64[D]')
df_GHCN.loc[:, 'Date'] = list(map(pd.to_datetime, df_GHCN.loc[:, 'Date']))
df_GHCN.loc[:, 'Day Month'] = df_GHCN.loc[:, 'Date'].dt.strftime('%d/%b')
df_GHCN.loc[:, 'Day of Year'] = df_GHCN.loc[:, 'Date'].dt.dayofyear
df_GHCN.loc[:, 'Day'] = df_GHCN.loc[:, 'Date'].dt.day
df_GHCN.loc[:, 'Month'] = df_GHCN.loc[:, 'Date'].dt.month
df_GHCN.loc[:, 'Year'] = df_GHCN.loc[:, 'Date'].dt.year
# df_GHCN.dtypes

# Sort df on Date
df_GHCN.sort_values(['Date', 'ID'], inplace=True)


# Mask leap day
mask_leap = (df_GHCN.loc[:, 'Month'] == 2) & (df_GHCN.loc[:, 'Day'] == 29)
df_GHCN = df_GHCN[~mask_leap]


# Split df
df_2005 = df_GHCN[df_GHCN.loc[:, 'Year'] < 2015]
df_2015 = df_GHCN[df_GHCN.loc[:, 'Year'] == 2015]


# Daily records
daily_records_2005 = df_2005.groupby(['Day Month']).agg({'TMIN':min, 'TMAX':max})
daily_records_2015 = df_2015.groupby(['Day Month']).agg({'TMIN':min, 'TMAX':max})
lower = daily_records_2015.loc[:, 'TMIN'] < daily_records_2005.loc[:, 'TMIN']
higher = daily_records_2015.loc[:, 'TMAX'] > daily_records_2005.loc[:, 'TMAX']


# Merge
df_2015 = pd.merge(df_2015, daily_records_2005, how='left', left_on=['Day Month'], right_index=True)
df_2015.columns = ['ID', 'Date', 'TMIN', 'TMAX', 'Day Month', 'Day of Year', 'Day', 'Month', 'Year', 'Low', 'High']
df_2015.head(-1)


# Plot data
df_plot = df_2015.groupby('Day of Year').agg({'TMIN':min, 'TMAX':max, 'Low':min, 'High':max})
df_scatter_min = df_plot[df_plot.loc[:, 'TMIN'] < df_plot.loc[:, 'Low']]
df_scatter_max = df_plot[df_plot.loc[:, 'TMAX'] > df_plot.loc[:, 'High']]


# Plot init
plt.rcdefaults()  # restore plot defaults
rc('mathtext', default='regular')
mpl.rcParams['figure.titlesize'] = 'Large'

plt.figure(figsize=(20, 6))
fig, ax = plt.gcf(), plt.gca()  # or # fig, ax = plt.subplots()

# Line plots record low and high
plt.plot(df_plot.index.values, df_plot.loc[:, 'Low'], '-', c='black', linewidth=.7, label='Extremes 2005-2014')
plt.plot(df_plot.index.values, df_plot.loc[:, 'High'], '-', c='black', linewidth=.7, label='')

# Title, axis, labels
plt.xlabel('')
plt.ylabel('')
plt.title(r'Temperature extremes ($^\circ$C) in 2015' +
          ' (Ann Arbor, Michigan, United States)\n' +
          'Against temperature range in 2005-2014\n')

# Fill the area between the High and Low temperatures
plt.gca().fill_between(df_plot.index.values,
                       df_plot.loc[:, 'Low'], df_plot.loc[:, 'High'],
                       facecolor='gray',
                       alpha=0.25)
plt.scatter(df_scatter_min.index, df_scatter_min.loc[:, 'TMIN'], c='red', s=25, marker = 'o', alpha=.6, label='Extremes 2015')
plt.scatter(df_scatter_max.index, df_scatter_max.loc[:, 'TMAX'], c='red', s=25, marker = 'o', alpha=.6, label='')

# Legend
ax.legend(loc=4, frameon=False)

# yticks
min_temp = (df_plot.loc[:, 'Low'].min()//50 - 1)*50
max_temp = (df_plot.loc[:, 'High'].max()//50 + 1)*50
temperatures = np.arange(min_temp, max_temp, 100)
ylabel = [r'{:3.0f}$^\circ$C'.format(t//10) for t in temperatures]
plt.yticks(temperatures, ylabel, rotation=0)

# xticks
months = [month[:3] for month in calendar.month_name[1:13]]
plt.xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365], months, rotation=0)
plt.xticks(ha='left')

# plt.axis('off') no axis, no ticks
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Coordinates formatting (hover)
myFmt = DateFormatter("%d/%b")
ax.xaxis.set_major_formatter(myFmt)
ax.xaxis.set_major_locator(mdates.MonthLocator())

plt.show()



#######

# # Create ranges 0-12
# np.linspace(-200, 200, 9, endpoint=True)
# np.arange(12)
# range(1, len(df_plot.index))



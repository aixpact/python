import pandas as pd
import numpy as np
import re

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head()


def date_sorter():
    # months_abbr = pd.Series(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # months_name = pd.Series(
    #     ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
    #      'December'])
    # vals = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    # months_dict = dict(zip(months_name, vals))
    # months_dict.update(dict(zip(months_abbr, vals)))
    #
    # regx_month = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, months_name)))
    # re.compile(r'(?:%s)' % r'|'.join(map(re.escape, months_name)))

    # for i, text in enumerate(df):
    #     print(regx_month.findall(df[i]))
    #     df[i] = regx_month.findall(df[i])
    #
    # # Replace all words in dictionary by values
    # def multiple_replace(text, lookup, adict=None):
    #     regx = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, lookup)))
    #
    #     def dict_replace(match):
    #         return adict[match.group(0)]
    #
    #     return regx.sub(dict_replace, text)
    #
    # for i, text in enumerate(df):
    #     print(multiple_replace(text, lookup=months_name , adict=months_dict))
    #     df[i] = multiple_replace(text, lookup=months_abbr, adict=months_dict)

    # df_dates2 = df.str.extractall(r'(?P<month>\d?\d?\d?\d?\b' + \
    #                               r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)' + \
    #                               r'[\s\.,\-/]+?\d?\d?[\s\.,\-/]+?\d?\d?\d?\d?)' + \
    #                               r'[\s\.,\-/]+?\d?\d?[\s\.,\-/]+?\d?\d?\d?\d?)'

                                       # '[\s\.,\-/]+?\d\d\d?\d?[\s\.,\-/]+?(?:19|20)\d?\d?)|' + \
                                       # '[\s\.,\-/]+?\d\d\d?\d?[\s\.,\-/]+?(?:19|20)\d?\d?)|' + \

    # df_dates = df_dates.str.extractall(r'[\s\.,\-/]*?(?P<month>(?:January|February|March|April|May|June|July|August|September|October|November|December|' + \
    #                                    'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\.,\-/]+(?:19|20)\d\d)|' + \
    #                                    r'[\s\.,\-/]*?(?P<month2>(?:January|February|March|April|May|June|July|August|September|October|November|December|' + \
    #                                    'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\.,\-/]+\d\d[\s\.,\-/]+?(?:19|20)?\d\d)|' + \
    #                                    r'(?P<mmddyyyy>[0-3]?\d[\-/]+[0-3]?\d[\-/]+(?:19|20)\d\d)|' + \
    #                                    r'(?P<mmddyy>[0-3]?\d[\-/]+[0-3]?\d[\-/]+\d\d)|' + \
    #                                    r'(?P<mmyyyy>[0-1]?\d[\-/]+(?:19|20)\d\d)|' + \
    #                                    r'(?P<year>(?:19|20)\d\d)')


    # Extract dates
    df_dates = df.str.replace(r'(\d+\.\d+)', '')
    # df_dates = df.str.replace(r'(\d+\.\d+)', '')
    #

    df_dates = df_dates.str.extractall(r'[\s\.,\-/]*?(?P<ddmonthyyyy>\d\d[\s\.,\-/]+(?:January|February|March|April|May|June|July|August|September|October|November|December|' + \
                                       'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\.,\-/]+(?:19|20)\d\d)|' + \
                                       r'[\s\.,\-/]*?(?P<monthddyyyy>(?:Jan.*\b|February|March|April|May|June|July|August|September|October|November|Dec.*\b|' + \
                                       'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\.,\-/]+\d\d[\s\.,\-/]+?(?:19|20)?\d\d)|' + \
                                       r'[\s\.,\-/]*?(?P<monthyyyy>(?:January|February|March|April|May|June|July|August|September|October|November|December|' + \
                                       'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\.,\-/]+(?:19|20)\d\d)|' + \
                                       r'(?P<mmddyyyy>[0-3]?\d[\-/]+[0-3]?\d[\-/]+(?:19|20)\d\d)|' + \
                                       r'(?P<mmddyy>[0-3]?\d[\-/]+[0-3]?\d[\-/]+\d\d)|' + \
                                       r'(?P<mmyyyy>[0-1]?\d[\-/]+(?:19|20)\d\d)|' + \
                                       r'(?P<year>(?:19|20)\d\d)')  # Ignore case (?i)
    # Munge dates
    print(pd.isnull(df_dates['year']).sum())
    df_dates = df_dates.fillna('')
    df_dates = df_dates.sum(axis=1).apply(pd.to_datetime)
    df_dates = df_dates.reset_index()
    df_dates.columns = ['index', 'match', 'dates']


    # Sort dates
    df_dates.sort_values(by='dates', inplace=True)
    result = df_dates.loc[:, 'index'].astype('int32')

    # Unit test & Sanity check
    assert result.shape[0] == 500
    assert result[0].dtype == 'int32'

    return result

print(type(date_sorter()), date_sorter().shape)

df[342]
df[313]
df_dates.iloc[125, :]
df_dates.iloc[313, :]
df[196]
df[335]
df[228]
df[455]
df_dates.iloc[228, :]
df[323]

sep = r'[\.\s,-/]+'
reg_date_us = r'yyyy|mm/yyyy|mmm/yyyy|mmmm/yyyy|mm/dd/yyyy|mm/dd/yy|mmm/dd/yy|mmmm/dd/yy|mmm/ddth/yy|mmmm/ddth/yy'
reg_date_eu ='dd/mmm/yyyy|dd/mmmm/yyyy'

############# notes

dates = '''04/20/2009 some text here:
04/20/09 some text here:
4/20/09 some text here:
4/3/09 some text here:
Mar-20-2009 some text here:
Mar 20, 2009 some text here:
March 20, 2009 some text here:
Mar. 20, 2009 some text here: 
Mar 20 2009 some text here:
20 Mar 2009 some text here: 
20 March 2009 some text here:
20 Mar. 2009 some text here: 
20 March, 2009 some text here:
Mar 20th, 2009 some text here:
Mar 21st, 2009 some text here:
Mar 22nd, 2009 some text here:
Feb 2009 some text here: 
Sep 2009 some text here: 
Oct 2010 some text here:
6/2008 some text here:
12/2009 some text here:
2009 some text here:
2010 some text here:'''
dates = dates.split('\n')#.strip()
df = pd.Series(dates)
df.head()
df.shape

months_abbr = pd.Series(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
months_name = pd.Series(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
vals = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
months_dict = dict(zip(months_name, vals))
months_dict.update(dict(zip(months_abbr, vals)))


# Replace all words in dictionary by values
def multiple_replace(text, adict=None):
    regx = re.compile(r'\b%s\w*\b' % r'\w*\b|\b'.join(map(re.escape, months_abbr)))
    def dict_replace(match):
        return adict[match.group(0)]
    return regx.sub(dict_replace, text)
    # TODO return month no. without replacement

# df = df.apply(multiple_replace, adict=months_dict)

def extract_date(text, adict=None):
    regx = re.compile(r'\b%s\w*\b' % r'\w*\b|\b'.join(map(re.escape, months_abbr)))
    def dict_replace(match):
        return adict[match.group(0)]
    return regx.sub(dict_replace, text)
    # TODO return month no. without replacement

'frank'.upper()
'frank  '.rstrip()
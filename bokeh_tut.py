
import numpy as np
import pandas as pd
import os
from subprocess import check_output

from bokeh.plotting import figure, output_file, output_notebook, show
# To view the above examples in a notebook,
# you would only change output_file() to a call to output_notebook() instead.

# Import dataset and show directory
data_dir = 'churn_data'
data_file = 'Churn_Modelling.csv'
path = os.path.join(data_dir, data_file)
try:
    df = pd.read_csv(path)
    print('$: root/' + str(data_dir))
    print(check_output(["ls", data_dir]).decode("utf8"))
except IOError as err:
    print("IO error: {0}".format(err))
    print("Oops! Try again...")

df.head(5)

# df.filter(['Surname', 'CustomerId', 'Exited']).sort_values(by=['Surname', 'CustomerId', 'Exited']).head(10)
df.columns
df.dtypes
df.describe()
df.info()

# Helper function to check for right type
def assert_type(objects, types):
    """Check dtype"""
    assert types is type or tuple
    wrong = ([o for o in objects if not isinstance(o, types)])
    if wrong:
        raise Exception('Not the asserted type')

def df_totype(object, types):
    """Convert a copy of pd.dataframe.columns to requested type."""
    assert type(object) not in (np.ndarray, list), 'Type is already ndarray or list...'
    if types == np.ndarray:
        return object.values.flatten()
    elif types == list:
        return object.values.flatten().tolist()
    else:
        raise Exception('Type is not supported')

df_totype?

# --------------------------------------------->  <-------------------------------------------- #

# prepare some data
x = df.filter(regex='^T', axis=1).head(50)
y = df.filter(like='Credit', axis=1).head(50)
assert_type([x, y], pd.DataFrame)
xlabel , ylabel = x.columns[0], y.columns[0]

# Bokeh need [list or 1D array] for each axis-parameter
x = df_totype(x, list)
y = df_totype(y, list)
assert_type([x, y], (list, np.ndarray))


# output to static HTML file
output_file("bokeh_output_temp.html", title="Bokeh plot")  # tab title

# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label=xlabel, y_axis_label=ylabel)

# add a line renderer with legend and line thickness
p.vbar(x, y, legend="Temp1", line_width=2, top=100)
p.line(y, x, legend="Temp2", line_width=3)
p.line(x, y, color='aqua')

# show the results
show(p)

# --------------------------------------------->  <-------------------------------------------- #

p = figure(
    title='log axis example',
    tools='pan,box_zoom,reset,save',
    x_axis_label=xlabel,
    y_axis_label=ylabel, #y_range=[0, 100], #y_axis_type='log',
)

# create a new plot

# add some renderers
p.line(x, y, legend="y=y", line_width=3)
p.circle(x, y, legend="y=y", fill_color="white", line_color="red", size=6)

# p.line(x, x, legend="y=x")
# p.circle(x, x, legend="y=x", fill_color="white", size=8)
#
# p.line(y, y, legend="x=y", line_color="red")
# p.line(y, x+y, legend="x=y", line_color="orange", line_dash="4 4")

# show the results
show(p)

# --------------------------------------------->  <-------------------------------------------- #

N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = [
    "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x%100, 30+2*y%100)
]

TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset," \
      "tap,save,box_select,poly_select,lasso_select,"

p = figure(tools=TOOLS)

p.scatter(x, y, radius=radii,
          fill_color=colors, fill_alpha=0.6,
          line_color=None)

output_file("color_scatter.html", title="color_scatter.py example")

show(p)  # open a browser


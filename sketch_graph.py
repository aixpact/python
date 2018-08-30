import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.xkcd()
#plt.rcdefaults()  # restore defaults

################
# better use setting teporary for certain plots
with plt.xkcd():
    # This figure will be in XKCD-style
    fig1 = plt.figure()
    # ...

# This figure will be in regular style
fig2 = plt.figure()
##############



X = np.linspace(0,2.32,100)
Y = X*X - 5*np.exp(-5*(X-2)*(X-2))

fig = plt.figure(figsize=(12,5), dpi=72,facecolor="white")
axes = plt.subplot(111)

plt.plot(X,Y, color = 'k', linewidth=2, linestyle="-", zorder=+10)

axes.set_xlim(X.min(),X.max())
axes.set_ylim(1.01*Y.min(), 1.01*Y.max())

axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.xaxis.set_ticks_position('bottom')
axes.spines['bottom'].set_position(('data',0))
axes.yaxis.set_ticks_position('left')
axes.spines['left'].set_position(('data',X.min()))

axes.set_xticks([])
axes.set_yticks([])
axes.set_xlim( 1.05*X.min(), 1.10*X.max() )
axes.set_ylim( 1.15*Y.min(), 1.05*Y.max() )

t = [10,40,82,88,93,99]
plt.scatter( X[t], Y[t], s=50, zorder=+12, c='k')

plt.text(X[t[0]]-.1, Y[t[0]]+.1, "Industrial\nRobot", ha='left', va='bottom')
plt.text(X[t[1]]-.15, Y[t[1]]+.1, "Humanoid\nRobot", ha='left', va='bottom')
plt.text(X[t[2]]-.25, Y[t[2]], "Zombie", ha='left', va='center')
plt.text(X[t[3]]+.05, Y[t[3]], "Prosthetic\nHand", ha='left', va='center')
plt.text(X[t[4]]+.05, Y[t[4]], "Bunraku\nPuppet", ha='left', va='center')
plt.text(X[t[5]]+.05, Y[t[5]], "Human", ha='left', va='center')
plt.text(X[t[2]]-0.05, 1.5, "Uncanny\nValley", ha='center', va='center', fontsize=24)

plt.ylabel("-      Comfort Level      +",y=.5, fontsize=20)
plt.text(.05, -.1, "Human Likeness ->",ha='left', va='top', color='r', fontsize=20)

X = np.linspace(0,1.1*2.32,100)
axes.fill_between(X, 0, -10, color = '0.85', zorder=-1)
axes.fill_between(X, 0, +10, color = (1.0,1.0,0.9), zorder=-1)

#X = np.linspace(1.652,2.135,100)
X = np.linspace(1.5,2.25,100)
Y = X*X - 5*np.exp(-5*(X-2)*(X-2))
axes.fill_between(X, Y, +10, color = (1,1,1), zorder=-1)

axes.axvline(x=1.5,ymin=0,ymax=1, color='.5', ls='--')
axes.axvline(x=2.25,ymin=0,ymax=1, color='.5', ls='--')

plt.savefig("figure-8.pdf")
plt.show()


###########

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Matplotlib Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#
# Author: Nicolas P. Rougier
# Source: New York Times graphics, 2007
# -> http://www.nytimes.com/imagepages/2007/07/29/health/29cancer.graph.web.html
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------
# Data to be represented
diseases   = ["Kidney Cancer", "Bladder Cancer", "Esophageal Cancer",
              "Ovarian Cancer", "Liver Cancer", "Non-Hodgkin's\nlymphoma",
              "Leukemia", "Prostate Cancer", "Pancreatic Cancer",
              "Breast Cancer", "Colorectal Cancer", "Lung Cancer"]
men_deaths = [10000, 12000, 13000, 0, 14000, 12000,
              16000, 25000, 20000, 500, 25000, 80000]
men_cases = [30000, 50000, 13000, 0, 16000, 30000,
             25000, 220000, 22000, 600, 55000, 115000]
women_deaths = [6000, 5500, 5000, 20000, 9000, 12000,
                13000, 0, 19000, 40000, 30000, 70000]
women_cases = [20000, 18000, 5000, 25000, 9000, 29000,
               24000, 0, 21000, 160000, 55000, 97000]

# ----------
# Choose some nice colors
matplotlib.rc('axes', facecolor = 'white')
matplotlib.rc('figure.subplot', wspace=.65)
matplotlib.rc('grid', color='white')
matplotlib.rc('grid', linewidth=1)

# Make figure background the same colors as axes
fig = plt.figure(figsize=(12,7), facecolor='white')


# ---WOMEN data ---
axes_left  = plt.subplot(121)

# Keep only top and right spines
axes_left.spines['left'].set_color('none')
axes_left.spines['right'].set_zorder(10)
axes_left.spines['bottom'].set_color('none')
axes_left.xaxis.set_ticks_position('top')
axes_left.yaxis.set_ticks_position('right')
axes_left.spines['top'].set_position(('data',len(diseases)+.25))
axes_left.spines['top'].set_color('w')

# Set axes limits
plt.xlim(200000,0)
plt.ylim(0,len(diseases))

# Set ticks labels
plt.xticks([150000, 100000, 50000, 0],
           ['150,000', '100,000', '50,000', 'WOMEN'])
axes_left.get_xticklabels()[-1].set_weight('bold')
axes_left.get_xticklines()[-1].set_markeredgewidth(0)
for label in axes_left.get_xticklabels():
    label.set_fontsize(10)
plt.yticks([])



# Plot data
for i in range(len(women_deaths)):
    H,h = 0.8, 0.55
    # Death
    value = women_cases[i]
    p = patches.Rectangle(
        (0, i+(1-H)/2.0), value, H, fill=True, transform=axes_left.transData,
        lw=0, facecolor='red', alpha=0.1)
    axes_left.add_patch(p)
    # New cases
    value = women_deaths[i]
    p = patches.Rectangle(
        (0, i+(1-h)/2.0), value, h, fill=True, transform=axes_left.transData,
        lw=0, facecolor='red', alpha=0.5)
    axes_left.add_patch(p)

# Add a grid
axes_left.grid()

plt.text(165000,8.2,"Leading Causes\nOf Cancer Deaths", fontsize=18,va="top")
plt.text(165000,7,"""In 2007, there were more\n"""
                  """than 1.4 million new cases\n"""
                  """of cancer in the United States.""", va="top", fontsize=10)

# --- MEN data ---
axes_right = plt.subplot(122, sharey=axes_left)

# Keep only top and left spines
axes_right.spines['right'].set_color('none')
axes_right.spines['left'].set_zorder(10)
axes_right.spines['bottom'].set_color('none')
axes_right.xaxis.set_ticks_position('top')
axes_right.yaxis.set_ticks_position('left')
axes_right.spines['top'].set_position(('data',len(diseases)+.25))
axes_right.spines['top'].set_color('w')


# Set axes limits
plt.xlim(0,200000)
plt.ylim(0,len(diseases))

# Set ticks labels
plt.xticks([0, 50000, 100000, 150000, 200000],
           ['MEN', '50,000', '100,000', '150,000', '200,000'])
axes_right.get_xticklabels()[0].set_weight('bold')
for label in axes_right.get_xticklabels():
    label.set_fontsize(10)
axes_right.get_xticklines()[1].set_markeredgewidth(0)
plt.yticks([])

# Plot data
for i in range(len(men_deaths)):
    H,h = 0.8, 0.55
    # Death
    value = men_cases[i]
    p = patches.Rectangle(
        (0, i+(1-H)/2.0), value, H, fill=True, transform=axes_right.transData,
        lw=0, facecolor='blue', alpha=0.1)
    axes_right.add_patch(p)
    # New cases
    value = men_deaths[i]
    p = patches.Rectangle(
        (0, i+(1-h)/2.0), value, h, fill=True, transform=axes_right.transData,
        lw=0, facecolor='blue', alpha=0.5)
    axes_right.add_patch(p)

# Add a grid
axes_right.grid()

# Y axis labels
# We want them to be exactly in the middle of the two y spines
# and it requires some computations
for i in range(len(diseases)):
    x1,y1 = axes_left.transData.transform_point( (0,i+.5))
    x2,y2 = axes_right.transData.transform_point((0,i+.5))
    x,y = fig.transFigure.inverted().transform_point( ((x1+x2)/2,y1) )
    plt.text(x, y, diseases[i], transform=fig.transFigure, fontsize=10,
             horizontalalignment='center', verticalalignment='center')


# Devil hides in the details...
arrowprops = dict(arrowstyle="-",
                  connectionstyle="angle,angleA=0,angleB=90,rad=0")
x = women_cases[-1]
axes_left.annotate('NEW CASES', xy=(.9*x, 11.5),  xycoords='data',
                   horizontalalignment='right', fontsize= 10,
                   xytext=(-40, -3), textcoords='offset points',
                   arrowprops=arrowprops)

x = women_deaths[-1]
axes_left.annotate('DEATHS', xy=(.85*x, 11.5),  xycoords='data',
                   horizontalalignment='right', fontsize= 10,
                   xytext=(-50, -25), textcoords='offset points',
                   arrowprops=arrowprops)

x = men_cases[-1]
axes_right.annotate('NEW CASES', xy=(.9*x, 11.5),  xycoords='data',
                   horizontalalignment='left', fontsize= 10,
                   xytext=(+40, -3), textcoords='offset points',
                   arrowprops=arrowprops)

x = men_deaths[-1]
axes_right.annotate('DEATHS', xy=(.9*x, 11.5),  xycoords='data',
                   horizontalalignment='left', fontsize= 10,
                   xytext=(+50, -25), textcoords='offset points',
                   arrowprops=arrowprops)


# Done
plt.savefig('figure-1.pdf')
plt.show()

####

# Data
# -----------------------------------------------------------------------------
p, n = 7, 32
X = np.linspace(0,2,n)
Y = np.random.uniform(-.75,.5,(p,n))

# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20,8))
ax = plt.subplot(1,2,1, aspect=1)
ax.patch.set_facecolor((1,1,.75))
for i in range(p):
    plt.plot(X, Y[i], label = "Series %d     " % (1+i), lw=2)
plt.xlim( 0,2)
plt.ylim(-1,1)
plt.yticks(np.linspace(-1,1,18))
plt.xticks(np.linspace(0,2,18))
plt.legend()
plt.grid()

# -----------------------------------------------------------------------------
ax = plt.subplot(1,2,2, aspect=1)
Yy = p-(np.arange(p)+0.5)
Xx = [p,]*p
rects = plt.barh(Yy, Xx, align='center', height=0.75, color='.95', ec='None', zorder=-20)
plt.xlim(0,p), plt.ylim(0,p)

for i in range(p):
    label = "Series %d" % (1+i)
    plt.text(-.1, Yy[i], label, ha = "right", fontsize=16)
    plt.axvline(0,   (Yy[i]-.4)/p, (Yy[i]+.4)/p, c='k', lw=3)
    plt.axvline(.25*p, (Yy[i]-.375)/p, (Yy[i]+.375)/p, c='.5', lw=.5, zorder=-15)
    plt.axvline(.50*p, (Yy[i]-.375)/p, (Yy[i]+.375)/p, c='.5', lw=.5, zorder=-15)
    plt.axvline(.75*p, (Yy[i]-.375)/p, (Yy[i]+.375)/p, c='.5', lw=.5, zorder=-15)
    plt.plot(X*p/2, i+.5+2*Y[i]/p, c='k', lw=2)
    for j in range(p):
        if i != j:
            plt.plot(X*p/2, i+.5+2*Y[j]/p, c='.5', lw=.5, zorder=-10)
plt.text(.25*p, 0, "0.5", va = "top", ha="center", fontsize=10)
plt.text(.50*p, 0, "1.0", va = "top", ha="center", fontsize=10)
plt.text(.75*p, 0, "1.5", va = "top", ha="center", fontsize=10)
plt.axis('off')

plt.savefig("figure-7.pdf")
plt.show()


##################

# This figure shows the name of several matplotlib elements composing a figure

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter


np.random.seed(19680801)

X = np.linspace(0.5, 3.5, 100)
Y1 = 3+np.cos(X)
Y2 = 1+np.cos(1+X/0.75)/2
Y3 = np.random.uniform(Y1, Y2, len(X))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect=1)


def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

ax.xaxis.set_major_locator(MultipleLocator(1.000))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_major_locator(MultipleLocator(1.000))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 4)
ax.set_ylim(0, 4)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=10)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

ax.plot(X, Y1, c=(0.25, 0.25, 1.00), lw=2, label="Blue signal", zorder=10)
ax.plot(X, Y2, c=(1.00, 0.25, 0.25), lw=2, label="Red signal")
ax.plot(X, Y3, linewidth=0,
        marker='o', markerfacecolor='w', markeredgecolor='k')

ax.set_title("Anatomy of a figure", fontsize=20, verticalalignment='bottom')
ax.set_xlabel("X axis label")
ax.set_ylabel("Y axis label")

ax.legend()


def circle(x, y, radius=0.15):
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke
    circle = Circle((x, y), radius, clip_on=False, zorder=10, linewidth=1,
                    edgecolor='black', facecolor=(0, 0, 0, .0125),
                    path_effects=[withStroke(linewidth=5, foreground='w')])
    ax.add_artist(circle)


def text(x, y, text):
    ax.text(x, y, text, backgroundcolor="white",
            ha='center', va='top', weight='bold', color='blue')


# Minor tick
circle(0.50, -0.10)
text(0.50, -0.32, "Minor tick label")

# Major tick
circle(-0.03, 4.00)
text(0.03, 3.80, "Major tick")

# Minor tick
circle(0.00, 3.50)
text(0.00, 3.30, "Minor tick")

# Major tick label
circle(-0.15, 3.00)
text(-0.15, 2.80, "Major tick label")

# X Label
circle(1.80, -0.27)
text(1.80, -0.45, "X axis label")

# Y Label
circle(-0.27, 1.80)
text(-0.27, 1.6, "Y axis label")

# Title
circle(1.60, 4.13)
text(1.60, 3.93, "Title")

# Blue plot
circle(1.75, 2.80)
text(1.75, 2.60, "Line\n(line plot)")

# Red plot
circle(1.20, 0.60)
text(1.20, 0.40, "Line\n(line plot)")

# Scatter plot
circle(3.20, 1.75)
text(3.20, 1.55, "Markers\n(scatter plot)")

# Grid
circle(3.00, 3.00)
text(3.00, 2.80, "Grid")

# Legend
circle(3.70, 3.80)
text(3.70, 3.60, "Legend")

# Axes
circle(0.5, 0.5)
text(0.5, 0.3, "Axes")

# Figure
circle(-0.3, 0.65)
text(-0.3, 0.45, "Figure")

color = 'blue'
ax.annotate('Spines', xy=(4.0, 0.35), xycoords='data',
            xytext=(3.3, 0.5), textcoords='data',
            weight='bold', color=color,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle="arc3",
                            color=color))

ax.annotate('', xy=(3.15, 0.0), xycoords='data',
            xytext=(3.45, 0.45), textcoords='data',
            weight='bold', color=color,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle="arc3",
                            color=color))

ax.text(4.0, -0.4, "Made with http://matplotlib.org",
        fontsize=10, ha="right", color='.5')

plt.show()

###########################

"""
=======
Firefox
=======

This example shows how to create the Firefox logo with path and patches.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# From: http://raphaeljs.com/icons/#firefox
firefox = "M28.4,22.469c0.479-0.964,0.851-1.991,1.095-3.066c0.953-3.661,0.666-6.854,0.666-6.854l-0.327,2.104c0,0-0.469-3.896-1.044-5.353c-0.881-2.231-1.273-2.214-1.274-2.21c0.542,1.379,0.494,2.169,0.483,2.288c-0.01-0.016-0.019-0.032-0.027-0.047c-0.131-0.324-0.797-1.819-2.225-2.878c-2.502-2.481-5.943-4.014-9.745-4.015c-4.056,0-7.705,1.745-10.238,4.525C5.444,6.5,5.183,5.938,5.159,5.317c0,0-0.002,0.002-0.006,0.005c0-0.011-0.003-0.021-0.003-0.031c0,0-1.61,1.247-1.436,4.612c-0.299,0.574-0.56,1.172-0.777,1.791c-0.375,0.817-0.75,2.004-1.059,3.746c0,0,0.133-0.422,0.399-0.988c-0.064,0.482-0.103,0.971-0.116,1.467c-0.09,0.845-0.118,1.865-0.039,3.088c0,0,0.032-0.406,0.136-1.021c0.834,6.854,6.667,12.165,13.743,12.165l0,0c1.86,0,3.636-0.37,5.256-1.036C24.938,27.771,27.116,25.196,28.4,22.469zM16.002,3.356c2.446,0,4.73,0.68,6.68,1.86c-2.274-0.528-3.433-0.261-3.423-0.248c0.013,0.015,3.384,0.589,3.981,1.411c0,0-1.431,0-2.856,0.41c-0.065,0.019,5.242,0.663,6.327,5.966c0,0-0.582-1.213-1.301-1.42c0.473,1.439,0.351,4.17-0.1,5.528c-0.058,0.174-0.118-0.755-1.004-1.155c0.284,2.037-0.018,5.268-1.432,6.158c-0.109,0.07,0.887-3.189,0.201-1.93c-4.093,6.276-8.959,2.539-10.934,1.208c1.585,0.388,3.267,0.108,4.242-0.559c0.982-0.672,1.564-1.162,2.087-1.047c0.522,0.117,0.87-0.407,0.464-0.872c-0.405-0.466-1.392-1.105-2.725-0.757c-0.94,0.247-2.107,1.287-3.886,0.233c-1.518-0.899-1.507-1.63-1.507-2.095c0-0.366,0.257-0.88,0.734-1.028c0.58,0.062,1.044,0.214,1.537,0.466c0.005-0.135,0.006-0.315-0.001-0.519c0.039-0.077,0.015-0.311-0.047-0.596c-0.036-0.287-0.097-0.582-0.19-0.851c0.01-0.002,0.017-0.007,0.021-0.021c0.076-0.344,2.147-1.544,2.299-1.659c0.153-0.114,0.55-0.378,0.506-1.183c-0.015-0.265-0.058-0.294-2.232-0.286c-0.917,0.003-1.425-0.894-1.589-1.245c0.222-1.231,0.863-2.11,1.919-2.704c0.02-0.011,0.015-0.021-0.008-0.027c0.219-0.127-2.524-0.006-3.76,1.604C9.674,8.045,9.219,7.95,8.71,7.95c-0.638,0-1.139,0.07-1.603,0.187c-0.05,0.013-0.122,0.011-0.208-0.001C6.769,8.04,6.575,7.88,6.365,7.672c0.161-0.18,0.324-0.356,0.495-0.526C9.201,4.804,12.43,3.357,16.002,3.356z"


def svg_parse(path):
    commands = {'M': (Path.MOVETO,),
                'L': (Path.LINETO,),
                'Q': (Path.CURVE3,)*2,
                'C': (Path.CURVE4,)*3,
                'Z': (Path.CLOSEPOLY,)}
    path_re = re.compile(r'([MLHVCSQTAZ])([^MLHVCSQTAZ]+)', re.IGNORECASE)
    float_re = re.compile(r'(?:[\s,]*)([+-]?\d+(?:\.\d+)?)')
    vertices = []
    codes = []
    last = (0, 0)
    for cmd, values in path_re.findall(path):
        points = [float(v) for v in float_re.findall(values)]
        points = np.array(points).reshape((len(points)//2, 2))
        if cmd.islower():
            points += last
        cmd = cmd.capitalize()
        last = points[-1]
        codes.extend(commands[cmd])
        vertices.extend(points.tolist())
    return codes, vertices


# SVG to matplotlib
codes, verts = svg_parse(firefox)
verts = np.array(verts)
path = Path(verts, codes)

# Make upside down
verts[:, 1] *= -1
xmin, xmax = verts[:, 0].min()-1, verts[:, 0].max()+1
ymin, ymax = verts[:, 1].min()-1, verts[:, 1].max()+1

fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, aspect=1)

# White outline (width = 6)
patch = patches.PathPatch(path, facecolor='None', edgecolor='w', lw=6)
ax.add_patch(patch)

# Actual shape with black outline
patch = patches.PathPatch(path, facecolor='orange', edgecolor='k', lw=2)
ax.add_patch(patch)

# Centering
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# No ticks
ax.set_xticks([])
ax.set_yticks([])

# Display
plt.show()
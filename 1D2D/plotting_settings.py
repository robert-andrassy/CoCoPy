from matplotlib.pyplot import *

ls = ('-', '--', '-.', (0, (5, 1, 1, 1, 1, 1)), ':')*2

# These colours should be distinguishable by colourblind people.
palette = [
             (0,0,0), # black
             (0.8,0.4,0),    # vermillion (cold)
             (0.35,0.7,0.9), # sky blue
             (0.9,0.6,0),    # orange (cold)
             (0,0.6,0.5),    # bluish green
             (0.8,0.6,0.7),  # reddish purple
             (0,0.45,0.7),   # blue
             (0.95,0.9,0.25),# yellow (cold)
             (1.0, 0, 1.0),  # fuchsia
          ]
lc = [palette[i] for i in (5, 2, 7, 4, 1, 0, 0, 1, 4, 7, 2, 5)]

# figure sizes
## columnwidth
ptwidth = 256.0748
## textheight
ptheight = 705
#textwidth
ptwidthp = 523.5307

inchpt = 0.0138889

width = ptwidth * inchpt
pwidth = ptwidthp * inchpt
height = ptheight * inchpt
scrdpi = 150
savedpi = 300

# for LaTeX
textsize = 7
#textsize = 6
size = textsize
rcParams['font.size'] = size
rcParams['axes.labelsize'] = size
rcParams['legend.fontsize'] = size
rcParams['xtick.labelsize'] = size
rcParams['ytick.labelsize'] = size
rcParams['legend.frameon'] = True
rcParams['legend.facecolor'] = 'white'
rcParams['legend.framealpha'] = 0.8
rcParams['legend.fancybox'] = True
rcParams['legend.edgecolor'] = 'lightgray'
rcParams['lines.linewidth'] = 1.
#rcParams["font.family"] = "Times New Roman"
#rc('font',**{'family':'serif','serif':['Times']}, size=size)
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{txfonts}')
rcParams['legend.fontsize'] = 0.93*size

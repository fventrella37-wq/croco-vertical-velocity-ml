# Common style of the figures of the AGU manuscript and Supporting Information:
# all text 8 pt as printed (figures are drawn at their printed size and included
# in LaTeX at natural size; the AGU text width is 5.5 in).
import matplotlib
FS = 8.0
TEXTWIDTH_IN = 397.48499 / 72.27 * 72.27 / 72.0   # 397.485 pt (TeX) in inches
TEXTWIDTH_IN = 397.48499 / 72.27
STYLE = {'font.size': FS, 'axes.titlesize': FS, 'axes.labelsize': FS, 'xtick.labelsize': FS,
         'ytick.labelsize': FS, 'legend.fontsize': FS, 'legend.title_fontsize': FS, 'figure.titlesize': FS,
         'mathtext.fontset': 'cm', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
         'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'lines.linewidth': 1.0, 'pdf.fonttype': 42,
         'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'}
def apply():
    matplotlib.rcParams.update(STYLE)

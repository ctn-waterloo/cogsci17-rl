import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sns.set_style('ticks')

title_font_size = 24
font_size = 22

datafilename = 'data.txt'
multi_plot = True # If plots of the comparison data are to be made as well
if len(sys.argv) == 2:
    datafilename = sys.argv[1]
if len(sys.argv) == 3:
    datafilename = sys.argv[1]
    multi_plot = sys.argv[2] == 'True'

print(datafilename)
datafile = open(datafilename, "r").readlines()
all_data = np.zeros((len(datafile),4))

for i, line in enumerate(datafile):
        total = line.split(" ")
        all_data[i,:] = total

colour_common = '#4878cf'#'blue'
colour_rare = '#d65f5f'#'red'

df = pd.DataFrame(data=all_data)
ax = sns.barplot(data=df, palette=[colour_common,colour_rare,colour_common,colour_rare])
ax.set_xticks([0.5,2.5])
ax.set( ylim=[.5,1])
ax.set_ylabel('Stay Probability', fontsize=font_size)

legend_rect_common = plt.Rectangle((0,0), 1, 1, fc=colour_common)
legend_rect_rare = plt.Rectangle((0,0), 1, 1, fc=colour_rare)

ax.legend((legend_rect_common, legend_rect_rare), ('Common', 'Rare'), fontsize=font_size)
ax.set_xticklabels(['Rewarded', 'Unrewarded'], fontsize=font_size)
ax.set_title('Model Data', fontsize=title_font_size)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
sns.despine()

m_data = {'Model-Free':np.array([.8, .8, .656, .665]),
          'Model-Based':np.array([.774, .691, .697, .779]),
          'Human Data':np.array([.860, .767, .632, .731])
         }

human_errors = np.array([.890-.832,
                         .807-.728,
                         .667-.596,
                         .766-.696
                        ])

#TODO: put error bars on the human data plot
if multi_plot:
    for i, title in enumerate(['Model-Free', 'Model-Based', 'Human Data']):
        fig = plt.figure()
        df = pd.DataFrame(data=m_data[title].reshape((1,4)))
        ax = sns.barplot(data=df, palette=[colour_common,colour_rare,colour_common,colour_rare])
        ax.set_xticks([0.5,2.5])
        ax.set(#xlabel='Condition', 
               ylim=[.5,1],
              )
        ax.set_ylabel('Stay Probability', fontsize=font_size)
        axs = ax.get_axes()

        legend_rect_common = plt.Rectangle((0,0), 1, 1, fc=colour_common)
        legend_rect_rare = plt.Rectangle((0,0), 1, 1, fc=colour_rare)

        ax.legend((legend_rect_common, legend_rect_rare), ('Common', 'Rare'), fontsize=font_size)
        ax.set_xticklabels(['Rewarded', 'Unrewarded'], fontsize=font_size)
        ax.set_title(title, fontsize=title_font_size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

        # Add error bars to the human data manually
        if title == 'Human Data':
            #NOTE: scaling error bar size to match the plot scaling
            ax.errorbar(x=[0,1,2,3], y=m_data[title], yerr = human_errors/2, fmt=' ', color='#434b58', capthick=2)
        sns.despine()

plt.show()

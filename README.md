# cogsci17-rl

To run the model and generate data, use the script `gather_nengo_data.py`. This script takes various parameters such as the number of runs and number of steps for each run. To see a list of all available parameters, run `gather_nengo_data.py --help`.

The output will be saved in a file in the data directory, with a name beginning with `out_` and based on the parameter settings and the time at which it was run. The data can be plotted by running `plotting_daw.py data/<filename>`

For plots in the same style as the paper, along with the reference plots, run `pretty_plot.py data/<filename>`

To view a version of the model in the Nengo GUI, run `nengo full_model.py`. This file can be edited to change specific parameters.

Some dependencies that may need to be installed:

- nengo
- nengo_gui
- numpy
- scipy
- pandas
- nengolib
- seaborn

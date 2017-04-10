# cogsci17-rl/refactor

To run the model, use pytry with the file `rl_synapse_trial.py`. This takes various parameters such as the number of runs and number of steps for each run. To see a list of all available parameters, run `pytry rl_synapse_trial.py --help`.

For plots in the same style as the paper, along with the reference plots, see `Plotting Figures.ipynb` and `Plotting Dummy Figures.ipynb`.

To view a version of the model in the Nengo GUI, run `nengo rl_synapse_trial.py`. This file can be edited to change specific parameters.

Some dependencies that may need to be installed:

- nengo (pes-synapse branch)
- nengo_gui
- pytry
- numpy
- scipy
- pandas
- nengolib
- seaborn
- jupyter notebook

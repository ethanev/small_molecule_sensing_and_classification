# small_molecule_sensing_and_classification
Project to classify various small molecules and mixtures using sensing equipment developed in the Swager lab at MIT.
Collaboration with: Vera Schroeder 

Start analysis with:
1) selecting_selectors_all.ipynb  - here you work with the full panel of twenty selectors with different models (featurized or not with KNN classifiers). Goal is to find the optimal selectors to collect more data with for a seperate test set.

To find the best combo of selectors use:
0) process the data with selecting_selectors_all.ipynb ( GET the {your_thing}_data.pkl and featurized_{your_thing}.pkl and upload to where you want to do the combo calculations). 
1) making_parallel_script.py with appropriate -n for the number of selectors per combo
2) use the launch_all.sh file (chmod +x launch_all.sh then ./launch_all)
    This is best done on say starcluster with a lot of nodes cause it launchs many copies of 'parallel_opt_selectors.py'
3) download all the data and then analyze with selector_combos_analysis.ipynb

With optimal selector, collect more data on old food items and then on more (2 in our case)
1) 4_optimal_selectors_5_examples.ipynb - used for data workup and analysis
2) feature_analysis.ipynb - look at the feature importances from the featurized random forest you build in 1)
3) paper_plots.ipynb - makes a few of the plots in the paper. rest are from the other notebooks

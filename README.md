# small_molecule_sensing_and_classification
Project to classify various small molecules and mixtures using sensing equipment developed in the Swager lab at MIT.
Collaboration with: Vera Schroeder 

Start analysis with:
1) selecting_selectors_cheese.ipynb  - here you work with the full panel of twenty selectors with different models (featurized or non with KNN/GP classifiers). Goal is to find the optimal selectors to collect more data with for a seperate test set.
2) With optimal selectors and new data...use: optimal_selector_cheese.ipynb (if only data with 3 different compounds) or optimal_selectors_5_cheese.ipynb if you have data with 5 samples (cheese, liquor etc in our case)

To find the best combo of selectors use:
0) process the data with selecting_selectors_cheese.ipynb ( GET the {your_thing}_data.pkl and featurized_{your_thing}.pkl and upload to where you want to do the combo calculations). 
1) making_parallel_script.py with appropriate -n for the number of selectors per combo
2) use the launch_all.sh file (chmod +x launch_all.sh then ./launch_all)
    This is best done on say starcluster with a lot of nodes cause it launchs make copies of 'parallel_opt_selectors.py'
3) download all the data and then analyze with selector_combos_analysis.ipynb

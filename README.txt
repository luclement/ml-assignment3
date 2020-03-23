All code is contained in Jupyter notebooks part1and5.ipynb, part2.ipynb, and part3and4.ipynb.

Their names correspond to the dataset they are used for.

Code can be found here: https://github.com/luclement/ml-assignment3

Datasets:

1. Credit Card Default dataset is provided under datasets folder, if it does not work for some reason, it can be downloaded here: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset/download
  a. Place under datasets/
2. Free Music Archive is quite large, it can be downloaded here: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
  a. Place under datasets/fma_metadata/

Install:

1. Run "pip install -r requirements.txt" to install required dependencies
2. Make sure you have Jupyter installed by running "jupyter --version" and you should see jupyter-notebook installed
3. To start the notebook, run: jupyter notebook
4. Navigate to desired notebook and run from the top down

Code references:
1. Sklearn for general tutorials and library usage: https://scikit-learn.org/stable/user_guide.html
2. Yellowbrick elbow method: https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
3. GMM model selection from Sklearn: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
4. Tutorial on plotting explained variance for PCA: https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe
5. In utils, credit to Sklearn for the learning curve plotting code: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
6. In utils, credit to Sklearn for the validation curve plotting code: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
7. In utils, credit to FMA for FMA data loading code: https://github.com/mdeff/fma/blob/master/utils.py
8. Credit to FMA for usage instructions/code: https://github.com/mdeff/fma/blob/master/usage.ipynb
9. PCA tutorial: https://cmdlinetips.com/2018/03/pca-example-in-python-with-scikit-learn/
10. SO post on calculating reconstruction error for LDA: https://stackoverflow.com/questions/42957962/linear-discriminant-analysis-inverse-transform

Packages used:
1. scikit-learn: https://scikit-learn.org/stable/
2. pandas: https://pandas.pydata.org/
3. seaborn: https://seaborn.pydata.org/
4. yellowbrick: https://www.scikit-yb.org/en/latest/
5. jupyter: https://jupyter.org/
6. numpy: https://numpy.org/
7. matplotlib: https://matplotlib.org/

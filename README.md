This is my first Machine Learning project. I decided to create my own
problem domain and own dataset.

## Problem
Determine if a github repo is popular with the development community.

## Solution approach
After some trial and error I decided to use these features of a repo to
determine its popularity:

- Contributors   
- Stars   
- Watches 
- Issues
- Forks

The ``train_data.txt`` file has a small number of training data.

The ``train.py`` file uses Support Vector Machine (SVM) to come up with a set of
trained parameters. These are then saved in the `model.pkl` file.

The ``predict.py`` file loads the trained parameters and runs predictions on
test data in ``test_data.txt``.

## Lessons learned
The hardest part was to decide what features to use. I learned the hard way
that classifying the popularity of a repo is harder than it sounds. 
# Deep learning inference of the neutron star equation of state
 Pipeline to infer the equation of state of neutron stars from observations based on deep neural networks (https://arxiv.org/abs/2405.17908).

The data used for the machine learning analysis were generated on Julia (e.g. solution of ODEs, construction of mass-radius-tidal data, etc.). The deep learning models and analysis were done in Python using Tensorflow/Keras.

# Content
Here is a list of the files:
- Datagenerator.ipynb: This is the notebook (Julia) which solves the equations and generates the training/testing data set we used in the deep-learning analysis.
- Deterministic_Neural_Network.ipynb: This is the notebook (Python) for the analysis of the equation of state based on the deterministic neural network.
- model_det.py: The Python file containing the definition of the deep-learning model and the required functionality needed for the notebook "Deterministic_Neural_Network.ipynb".
- Rescaledap4.dat and Rescaledsly.dat, tabulated EOS (AP4 and SLy) for the low-density description. They are already rescaled in G=c=M_sun=1 units.
- matrixrho2.dat and matrixcs2.dat, datasets of random values for mass density and speed of sound used to construct the phenomenological EOS at high-density for training/validation datasets.
- matrixrhoTest.dat and matrixcsTest.dat, datasets of random values for mass density and speed of sound used to construct the phenomenological EOS at high-density for testing dataset.

# Version
The version of libraries used are as follows:

Julia:
- julia 1.10.2
- DifferentialEquations v7.13.0
- DelimitedFiles v1.9.1
- Interpolations v0.15.1
- Symbolics v5.22.1
- Plots v1.40.1
- Roots v2.1.2
- Distributions v0.25.107
- CSV v0.10.13
- LaTeXStrings v.1.3.1
- Images v0.26.0

Python:
- python 3
- tensorflow version 2.14.0
- numpy version 1.24.3
- pandas version 2.0.3
- sklearn version 1.3.0
- matplotlib version 3.5.1
- statmodels version 0.14.1 

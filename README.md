# matlab-rocf
This repo contains a MATLAB implementation of the [ROCF (Relative Outlier Cluster Factor) clustering algorithm by Jinlong Huang et al.](https://doi.org/10.1016/j.knosys.2017.01.013).

ROCF (Relative Outlier Cluster Factor) detects isolated outliers and outlier clusters based upon a mutual k-nearest neighbor graph and the idea that outlier clusters are much smaller in size than normal clusters.  It also detects normal clusters in addition to outlier clusters, but it's main intent is outlier detection. For more details, see the original [paper](https://doi.org/10.1016/j.knosys.2017.01.013).

## Dependencies
- My [knn-graphs](https://github.com/tvannoy/knn-graphs) MATLAB library
- Statistics and Machine Learning Toolbox

To run the tests contained in the Jupyter notebook, you will need to install the Jupyter [matlab kernel](https://github.com/calysto/matlab_kernel).

To use the NN Descent algorithm to construct the mutual KNN graph used by ROCF, you need [pynndescent](https://github.com/lmcinnes/pynndescent) and [MATLAB's Python language interface](https://www.mathworks.com/help/matlab/call-python-libraries.html). I recommend using Conda to set up an environment, as MATLAB is picky about which Python versions it supports. 

## Usage
`Rocf` is a class with a single public method, `cluster`. The results of the clustering operation are stored in read-only public properties. `Rocf` is an iceberg class, which, depending on who you ask, is either [bad](https://www.artima.com/weblogs/viewpost.jsp?thread=125574) or [good](https://calebhearth.com/iceberg-classes)...

Creating an Rocf object:
```matlab
% Create an Rocf object using a 5-nearest-neighbor graph.
% nNeighborsIndex is how many neighbors used to create the knn index, and must be >= nNeighbors + 1
% because the index includes self-edges (each point is it's own nearest neighbor).
nNeighors = 5;
nNeighborsIndex = 6;
rocf = Rocf(data, nNeighbors, nNeighborsIndex);

% Use the NN Descent algorithm to create the knn index; this is much faster than an exhaustive search
rocf = Rocf(data, nNeighbors, nNeighborsIndex, 'Method', 'nndescent');

% Explicitly use an exhaustive search, which is the default
rocf = Rocf(data, nNeighbors, nNeighborsIndex, 'Method', 'knnsearch');

% Use a precomputed knn index
knnidx = knnindex(data, nNeighborsIndex);
rocf = Rocf(data, nNeighbors, knnidx);
```

Clustering:
```matlab
rocf.cluster();
% Or
cluster(rocf);

% Inspect clusters, outliers, and labels
rocf.Clusters
rocf.Outliers
rocf.Labels
```

For more details, see the help text: `help Rocf`. `ROCF tests.ipynb` also contains many tests, which can be used as usage examples. 

## Contributing
All contributions are welcome! Just submit a pull request or open an issue.
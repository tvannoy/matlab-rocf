classdef Rocf < handle
%Rocf Cluster-based outlier detection 
%   ROCF (Relative Outlier Cluster Factor) detects isolated outliers and
%   outlier clusters based upon a mutual k-nearest neighbor graph and the idea
%   that outlier clusters are much smaller in size than normal clusters. 
%   It also detects normal clusters in addition to outlier clusters, 
%   but it's main intent is outlier detection. 
%
%   Rocf constructor:
%       rocf = ROCF(X, k, indexNeighbors) creates an ROCF object for input
%       data X, with k nearest neighbors used in constructing the mutual knn
%       graph. indexNeighbors is the number of neighbors used to build the
%       k-nearest neighbors index. indexNeighbors must be >= k + 1
%
%       rocf = Rocf(X, k, indexNeighbors, 'Method', method) creates an Rocf
%       object for input data X, with k nearest neighbors used in constructing
%       the mutual knn graph, using the specified knn method.
%
%       Optional parameters:
%       'Method'        - The method to use when building the knn index
%           "knnsearch" - Use KNNSEARCH from Mathwork's Statistics and
%                         Machine Learning Toolbox. This is the default
%                         method. 
%           "nndescent" - Use the nearest neighbor descent algorithm to
%                         build an approximate knn index. For large
%                         matrices, this is much faster than KNNSEARCH, at
%                         the cost of slightly less accuracy. This method
%                         requires the pynndescent python package to be
%                         installed and accessible through MATLAB's Python
%                         external language interface
%
%   Rows of X correspond to observations, and columns correspond to variables.
%
%   Rocf properties:
%       K                - number of nearest neighbors 
%       Data             - input data X 
%       KnnIndex         - k-nearest neighbor index
%       MutualKnnGraph   - mutual knn graph used for clustering
%       Clusters         - Clusters{i} contains indices for points in cluster i
%       OutlierClusters  - same as Clusters, but for outlier clusters
%       IsolatedOutliers - array of outliers that do not belong to a cluster
%       RocfValues       - ROCF values for each cluster
%       OutlierRate      - percentage of outliers in the input data
%       Labels           - cluster labels; -1 corresponds to isolated outliers  
%
%   Rocf methods:
%       cluster          - cluster the input data
%
%   This algorithm was presented in http://dx.doi.org/10.1016/j.knosys.2017.01.013
%   by Jinlong Huang et al., but no implementation was given, so this 
%   implementation is based upon the algorithm descriptions given in the paper.
%
%   See also KNNINDEX, MUTUALKNNGRAPH

    properties (SetAccess = public, AbortSet)
        % Number of nearest neighbors used to construct the knn graph
        K (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    properties (SetAccess = private)
        % Input data; rows are observations, and columns are variables
        Data (:,:) double
        % k-nearest neighbor index
        KnnIndex (:,:) int32
        % Mutual knn graph used for clustering (connected components are clusters)
        MutualKnnGraph (1,1) graph
        % Cell array of clusters; Clusters{i} contains indices for all points
        % that are in cluster i; clusters are sorted in ascending order of size
        Clusters (:,1) cell
        % Same thing as Clusters, except for outlier clusters; outlier clusters
        % are also included in Clusters
        OutlierClusters (:,1) cell
        % Outliers that do not belong to any cluster
        IsolatedOutliers (1,:) int32
        % ROCF values for each clusters; the ROCF value is a measure of how big 
        % the next largest cluster is compared to the current cluster
        RocfValues (1,:) double
        % Percentage of outliers in the data
        OutlierRate (1,1) double
        % Cluster labels. Isolated outliers have a label value of -1; all other 
        % clusters have labels starting at 1
        Labels (:,1) int32 = []
    end
    
    properties (Constant)
        % Threshold used to determine the transition between outlier clusters
        % and normal clusters; values greater than this threshold indicate
        % an outlier cluster
        ROCF_THRESHOLD = 0.1
    end

    methods(Access = public)
        function obj = Rocf(X, k, index, options)
        %ROCF Construct an ROCF object
        %   rocf = ROCF(X, k, nIndexNeighbors) creates an ROCF object for input
        %   data X, with k nearest neighbors used in constructing the mutual knn
        %   graph. indexNeighbors is the number of neighbors used to build the
        %   k-nearest neighbors index. indexNeighbors must be >= k + 1
        %
	%
        %   rocf = rocf(X, k, knnIndex) creates an ROCF object
        %   using a precomputed knn index, knnIndex. knnIndex must have the same
        %   number of rows as X.
	%
        %   rocf = Rocf(X, k, nIndexNeighbors, 'Method', method) creates an Rocf
        %   object for input data X, with k nearest neighbors used in
        %   constructing the mutual knn graph, using the specified knn method.
        %
        %   Optional parameters:
        %   'Method'        - The method to use when building the knn index
        %       "knnsearch" - Use KNNSEARCH from Mathwork's Statistics and
        %                     Machine Learning Toolbox. This is the default
        %                     method. 
        %       "nndescent" - Use the nearest neighbor descent algorithm to
        %                     build an approximate knn index. For large
        %                     matrices, this is much faster than KNNSEARCH, at
        %                     the cost of slightly less accuracy. This method
        %                     requires the pynndescent python package to be
        %                     installed and accessible through MATLAB's Python
        %                     external language interface

            arguments
                X (:,:) double
                k (1, 1) {mustBePositive, mustBeInteger}
                index
                options.Method (1,1) string {mustBeMember(options.Method, ["knnsearch", "nndescent"])} = "knnsearch"
            end

            obj.Data = X;

	    if numel(index) == 1
               indexNeighbors = index;

                if indexNeighbors < k + 1
                    error("indexNeighbors must be >= k + 1")
                end

               obj.KnnIndex = knnindex(obj.Data, indexNeighbors, ...
                   'Method', options.Method);
            elseif size(index, 1) == size(X, 1)
                if size(index, 2) < k + 1
                    error("knnIndex must have # columns >= k + 1")
                end
                obj.KnnIndex = index;
	    else
		% TODO: better error message
		error("argument 3 is incorrect")
	    end

            obj.Labels = zeros(size(X,1), 1, 'int32');
            obj.K = k;

            % the setter for k won't execute for k = 1 because 1 is the
            % default value for k, so AbortSet will stop set.K from running
            if k == 1
                obj.MutualKnnGraph = mutualknngraph(obj.KnnIndex, obj.K, ...
                    'Precomputed', true);
            end
        end

        function cluster(obj)
        %CLUSTER Cluster the input data used to create the ROCF object
        %   The ROCF algorithm finds rough clusters, which may not always be
        %   the true clusters, by finding the connected components of the input
        %   data's mutual k-nearest neighbor graph. It then determines which 
        %   clusters, if any, are outlier clusters, based upon their ROCF score.
        %   Clusters of size < k are considered isolated outliers.
        %
        %   The resulting cluster assignments and labels are saved to the 
        %   object, rather than being output by this method.

            obj.roughlyCluster();
            obj.detectIsolatedOutliers();
            obj.detectOutlierClusters();
            obj.computeOutlierRate();
            obj.setLabels();
        end
    end

    methods
        function obj = set.K(obj, k)
            % this method only gets called if k != obj.K, due to AbortSet
            if k >= size(obj.KnnIndex, 2)
                error("Error setting property 'K' of  class 'RnnDbsan':\n" + ...
                    "'K' must be less than knn index size %d", ...
                    size(obj.KnnIndex, 2));
            end

            obj.K = k;
            obj.MutualKnnGraph = mutualknngraph(obj.KnnIndex, obj.K, ...
                'Precomputed', true);
        end
    end

    methods(Access = private)

        function roughlyCluster(obj)
        % The data is clustered by finding connected components in the 
        % mutual knn graph. 
        %
        % This corresponds to the RoughlyCluster algorithm in the original paper

            % The clusters are represented by the connected components
            % of the mutual knn graph. This is not how the paper describes
            % the "RoughlyCluster" algorithm, but this graph interpretation
            % is much simpler to implement. 'OutputForm' = 'cell' returns a 
            % cell array where the i-th cell contains all of the node IDs 
            % belonging to component i. 
            obj.Clusters = conncomp(obj.MutualKnnGraph, 'OutputForm', 'cell');
	    obj.Clusters = cellfun(@(c) int32(c), obj.Clusters, 'UniformOutput', false);

            obj.sortClustersBySize();
        end

        function computeRocfValues(obj)
        % Compute the ROCF value for each cluster. The ROCF value is an 
        % indication of how likely it is that the cluster is an outlier.
        % Larger values indicate that the next largest cluster is much larger
        % than the current cluster, indicating that the current cluster, and 
        % all smaller clusters, are possibly outliers.
        %
        % ROCF(C_i) = 1 - exp(- |C_{i+1}| / |C_i|^2)

            obj.RocfValues = zeros(length(obj.Clusters), 1);

            % we can't compute an ROCF for the last cluster, as it is the 
            % biggest cluster 
            for i = 1:length(obj.Clusters) - 1
                clusterSize = length(obj.Clusters{i});
                nextClusterSize = length(obj.Clusters{i + 1});
                obj.RocfValues(i) = 1 - exp(-nextClusterSize / clusterSize^2);
            end
        end

        function sortClustersBySize(obj)
        % Sort clusters in ascending order by size. 
        %
        % The paper sorts the set of clusters in ascending or by size, and 
        % doing so is necessary for finding the outlier clusters.

            % compute the sizes of each cluster, then sort these sizes
            % and return the cluster indices in order of sorted cluster sizes
            [~, sortIdx] = sort(cellfun('length', obj.Clusters));

            % sort/rearrange the clusters
            obj.Clusters = obj.Clusters(sortIdx);
        end

        function detectIsolatedOutliers(obj)
        % Detect and remove isolated outliers. 
        % 
        % Isolated outliers are defined as clusters of size < k.

            clustersToRemove = [];

            for i = 1 : length(obj.Clusters)
                if length(obj.Clusters{i}) < obj.K
                    % append points in Cluster{i} to the isolated outliers
                    obj.IsolatedOutliers = horzcat(obj.IsolatedOutliers, obj.Clusters{i});

                    clustersToRemove(end + 1) = i;
                else
                    % since the clusters are sorted by size, we can stop once
                    % we reach of cluster of size >= k
                    break
                end
            end

            % remove the isolated outlier "clusters" from the set of clusters
            obj.Clusters(clustersToRemove) = [];
        end
       
        function detectOutlierClusters(obj)
        % Detect outlier clusters
        %
        % The number of outlier clusters is based upon the ROCF values.
        % The paper defines the number of outlier clusters as the cluster
        % that has the maximum ROCF value, if that ROCF value is greater than 
        % a predefined threshold (0.1 in the paper). If the max ROCF is less
        % than the threshold, all clusters are normal clusters.

            obj.computeRocfValues();

            [maxRocf, maxRocfIdx] = max(obj.RocfValues);

            if maxRocf > obj.ROCF_THRESHOLD
                obj.OutlierClusters = obj.Clusters(1:maxRocfIdx);
            end
        end

        function computeOutlierRate(obj)
        % Compute the outlier rate of the dataset
        %
        % The outlier rate is the percentage of outliers in the dataset,
        % or equivalently (as the paper defines it) as 
        % 1 - (# points in normal clusters) / (# total points)

            nOutlierClusters = length(obj.OutlierClusters);

            normalClusterSizes = cellfun('length', ...
                obj.Clusters(nOutlierClusters + 1 : end));

            obj.OutlierRate = 1 - sum(normalClusterSizes)/size(obj.Data, 1);
        end
        
        function setLabels(obj)
        % Set cluster membership labels
        %
        % Isolated outliers are given a label of -1, somewhat arbitrarily,
        % but also because that's what DBSCAN implementations often do. 
        % All other clusters have labels starting from 1, where cluster 1
        % is the smallest cluster.

            obj.Labels(obj.IsolatedOutliers) = int32(-1);

            for i = 1:length(obj.Clusters)
                obj.Labels(obj.Clusters{i}) = int32(i);
            end
        end
    end
end

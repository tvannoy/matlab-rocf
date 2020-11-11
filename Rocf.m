classdef Rocf < handle
%Rocf Cluster-based outlier detection 
%   ROCF (Relative Outlier Cluster Factor) detects isolated outliers and
%   outlier clusters based upon a mutual k-nearest neighbor graph and the idea
%   that outlier clusters are much smaller in size than normal clusters. 
%   It also detects normal clusters in addition to outlier clusters, 
%   but it's main intent is outlier detection. 
%
%   Rocf constructor:
%       rocf = ROCF(X, k)
%
%   Rows of X correspond to observations, and columns correspond to variables.
%   k is the number of nearest neighbors to use when constructing the mutual 
%   k-nearest neighbor graph.
%
%   Rocf properties:
%       Data             - input data X 
%       K                - number of nearest neighbors 
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

    properties (SetAccess = private)
        % Input data; rows are observations, and columns are variables
        Data (:,:) double
        % Number of nearest neighbors used to construct the mutual knn graph
        K (1,1) double {mustBePositive, mustBeInteger} = 1
        % Mutual knn graph used for clustering (connected components are clusters)
        MutualKnnGraph (1,1) graph
        % Cell array of clusters; Clusters{i} contains indices for all points
        % that are in cluster i; clusters are sorted in ascending order of size
        Clusters (:,1) cell
        % Same thing as Clusters, except for outlier clusters; outlier clusters
        % are also included in Clusters
        OutlierClusters (:,1) cell
        % Outliers that do not belong to any cluster
        IsolatedOutliers (1,:) double
        % ROCF values for each clusters; the ROCF value is a measure of how big 
        % the next largest cluster is compared to the current cluster
        RocfValues
        % Percentage of outliers in the data
        OutlierRate (1,1) double
        % Cluster labels. Isolated outliers have a label value of -1; all other 
        % clusters have labels starting at 1
        Labels (:,1) double {mustBeInteger} = []
    end
    
    properties (Constant)
        % Threshold used to determine the transition between outlier clusters
        % and normal clusters; values greater than this threshold indicate
        % an outlier cluster
        ROCF_THRESHOLD = 0.1
    end

    methods(Access = public)
        function obj = Rocf(X, k)
        %ROCF Construct an ROCF object
        %   rocf = ROCF(X, k) creates an ROCF object for input data X, with k 
        %   nearest neighbors used in constructing the mutual knn graph.

            obj.Data = X;
            obj.K = k;
            obj.Labels = zeros(size(X,1), 1);
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

    methods(Access = private)

        function roughlyCluster(obj)
        % The data is clustered by finding connected components in the 
        % mutual knn graph. 
        %
        % This corresponds to the RoughlyCluster algorithm in the original paper

            obj.MutualKnnGraph = mutualknngraph(obj.Data, obj.K);

            % The clusters are represented by the connected components
            % of the mutual knn graph. This is not how the paper describes
            % the "RoughlyCluster" algorithm, but this graph interpretation
            % is much simpler to implement. 'OutputForm' = 'cell' returns a 
            % cell array where the i-th cell contains all of the node IDs 
            % belonging to component i. 
            obj.Clusters = obj.MutualKnnGraph.conncomp('OutputForm', 'cell');

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
            [~, sort_idx] = sort(cellfun('length', obj.Clusters));

            % sort/rearrange the clusters
            obj.Clusters = obj.Clusters(sort_idx);
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

            obj.Labels(obj.IsolatedOutliers) = -1;

            for i = 1:length(obj.Clusters)
                obj.Labels(obj.Clusters{i}) = i;
            end
        end
    end
end
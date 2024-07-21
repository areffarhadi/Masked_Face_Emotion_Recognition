classdef (InferiorClasses = {?matlab.graphics.axis.Axes, ?matlab.ui.control.UIAxes}) rocmetrics < matlab.mixin.CustomDisplay & classreg.learning.internal.DisallowVectorOps
    %ROCMETRICS Compute Receiver Operating Characteristic (ROC) curve or other
    %   performance curve for binary or multi-class classification model.
    %
    %   Create a ROCMETRICS object to evaluate the performance of
    %   classification model by providing the true labels, a matrix of
    %   scores associated with predictions for those labels, and the
    %   classes for which to compute curves. For each class name given,
    %   the ROCMETRICS object will compute a one-vs-all ROC curve by
    %   transforming the scores input as necessary. ROCMETRICS can also
    %   compute other performance metrics.
    %
    %   rocmetrics Properties:
    %   AUC                      - Area under the ROC curve for each class
    %   Cost                     - Matrix of misclassification costs
    %   Metrics                  - Table that stores the computed FPR, TPR,
    %                              and Threshold values, along with the
    %                              class names and any other computed
    %                              metrics
    %   Labels                   - Given true labels associated with Scores
    %                              in each one-vs-all problem
    %                              for ROC curves of binary models
    %   ClassNames               - Given class names
    %   Prior                    - Prior probabilities for each positive class
    %   Scores                   - Given set of classifier scores
    %   Weights                  - Given set of observation weights
    %
    %   rocmetrics Methods:
    %   rocmetrics               - Create a rocmetrics object
    %   plot                     - Provide a visualization of the computed
    %                              curves
    %   addMetrics               - Compute additional metrics
    %   average                  - Compute a single summary curve by averaging
    %
    %   Example: Plot ROC curves for classification by a classification
    %            tree
    %            % Load the data, and train the tree model on all 3
    %            % classes
    %            load fisheriris
    %            mdl = fitctree(meas, species, 'Crossval', 'on');
    %            [~,scores] = kfoldPredict(mdl);
    %
    %            % Generate curves for all 3 classes, and plot them
    %            treeROC = rocmetrics(species, scores, mdl.ClassNames);
    %            plot(treeROC)
    %
    %   See also perfcurve, rocmetrics.rocmetrics, rocmetrics.plot,
    %   rocmetrics.addMetrics
    
    %   Copyright 2021 The MathWorks, Inc.
    properties (GetAccess = public, SetAccess=private, Dependent)

        %ClassNames The names of the classes associated with each of the
        %   performance metrics. ClassNames can be numeric, logical, or
        %   categorical vector, or a cell array of character vectors.
        ClassNames(1,:);

        %Labels The true labels of the classification problem associated
        %   with the given set of Scores. Labels can be numeric, logical, a
        %   string array, a cell array of characters, or categorical. If
        %   Labels is given in cross-validated format, then it is a cell array
        %   containing one of the aforementioned types.
        Labels;
    end

    properties (GetAccess=public, SetAccess=private)
        %Metrics A table storing the class names, the computed metrics, and
        %   the thresholds. This property stores these values for each
        %   class, vertically concatenated.
        Metrics;

        %AUC The computed area under the ROC curve. AUC is a numeric matrix
        %   with a column for each class given.
        AUC;

        %Scores The input scores, as given by the classifier. Each row of
        %   scores corresponds to the same row of Labels. Scores is a
        %   numeric matrix with as many columns as there are positive
        %   classes. If given in cross-validated format, Scores is a cell
        %   array of numeric matrices.
        Scores;

        %Prior The prior probability for each class. This is a numeric
        %   vector with as many entries as positive classes. Prior(I)
        %   is the prior probability associated with ClassNames(I).
        %   For a binary-class problem where you specify a score vector and 
        %   class name only for a single class, Prior is a two-element vector 
        %   with Prior(1) representing the prior for the specified class.
        Prior(1,:)

        %Cost A square matrix with misclassification costs. Cost(I,J) is the
        %   cost of misclassifying class ClassNames(I) as class
        %   ClassNames(J). For binary class problems, Cost is a 2x2
        %   matrix containing the costs of misclassifying the positive
        %   class as the negative class, and vice versa.
        Cost(:,:)

        %Weights The observation weights. Weights is a numeric vector of
        %   input weights for each row of Scores. If given in
        %   cross-validated format, Weights is a cell array of numeric
        %   vectors.
        Weights
    end

    properties (Dependent, GetAccess=private, SetAccess=private)
        % NumClasses - The number of classes.
        NumClasses(1,1) double
    end

    properties (GetAccess=private, SetAccess=private)
        % IsCrossvalidated - An indicator of if the given data is
        % cross-validated or not. Cross-validated data must be given in
        % cell arrays, one for each cross-validation fold.
        IsCrossvalidated(1,1) logical

        % PrivClassNames - The ClassNames property depends on this
        % private property. This stores the positive classes in
        % clasreg.learning.internal.ClassLabel format, which allows for
        % more succinct set operations.
        PrivClassNames classreg.learning.internal.ClassLabel

        % PrivLabels - The Labels property depends on this private property.
        % This stores the given labels. If they are given as cell arrays in
        % cross-validated form, then this stores cell arrays of
        % classreg.learning.internal.ClassLabel. Otherwise, it is an array
        % of classreg.learning.internal.ClassLabel objects.
        PrivLabels(:,1)

        % ClassRangeIdx - A two-column vector of indices. The X, Y and T
        % data are stored in one table, with values for each class stacked
        % on top of eachother. This property stores the start and end
        % indices of each class within this table.
        ClassRangeIdx(:,2) double

        % ConfusionMetrics - Stores confusion metrics, which are used to
        % generate all metrics on the fly
        ConfusionMetrics table

        % HasConfidenceIntervals - An indicator of whether or not the
        % object has computed confidence intervals
        HasConfidenceIntervals(1,1) logical

        % ProcessNaN - The given 'NanFlag' input, converted to the format
        % expected by perfcurve
        ProcessNaN

        % UseNearest - The given 'UseNearestNeighbor' input, converted to
        % the format expected by perfcurve
        UseNearest

        % ConfidenceIntervalArgs - A cell of arguments to be passed to
        % perfcurve for the computation of confidence intervals
        ConfidenceIntervalArgs

        % ClassScales - A matrix of class scales, with one row per class
        ClassScales

        % FixedMetric - A struct containing values related to the
        % 'FixedMetric' and 'FixedMetricValues' inputs
        FixedMetric

        % NormalizedCost - A cell array containing the normalized costs for
        % each one-vs-all curve.
        NormalizedCost

        % NormalizedPrior - A cell array containing the normalized priors
        % for each one-vs-all curve
        NormalizedPrior
    end

    % get/set methods
    methods
        function res = get.ClassNames(obj)
            res = labels(obj.PrivClassNames)';
        end

        function res = get.Labels(obj)
            if iscell(obj.PrivLabels)
                % Cross-validated - map across cell
                res = cellfun(@(x) labels(x), obj.PrivLabels, 'UniformOutput',false);
            else
                res = labels(obj.PrivLabels);
            end
        end

        function res = get.NumClasses(obj)
            res = numel(obj.PrivClassNames);
        end
    end

    % constructor and object methods
    methods
        function obj = rocmetrics(labels, scores, classNames, NVPairs)
            %ROCMETRICS create a ROCMETRICS object.
            %   METRICS = ROCMETRICS(LABELS,SCORES,CLASSNAMES) computes ROC
            %   curves for a matrix of classifier predictions SCORES given
            %   true class labels, LABELS. The labels can be a numeric vector, 
            %   logical vector, categorical vector, character matrix, string 
            %   array, or cell array of character vectors. SCORES is a matrix 
            %   of floating-point scores returned by a classifier for some 
            %   data. This matrix must have as many rows as LABELS does. 
            %   CLASSNAMES is a list of the classes to compute curves for. 
            %   There must be as many classes as there are columns of SCORES. 
            %   CLASSNAMES must be the same type as LABELS, and each class 
            %   must appear in LABELS.
            %
            %   ROCMETRICS can compute curves for binary class problems or
            %   multi-class problems. If computing curves for multi-class
            %   problems, ROCMETRICS computes one-vs-all ROC curves,
            %   resulting in as many curves as CLASSNAMES.
            %
            %   LABELS and SCORES can also be cell arrays, in which case
            %   ROCMETRICS treats elements in the cell arrays as cross-validation
            %   folds. LABELS can be a cell array of numeric vectors, logical
            %   vectors, character matrices, cell arrays of character vectors,
            %   string array or categorical vectors. All elements in LABELS 
            %   must have the same type. SCORES can be a cell array of numeric 
            %   vectors. The cell arrays for labels and scores must have the
            %   same number of elements. The number of labels in cell j of 
            %   labels must be equal to the number of rows of scores in cell
            %   j of scores for any j in the range from 1 to the number of 
            %   elements in scores. When given data in this format, 
            %   ROCMETRICS computes pointwise confidence bounds.
            %
            %   METRICS = ROCMETRICS(LABELS,SCORES,CLASSNAMES,'NAME1',val1,'NAME2',val2,...)
            %   specifies optional name-value arguments:
            %
            %
            %     'AdditionalMetrics' - Which metrics to compute, in addition
            %                           to the false positive rates and true
            %                           positive rates. The following metrics
            %                           are supported:
            %           TP              - number of true positives
            %           FN              - number of false negatives
            %           FP              - number of false positives
            %           TN              - number of true negatives
            %           TP+FP           - sum of TP and FP
            %           RPP             = (TP+FP)/(TP+FN+FP+TN) rate of
            %                             positive predictions
            %           RNP             = (TN+FN)/(TP+FN+FP+TN) rate of
            %                             negative predictions
            %           ACCU            = (TP+TN)/(TP+FN+FP+TN) accuracy
            %           FNR, MISS       = FN/(TP+FN) false negative rate,
            %                             miss
            %           TNR, SPEC       = TN/(TN+FP) true negative rate,
            %                             specificity
            %           PPV, PREC       = TP/(TP+FP) positive predictive value,
            %                             precision
            %           NPV             = TN/(TN+FN) negative predictive value
            %           ECOST           = (TP*COST(P|P)+FN*COST(N|P)+FP*COST(P|N)+TN*COST(N|N))/(TP+FN+FP+TN)
            %                             expected cost
            %
            %                        In addition, you can define an arbitrary
            %                        metric by supplying a function handle
            %                        of 3 arguments, (C,scale,cost), where
            %                        C is a 2-by-2 confusion matrix, scale
            %                        is a 2-by-1 array of class scales, and
            %                        cost is a 2-by-2 misclassification cost
            %                        matrix. See doc for Performance Curves
            %                        for more info.
            %                        You can supply any of the above in a
            %                        char or string vector, as a single function,
            %                        handle or a cell array of characters or
            %                        function handles.
            %                        Warning: some of these criteria return
            %                        NaN values at one of the two special
            %                        thresholds, 'reject all' and 'accept all'.
            %                        The default is empty, meaning no other
            %                        metrics are computed.
            %
            %     'FixedMetric'    - A char or string of what metric to hold
            %                        fixed. ROCMETRICS computes ROC curves and
            %                        other metrics relative to a fixed metric.
            %                        By default, this is 'Thresholds', but it
            %                        can be a string or char of any of the
            %                        metrics as requested by 'AdditionalMetrics',
            %                        'tpr', 'fpr', or 'thresholds'. To hold
            %                        a custom metric fixed, as defined by a
            %                        function handle, use 'CustomMetricN',
            %                        where N is a number referring to the
            %                        custom metric (for example,
            %                        'CustomMetric1' to indicate the first
            %                        custom metric given in 'AdditionalMetrics').
            %
            %     'FixedMetricValues' - Values for the fixed metric given by
            %                           'FixedMetric'. By default, 'FixedMetricValues'
            %                           is 'all' and ROCMETRICS computes
            %                           all given metrics and thresholds for
            %                           all scores. You can set 'FixedMetricValues'
            %                           to either 'all' or a numeric array.
            %                           If 'FixedMetric' is 'Thresholds', and
            %                           'FixedMetricValues' is unset, 
            %                           ROCMETRICS computes metrics and threshold 
            %                           values for all scores and computes 
            %                           pointwise confidence bounds, if
            %                           requested, for the metrics using threshold
            %                           averaging. If 'FixedMetricValues'
            %                           is instead set to a numeric array,
            %                           ROCMETRICS computes the metric
            %                           values at those specified values.
            %                           If 'FixedMetric' is set to a
            %                           specific metric, then ROCMETRICS
            %                           computes Thresholds and the other
            %                           metrics relative to that fixed
            %                           metric at the values given by
            %                           'FixedMetricValues'. If confidence
            %                           intervals are requested, they are
            %                           computed via vertical averaging.
            %
            %     'NaNFlag'         - This argument specifies how ROCMETRICS
            %                         processes NaN scores. By default, it
            %                         is set to 'omitnan', and observations
            %                         with NaN scores are removed from the
            %                         data. If the parameter is set to 'includenan',
            %                         ROCMETRICS adds observations with NaN
            %                         scores to false classification counts
            %                         in the respective class.
            %
            %  'UseNearestNeighbor' - Set to TRUE to use nearest values found
            %                         in the data instead of the specified
            %                         numeric 'FixedMetricValues' and
            %                         FALSE otherwise. If you specify numeric
            %                         'FixedMetricValues' and set 'UseNearestNeighbor'
            %                         to true, ROCMETRICS returns the nearest
            %                         unique metric values found in the
            %                         data for the metric specified by
            %                         'FixedMetric' as well as corresponding
            %                         other metric values. If you specify
            %                         numeric 'FixedMetricValues' and set
            %                         'UseNearestNeighbor' to false,
            %                         ROCMETRICS will instead return those
            %                         'FixedMetricValues' for the given
            %                         'FixedMetric'. By default this parameter
            %                         is set to true. If you compute confidence 
            %                         bounds by cross-validation or bootstrap,
            %                         this parameter is always false.
            %
            %     'Prior'          - The prior probabilities. A string or
            %                        character of 'empirical' or 'uniform'.
            %                        The default is 'empirical', that is,
            %                        prior probabilities are derived from
            %                        class frequencies. If set to 'uniform',
            %                        all prior probabilities are set equal.
            %                        'Prior' can also be a numeric vector.
            %                        For binary problems, this numeric
            %                        vector must be two elements
            %                        representing the priors for each class.
            %                        For multi-class problems, this is a
            %                        vector with as many elements as
            %                        CLASSNAMES. In this case, Prior(I) is
            %                        the prior probability for
            %                        CLASSNAMES(I).
            %
            %     'Cost'            - A matrix of misclassification
            %                         costs. For binary problems, this is a
            %                         2x2 matrix [0 C(N|P); C(P|N) 0], where
            %                         C(N|P) is the cost of misclassifying
            %                         the positive class as the negative class,
            %                         and C(P|N) is the cost of misclassifying
            %                         the negative class as the positive class.
            %                         For multi-class problems, this is a
            %                         matrix of size KxK, for K positive
            %                         classes. Cost(I,J) is the cost of
            %                         missclassifying CLASSNAMES(I) with
            %                         CLASSNAMES(J). By default, this is a
            %                         matrix with Cost(I,J)=1 if I~=J, and
            %                         Cost(I,J)=0 if I=J
            %
            %     'Alpha'           - A numeric value between 0 and 1.
            %                         ROCMETRICS computes 100*(1-'Alpha') percent
            %                         pointwise confidence bounds for the metrics,
            %                         threshold, and AUC values. By default 
            %                         set to 0.05 for 95% confidence interval.
            %
            %     'Weights'         - A numeric vector of nonnegative
            %                         observation weights. This vector must
            %                         have as many elements as LABELS do.
            %                         If you supply cell arrays for SCORES
            %                         and LABELS and you need to supply 'Weights',
            %                         you must supply them as a cell array
            %                         too. In this case, every element in
            %                         'Weights' must be a numeric vector with
            %                         as many elements as the corresponding
            %                         element in LABELS:
            %                         NUMEL(WEIGHTS{1})==NUMEL(LABELS{1}) etc.
            %                         To compute X axis, Y axis, and Threshold
            %                         values or to compute confidence bounds
            %                         by cross-validation, ROCMETRICS uses these
            %                         observation weights instead of observation
            %                         counts. To compute confidence bounds
            %                         by bootstrap, ROCMETRICS samples N out
            %                         of N with replacement using these weights
            %                         as multinomial sampling probabilities.
            %                         By default, a vector of ones the same
            %                         size as the number of observations,
            %                         or a cell array of vectors of ones
            %                         for cross-validated data.
            %
            %     'NumBootstraps'  - Number of bootstrap replicas for
            %                        computation of confidence bounds. Must
            %                        be a positive integer. By default this
            %                        parameter is set to zero, and bootstrap
            %                        confidence bounds are not computed. If
            %                        you supply cell arrays for LABELS and
            %                        SCORES, this parameter must be set to
            %                        zero because ROCMETRICS cannot use both
            %                        cross-validation and bootstrap to
            %                        compute confidence bounds.
            %
            %     'BootstrapType'  - Confidence interval type used by BOOTCI
            %                        to compute confidence  bounds. You can
            %                        specify any type supported by BOOTCI.
            %                        Use 'doc bootci' for more info. By
            %                        default set to 'bca'.
            %
            %     'NumBootstrapsStudentizedSE' - Number of bootstrap replicates
            %                           to use to compute the studentized
            %                           standard error when computing
            %                           bootstrapped confidence bounds.
            %                           'NumBootstrapsStudentizedSE' is only
            %                           used when 'BootstrapType' is 'stud'
            %                           or 'student'. This is a positive
            %                           integer, indicating the number of
            %                           bootstrap replicas to take to
            %                           generate the estimate. By default
            %                           this is 100.
            %
            %     'BootstrapOptions' - A struct that contains options
            %                          specifying whether to use parallel 
            %                          computation for computing pointwise 
            %                          confidence bounds. This happens when 
            %                          you set 'NumBootstraps' to a
            %                          positive integer. This argument can 
            %                          be created by a call to STATSET. 
            %                          ROCMETRICS uses the following fields:
            %                               'UseParallel'
            %                               'UseSubstreams'
            %                               'Streams'
            %                          For information on these fields, enter
            %                          help parallelstats. The default is
            %                          generated by statset('rocmetrics').
            %                          NOTE: If 'UseParallel' is TRUE and
            %                          'UseSubstreams' is FALSE, then the 
            %                          length of 'Streams' must equal the 
            %                          number of workers used by ROCMETRICS. 
            %                          If a parallel pool is already open, 
            %                          this will be the size of the parallel 
            %                          pool. If a parallel pool is not already
            %                          open, then MATLAB may try to open a 
            %                          pool for you (depending on your 
            %                          installation and preferences). To 
            %                          ensure more predictable results, it
            %                          is best to use the PARPOOL command 
            %                          and explicitly create a parallel pool 
            %                          prior to invoking ROCMETRICS with 
            %                          'UseParallel' set to TRUE.
            %
            %   Example: Plot ROC curves for classification by a
            %   classification tree
            %      % Load the data, and train the tree model on all 3
            %      % classes
            %      load fisheriris
            %      mdl = fitctree(meas, species, 'Crossval', 'on');
            %      [~,scores] = kfoldPredict(mdl);
            %
            %      % Generate curves for all 3 classes
            %      treeROC = rocmetrics(species, scores, mdl.ClassNames);
            %      plot(treeROC)
            %
            %      % Obtain errors on TPR by horizontal averaging
            %      treeROC = rocmetrics(species, scores, mdl.ClassNames, 'NumBootstraps', 1000)
            %      plot(treeROC, 'ShowConfidenceIntervals', true)
            %
            %   See also perfcurve, rocmetrics.plot, rocmetrics.average,
            %   rocmetrics.addMetrics
            arguments
                labels
                scores
                classNames
                NVPairs.AdditionalMetrics(1,:) = '';
                NVPairs.Alpha(1,1) double {mustBeInRange(NVPairs.Alpha, 0, 1, 'exclusive')} = .05;
                NVPairs.BootstrapOptions struct = statset('rocmetrics');
                NVPairs.BootstrapType(1,1) string = "bca";
                NVPairs.Cost = [];
                NVPairs.FixedMetric(1,1) string  = "Thresholds";
                NVPairs.FixedMetricValues {validateFixedValues(NVPairs.FixedMetricValues)} = 'all';
                NVPairs.NaNFlag(1,1) string = "omitnan";
                NVPairs.NumBootstraps(1,1) {mustBeInteger(NVPairs.NumBootstraps), ...
                    mustBeNonnegative(NVPairs.NumBootstraps)} = 0;
                NVPairs.Prior = 'empirical';
                NVPairs.NumBootstrapsStudentizedSE {validateStudSE(NVPairs.NumBootstrapsStudentizedSE)} = [];
                NVPairs.UseNearestNeighbor = true;
                NVPairs.Weights = [];
            end

            % Validate and convert inputs, set some class properties
            origPrior = NVPairs.Prior;
            [scores, labels, classNames, NVPairs, addMetricsFullNames, fixedMetric] = ...
                validateInputs(scores, labels, classNames, NVPairs);
            obj.IsCrossvalidated = iscell(scores);
            obj.HasConfidenceIntervals = (obj.IsCrossvalidated || NVPairs.NumBootstraps > 0);
            obj.PrivLabels = labels;
            obj.PrivClassNames = classNames;

            % Rescale all of the scores as needed for multi-class problems
            % Set the scores property before rescaling so that we retain
            % the original set of scores
            obj.Scores = scores;
            scores = rescaleScores(scores);

            % Assign some inputs to the object
            obj = setInputProperties(obj, NVPairs, fixedMetric, origPrior);
            hasAdditionalMetrics = ~isempty(NVPairs.AdditionalMetrics);

            if obj.HasConfidenceIntervals
                % If the user requests confidence intervals, we have to
                % directly rely on perfcurve to compute them - they can't
                % be derived from the confusion metrics alone

                % Get the ROC metrics
                [obj, fixedMetricOutput] = obj.computeROCWithPerfcurve(scores);

                % Compute any additional metrics requested, using the
                % confusion metrics
                if hasAdditionalMetrics
                    obj = obj.computeAddMetricsFromPerfcurve(scores, NVPairs.AdditionalMetrics, ...
                        addMetricsFullNames, fixedMetricOutput);
                end
            else
                % Without confidence intervals, we can get all metrics from
                % the set of confusion metrics
                % These can be transformed into any other metrics
                [obj, fixedMetricOutput] = obj.computeConfusionMetrics(scores);

                % Compute FPR and TPR, and associated ROC properties
                obj = computeROCWithConfusionMetrics(obj);

                % Compute any additional metrics requested, using the
                % confusion metrics
                if hasAdditionalMetrics
                    obj = obj.computeAddMetricsFromConfusionMetrics(NVPairs.AdditionalMetrics, ...
                        addMetricsFullNames, fixedMetricOutput);
                end
            end
        end

        function [FPR, TPR, T, AUC] = average(obj, type)
            %AVERAGE Compute average ROC curve.
            %   [FPR,TPR,T,AUC] = AVERAGE(METRICS, TYPE) computes average FPR,
            %   TPR, Threshold, and AUC values for the given TYPE. TYPE can
            %   be a character or string of one of the following;
            %       - 'macro'
            %       - 'micro'
            %       - 'weighted'
            %   each corresponding to the type of averaging.
            %
            % In macro-averaging, at a threshold T, the function finds the
            % FPR and TPR values for each class, interpolating as necessary,
            % then averages all the values. These average FPR and TPR values
            % are combined, along with each unique Threshold value in the
            % data, to form a single curve. If instead FPR or TPR were
            % set fixed via the 'FixedMetric' NV pair, then the unique
            % values of that metric are found, and the Thresholds and other
            % metric are averaged.
            %
            % In weighted averaging, the macro-average is computed, except
            % that the contribution of each class is weighted by the prior,
            % given by METRICS.Prior on that class.
            %
            % In micro-averaging, the multi-class problem is converted into
            % a binary problem. Then, a curve is computed for this binary
            % reformulation of the problem.
            %
            % Example: Plot ROC curves for classification by a classification
            %   tree, and generate a summary curve
            %      % Load the data, and train the tree model on all 3 classes
            %      load fisheriris
            %      mdl = fitctree(meas, species, 'Crossval', 'on');
            %      [~, scores] = kfoldPredict(mdl);
            %
            %      % Generate curves for all 3 classes
            %      rocTree = rocmetrics(species, scores, mdl.ClassNames);
            %      % Generate a single macro-average curve for all 3 classes
            %      [xMacro, yMacro, tMacro, aucMacro] = average(rocTree, 'macro');
            %
            %   See also rocmetrics, rocmetrics.plot, rocmetrics.addMetrics

            arguments
                obj rocmetrics
                type(1,1) string
            end
            type = validatestring(type, {'macro','micro','weighted'});
            if obj.NumClasses == 1
                % The average is the same as the curves themselves
                FPR = obj.Metrics.FalsePositiveRate(:,1);
                TPR = obj.Metrics.TruePositiveRate(:,1);
                T = obj.Metrics.Threshold(:,1);
                AUC = obj.AUC(1);
            else
                scores = rescaleScores(obj.Scores);
                if strcmpi(type,'micro')
                    [FPR, TPR, T, AUC] = obj.microAverageCurve(scores);
                else
                    [FPR, TPR, T, AUC] = obj.macroAverageCurve(~strcmpi(type, 'macro'), scores);
                end
            end
        end

        function [perfChart, gObjs] = plot(varargin)
            %PLOT Plot ROC or other classifier performance curve
            %   PLOT(METRICS) produces a plot of all the curves computed by
            %   the rocmetrics METRICS.
            %
            %   PLOT(AX,...) plots into the axes with handle AX.
            %
            %   H = PLOT(__) returns one ROCCurve object for each curve.
            %
            %   [H,G] = PLOT(__) returns a vector H of handles to the
            %   rocmetrics objects in the plot, as well as an array of handles
            %   G. These handles are to the Scatter objects representing the
            %   model operating points (if they exist), and the diagonal (0,0) 
            %   to (1,1) Line in the plot (if it exists).
            %
            %   PLOT(___,Name,Value) specifies additional optional name-value
            %   arguments chosen from the following:
            %       'AverageROCType'          - Which, if any, average ROC curves
            %                                   to plot. The options are 'macro',
            %                                  'micro', 'weighted', or 'none'. 
            %                                   The default is 'none',
            %                                   indicating no curves are
            %                                   plotted. Additionally,
            %                                   AverageROCType is only valid
            %                                   when plotting ROC curves.
            %
            %       'ClassNames'              - Which classes to plot the curves
            %                                   of. Must be the same type as
            %                                   METRICS.ClassNames. The default
            %                                   is METRICS.ClassNames.
            %
            %       'ShowConfidenceIntervals' - A logical scalar to specify
            %                                   whether or not to show
            %                                   confidence intervals in the
            %                                   plot. This option is only 
            %                                   valid for rocmetrics objects 
            %                                   that have computed confidence
            %                                   intervals. The default is false.
            %
            %       'ShowDiagonalLine'        - Specify whether or not to show
            %                                   a diagonal line that extends
            %                                   from [0,0] to [1,1] on the plot.
            %                                   For a plot of ROC curves, this
            %                                   line represents the performance
            %                                   of a classifier choosing
            %                                   classes equally at random.
            %                                   The default is true for ROC 
            %                                   curves, and false for other
            %                                   curves.
            %
            %       'ShowModelOperatingPoint' - A logical to specify whether
            %                                   to show a point on the curve
            %                                   that indicates where on the
            %                                   curve the model which METRICS
            %                                   represents is currently operating.
            %                                   This is only valid for ROC curves.
            %                                   The default is true for ROC
            %                                   curves, and false for other
            %                                   curves.
            %
            %       'XAxisMetric'             - Which metric to display on the
            %                                   X axis. This input can be a char
            %                                   or string of a metric supported
            %                                   by rocmetrics, or the name of
            %                                   a computed metric in
            %                                   METRICS.METRICS. The default
            %                                   is FalsePositiveRate.
            %
            %       'YAxisMetric'             - Which metric to display on the
            %                                   Y axis. This input can be a char
            %                                   or string of a metric supported
            %                                   by rocmetrics, or the name of
            %                                   a computed metric in
            %                                   METRICS.METRICS. The default
            %                                   is TruePositiveRate.
            %
            %   In addition to the above, the function accepts name-value pairs
            %   to specify properties of the lines in the plot, such as 'LineWidth'
            %   and 'Color'.
            %
            %    Example:
            %       % Plot the ROC curves for a classifier trained on the fisheriris
            %       % dataset
            %       load fisheriris
            %       mdl = fitctree(meas, species, 'Crossval', 'on');
            %       [~,scores] = kfoldPredict(mdl);
            %
            %       rocTree = rocmetrics(species, scores, mdl.ClassNames);
            %       plot(rocTree);
            %
            %   See also rocmetrics, rocmetrics.average.

            % Get axes, object, and remaining args
            [varargin{:}] = convertStringsToChars(varargin{:});
            [axParent,obj,varargin] = getObjAndAxesArgs(varargin);

            % Extract any NV pairs specific to plot. The rest are given
            % directly to the graphics object
            args = {'AverageROCType', 'ClassNames', 'ShowDiagonalLine', ...
                'XAxisMetric', 'YAxisMetric', 'ShowModelOperatingPoint'};
            defs = {'none', obj.ClassNames, [], 'FalsePositiveRate', 'TruePositiveRate', []};
            [avgROC, classNames, showLine, xMetric, yMetric, showOperating, ~, graphicsNVPairs] = ...
                internal.stats.parseArgs(args, defs, varargin{:});

            % Validate plot NV pairs - other NV pairs get validated in the graphics
            % object
            [obj, xMetric] = validatePlotMetrics(obj, xMetric, 'XAxisMetric');
            [obj, yMetric] = validatePlotMetrics(obj, yMetric, 'YAxisMetric');
            classNameInds = validateAndExtractClassNames(classNames, obj);
            isROC = strcmpi(xMetric, 'FalsePositiveRate') && ...
                strcmpi(yMetric, 'TruePositiveRate');
            if ~isempty(showLine)
                validateattributes(showLine, {'logical'}, {'scalar'}, ...
                    'plot', 'ShowDiagonalLine');
            else
                showLine = isROC;
            end
            
            if ~isempty(showOperating)
                validateattributes(showOperating, {'logical'}, {'scalar'}, ...
                    'plot', 'ShowModelOperatingPoint');
                if showOperating && (~isROC)
                    % Model operating point not defined
                    error(message('stats:rocmetrics:NoOperatingPointToShow'))
                end
            else
                showOperating = isROC;
            end

            % If average ROC lines are requested, but the metrics aren't
            % ROC, error
            % Default is 'none', indicating no average ROC
            noAvgROC = all(strcmpi(avgROC, 'none'));
            if ~isROC && ~noAvgROC
                error(message('stats:rocmetrics:AvgROCUnsupported'))
            end

            % If any average ROC curves are requested, validate that input
            validateArgs = {{'micro', 'macro', 'weighted'}, 'plot', 'AverageROCType'};
            if iscell(avgROC)
                avgROC = cellfun(@(s) validatestring(s, validateArgs{:}), avgROC, ...
                    'UniformOutput', false);
            elseif ~noAvgROC
                avgROC = validatestring(avgROC, validateArgs{:});
            else
                % 'none' - don't plot any average ROC curves
                avgROC = '';
            end
            avgROC = cellstr(avgROC);

            % If metrics are plotted and no average curves are requested,
            % error, since there be nothing to display
            if noAvgROC && isempty(classNameInds)
                error(message('stats:rocmetrics:NoMetricsToPlot'))
            end

            % If two outputs are requested, but there aren't scatter/line
            % objects to return, error
            if (nargout == 2) && ((showLine + showOperating) == 0)
                error(message('MATLAB:TooManyOutputs'));
            end

            % Generate an axes, if we haven't received one already
            ax = createAndValidateAxes(axParent);

            % Generate one line per requested class
            rocObjs = gobjects(numel(classNameInds), 1);
            gObjs = gobjects(0, 1);

            % Set operating threshold. Binary models operate at threshold
            % .5, multi-class operate at 0 (based on how the scores are
            % rescaled)
            operatingThresh = .5 * (obj.NumClasses == 1);
            for i = 1:numel(rocObjs)
                currInd = classNameInds(i);
                [chartArgs, className] = makeChartArgs(currInd, xMetric, yMetric, obj, isROC);
                rocObjs(i) = mlearnlib.graphics.chart.ROCCurve('Parent',ax, ...
                    chartArgs{:}, graphicsNVPairs{:});

                % Add model operating points, if requested
                gObjs = addModelOperatingPoint(rocObjs(i), ax, showOperating, ...
                    gObjs, operatingThresh, className);
            end
           
            % Plot the average ROC curves, if requested
            % These never have confidence intervals, so remove that NV pair (if given)
            inds = strcmpi('ShowConfidenceIntervals', graphicsNVPairs);
            showCIInds = find(inds);
            showCIInds = [showCIInds, showCIInds + 1];
            graphicsNVPairs(showCIInds) = [];
            for i = 1:numel(avgROC)
                rocObjs = plotAverageROC(rocObjs, obj, avgROC{i}, ax, graphicsNVPairs);
            end

            % If output is requested, return the graphics objects array
            if nargout > 0
                perfChart = rocObjs;
            end

            % Plot unity line, if requested
            lineObj = plotUnityLine(ax, showLine);
            gObjs = [gObjs(:); lineObj];
            % If given an axes we can modify, add labels, etc
            canModifyAxes = (strcmpi(ax.NextPlot, 'replace') || strcmpi(ax.NextPlot, 'replaceall')) ...
                || strcmpi(ax.NextPlot, 'replacechildren');
            addAxesLabelsAndLegend(ax, canModifyAxes, xMetric, yMetric, isROC)
        end

        function obj = addMetrics(obj, metrics)
            %ADDMETRICS - Add other classification metrics to an existing
            %rocmetrics object.
            %   NEWROC = ADDMETRICS(OLDROC, METRICS) takes a rocmetrics object
            %   OLDROOC and computes the additional classification metrics
            %   specified in METRICS. It returns a new rocmetrics object NEWROC
            %   with those computed metrics found in NEWROC.Metrics. METRICS
            %   can be a string or char vector, or a cell array of char
            %   vectors of any of the following:
            %           - 'TP', 'TruePositives'
            %           - 'FN', 'FalseNegatives'
            %           - 'FP', 'FalsePositives'
            %           - 'TN', 'TrueNegatives'
            %           - 'TP+FP', 'SumOfTrueAndFalsePositives'
            %           - 'RPP', 'RateOfPositivePredictions'
            %           - 'RNP', 'RateOfNegativePredictions'
            %           - 'ACCU', 'Accuracy'
            %           - 'FNR', 'MISS', 'FalseNegativeRate'
            %           - 'TNR', 'SPEC', 'TrueNegativeRate'
            %           - 'PPV', 'PREC', 'PositivePredictiveValue'
            %           - 'NPV', 'NegativePredictiveValue'
            %           - 'ECOST', 'ExpectedCost'
            %   For specific definitions of these metrics, see 'help
            %   rocmetrics.rocmetrics', under 'AdditionalMetrics'.
            %
            %   METRICS can also be a function handle to an arbitrary metric
            %   with 3 arguments, (C,scale,cost), where C is a 2-by-2 confusion
            %   matrix, scale is a 2-by-1 array of class scales, and cost is
            %   a 2-by-2 misclassification cost matrix.
            %
            %   Finally, METRICS can be a cell array that contains a mix of
            %   named metrics (those listed above) and function handles to
            %   custom metrics.
            %   NOTE: If given a metric that the object has already
            %   computed, that metric will be ignored.
            %
            % Example: Add 'tnr', 'fnr', to an existing rocmetrics object
            %      load fisheriris
            %      mdl = fitctree(meas, species, 'Crossval', 'on');
            %      [~,scores] = kfoldPredict(mdl);
            %
            %      % Generate curves for all 3 classes
            %      rocTree = rocmetrics(species, scores, mdl.ClassNames);
            %
            %     % Add 'tnr', 'fnr'
            %     rocTree = addMetrics(rocTree, {'tnr', 'fnr'});
            %     rocTree.Metrics
            %     % rocTree.Metrics now has columns for 'tnr' and 'fnr'
            %
            %   See also rocmetrics, rocmetrics.plot, rocmetrics.average


            % Validate new metrics We can rely on some of the same
            % validation utilities, though in this case, metrics can't be
            % empty, unlike in other cases
            if isempty(metrics)
                error(message('stats:rocmetrics:EmptyMetrics'));
            end
            [metrics, metricsFullNames] = validateAdditionalMetrics(metrics, 'metrics', 'addMetrics');

            % See if there is any overlap between already computed metrics
            % and requested metrics. If there is overlap, don't recompute
            % those metrics - just remove them from the list and warn the
            % user we've done this
            namedMetrics = cellfun(@(m) ~isa(m, 'function_handle'), metrics);
            metricsAlreadyComputed = ismember(metricsFullNames(namedMetrics), ...
                obj.Metrics.Properties.VariableNames(3:end));
            if any(metricsAlreadyComputed)
                % Don't recompute anything we already have
                metrics(metricsAlreadyComputed) = [];
                metricsFullNames(metricsAlreadyComputed) = [];
            end

            % The default set of names for custom metrics are
            % CustomMetric1, 2, etc. Update them to be re-numbered relative
            % to the existing metrics in the table
            customMetrics = ~namedMetrics;
            if any(customMetrics)
                numExistingCustom = nnz(contains(obj.Metrics.Properties.VariableNames, 'CustomMetric'));
                customMetricNames = "CustomMetric" + ((1:nnz(customMetrics)) + numExistingCustom);
                metricsFullNames(customMetrics) = customMetricNames;
            end

            % Add the new metrics. We can do this via confusion metrics (if
            % there are no CI), or via perfcurve (if there are CI)
            % Only add metrics if there any left to add - we may have
            % removed all of them in removing already computed ones
            if ~isempty(metrics)
                if obj.HasConfidenceIntervals
                    % Need to go back to the original scores to get the
                    % results
                    scores = rescaleScores(obj.Scores);
                    obj = obj.computeAddMetricsFromPerfcurve(scores, metrics, metricsFullNames);
                else
                    obj = obj.computeAddMetricsFromConfusionMetrics(metrics, metricsFullNames);
                end
            end
        end
    end

    % property groups for disp
    methods (Hidden, Access='protected')
        function propGroups = getPropertyGroups(obj)
            groups = struct();
            groups.Metrics = obj.Metrics;
            groups.AUC = obj.AUC;
            propGroups = matlab.mixin.util.PropertyGroup(groups);
        end
    end

    % helper methods
    methods (Access=private)
        function [obj, fixedMetricOutput] = computeConfusionMetrics(obj, scores)
            % This function computes the four confusion metrics (TP, FP,
            % TN, FN) for all classes
            % These metrics can be transformed to any other metric
            % requested. This approach is only used when a user has not
            % requested confidence intervals

            % If we compute the confusion metrics relative to some other
            % fixed metric, hold on to that result to put in the metrics
            % table, rather than recomputing it
            fixedMetricOutput = [];

            % The metrics are all computed relative to some fixed value,
            % which is the thresholds by default
            if strcmpi(obj.FixedMetric.FullName, 'Thresholds')
                % If the values are fixed relative to a set of thresholds,
                % we can safely get them in 2 calls.
                [obj, tp, fp, T] = obj.perfcurveWrapper(scores, 'XCrit', 'tp', ...
                    'YCrit', 'fp', 'TVals', obj.FixedMetric.Values);
                [~, fn, tn] = obj.perfcurveWrapper(scores, 'XCrit', 'fn', ...
                    'YCrit', 'tn', 'TVals', obj.FixedMetric.Values);
            else
                % In all other cases, we need to call perfcurve once for
                % each metric. We need to get the 'true' metrics values, but
                % the fixed metric value can interfere with this. If a user
                % requests 'tp' to be fixed to some range, and sets
                % 'UseNearest' to be true, then the ground truth 'tp'
                % values get lost, as perfcurve will return the range
                % specified. We also will need to preserve the range the
                % user specified in order to be consistent with perfcurve,
                % in case the user is expecting those specified 'tp'
                % values. By calling perfcurve in this way, we can get
                % both - we get a set of 'tp' (or other metric) values
                % that the user is expecting, and we get the corresponding
                % ground truth 'tp' values for that set of values which we
                % can use to compute other values
                sharedArgs = {'XCrit', obj.FixedMetric.Metric, 'XVals', obj.FixedMetric.Values};
                [obj, fixedMetricOutput, tp, T] = obj.perfcurveWrapper(scores, 'YCrit', 'tp', sharedArgs{:});
                [~, ~, fp] = obj.perfcurveWrapper(scores, 'YCrit', 'fp', sharedArgs{:});
                [~, ~, fn] = obj.perfcurveWrapper(scores, 'YCrit', 'fn', sharedArgs{:});
                [~, ~, tn] = obj.perfcurveWrapper(scores, 'YCrit', 'tn', sharedArgs{:});
            end

            % Start construction of the metrics table, since the threshold
            % values T are consistent across metrics
            % For all the class names, repeat them as many times as there
            % are Threshold values for each class
            classOutputSizes = (obj.ClassRangeIdx(:,2) - obj.ClassRangeIdx(:,1)) + 1;
            classNameCol = arrayfun(@(classInd, repSize) repmat(labels(obj.PrivClassNames(classInd)), repSize, 1), ...
                (1:numel(obj.PrivClassNames))', classOutputSizes, 'UniformOutput', false);
            obj.Metrics = table(vertcat(classNameCol{:}), T, 'VariableNames', {'ClassName', 'Threshold'});

            % Set the ConfusionMetrics table
            obj.ConfusionMetrics = table(tp, fp, tn, fn, 'VariableNames', {'TP','FP','TN','FN'});

            % Compute and set the class scales
            obj = obj.computeClassScales();
        end

        function varargout = perfcurveWrapper(obj, scores, varargin)
            % This function serves as a wrapper around perfcurve
            % It converts inputs to one-vs-all format as needed, then calls
            % perfcurve with arguments specified by varargin.
            % It returns obj, followed by as many perfcurve outputs as are
            % requested by the remainder of the varargout
            classRangeEmpty = isempty(obj.ClassRangeIdx);
            if classRangeEmpty
                % Indices of where each of the classes live in the stacked
                % metrics table hasn't been populated yet
                obj.ClassRangeIdx = zeros(obj.NumClasses, 2);
            end

            % Generate argument list. Some args are specified here, the
            % remainder are given by varargin.
            argList = {'UseNearest', obj.UseNearest, 'ProcessNaN', obj.ProcessNaN, ...
                'Weights', obj.Weights};

            % Set containers for outputs from perfcurve. Use nargout-1
            % since the first output of this function is obj, rather than a
            % perfcurve output
            varargout = cell(1, nargout);
            perfOutputs = cell(1, nargout-1);
            for i = 1:obj.NumClasses
                % Scores - pull out column i for class i. Keep cross-validated scores in
                % that format
                if iscell(scores)
                    curScores = cellfun(@(s) s(:,i), scores, 'UniformOutput', false);
                else
                    curScores = scores(:,i);
                end

                % Call perfcurve for the current class
                [perfOutputs{:}] = perfcurve(obj.PrivLabels, curScores, obj.PrivClassNames(i), ...
                    'Cost', obj.NormalizedCost{i}, 'Prior', obj.NormalizedPrior{i}, argList{:}, varargin{:});

                % Stack the results together
                varargout(2:end) = cellfun(@(curr,total) [total; curr], perfOutputs, ...
                    varargout(2:end), 'UniformOutput', false);

                if classRangeEmpty
                    % Only update the ClassRangeIdx property if it has not
                    % been determined yet
                    prevEndInd = obj.ClassRangeIdx(max(i-1,1), 2);
                    obj.ClassRangeIdx(i,:) = [prevEndInd+1, prevEndInd + size(perfOutputs{1}, 1)];
                end
            end
            varargout{1} = obj;
        end

        function obj = setInputProperties(obj, nvPairs, fixedMetric, origPrior)
            % This function populates the object properties which correspond
            % to inputs, e.g., labels, classNames. It also sets some related
            % internal properties
            obj.Prior = nvPairs.Prior;
            obj.Cost = nvPairs.Cost;
            obj.Weights = nvPairs.Weights;

            % Internal properties
            obj.FixedMetric = fixedMetric;
            obj = obj.normalizeCostAndPrior(origPrior);

            % NaNFlag and UseNearestNeighbor look different between
            % rocmetrics and perfcurve - convert them to the perfcurve
            % format
            if strcmpi(nvPairs.NaNFlag, 'omitnan')
                obj.ProcessNaN = 'ignore';
            else
                obj.ProcessNaN = 'addtofalse';
            end

            if nvPairs.UseNearestNeighbor
                obj.UseNearest = 'on';
            else
                obj.UseNearest = 'off';
            end

            if obj.HasConfidenceIntervals
                % If we have confidence intervals, we need to hold on to
                % the original inputs used to specify how they should be
                % computed, in case a user adds another set of metrics and
                % expects confidence intervals on them.

                % Convert studentized bootstrap CI arguments into a cell
                % array of arguments for bootci, if any are given
                if isempty(nvPairs.NumBootstrapsStudentizedSE)
                    bootarg = {};
                else
                    bootarg = {'Nbootstd', nvPairs.NumBootstrapsStudentizedSE};
                end
                % Bundle all the confidence intervals args together, since either all of
                % them are used or none of them are
                obj.ConfidenceIntervalArgs = {'Alpha', nvPairs.Alpha, 'NBoot', nvPairs.NumBootstraps, ...
                    'BootType', nvPairs.BootstrapType, 'Options', nvPairs.BootstrapOptions, ...
                    'BootArg', bootarg};
            end
        end

        function obj = normalizeCostAndPrior(obj, origPrior)
            % This function computes and sets the internal normalized cost
            % and prior properties. In the multi-class setting with K
            % classes, a 1xK prior and KxK cost matrix get converted into
            % sets of 1x2 priors and 2x2 cost matrices for use by perfcurve
            if obj.NumClasses == 1
                % Binary case - no normalization needed
                obj.NormalizedCost = {obj.Cost};
                obj.NormalizedPrior = {origPrior};
            else
                obj.NormalizedCost = cell(1, obj.NumClasses);
                obj.NormalizedPrior = cell(1, obj.NumClasses);
                for i = 1:obj.NumClasses
                    % Reformat the prior vector into two vectors - one for the positive class
                    % (which only has nonzero prior at the index of the positive class), and
                    % one for the negative classes (which is the same as the original prior,
                    % except is 0 at the index of the positive class
                    probNegative = obj.Prior;
                    probNegative(i) = 0;
                    probPositive = zeros(size(obj.Prior));
                    probPositive(i) = obj.Prior(i);

                    % Normalized cost matrix
                    negAsPosMissCost = probNegative * obj.Cost * probPositive';
                    posAsNegMissCost = probPositive * obj.Cost * probNegative';
                    obj.NormalizedCost{i} = [0 posAsNegMissCost; negAsPosMissCost 0];

                    % Normalized prior
                    % For the one-vs-all setup, with some class A, the prior becomes:
                    % [Pr(A), Pr(~A)] = [Pr(A), 1-Pr(A)]
                    obj.NormalizedPrior{i} = [obj.Prior(i); 1-obj.Prior(i)];
                end
            end
        end

        function obj = computeROCWithConfusionMetrics(obj)
            % This function computes ROC curves for all given classes
            % It uses the ConfusionMetrics property to do so

            % Get the FPR and TPR, then add them to obj.Metrics
            tp = obj.ConfusionMetrics.TP(:,1);
            fp = obj.ConfusionMetrics.FP(:,1);
            tn = obj.ConfusionMetrics.TN(:,1);
            fn = obj.ConfusionMetrics.FN(:,1);
            FPR = fp ./ (fp+tn);
            TPR = tp ./ (tp+fn);
            obj.Metrics = [obj.Metrics, table(FPR, TPR, 'VariableNames', ...
                {'FalsePositiveRate', 'TruePositiveRate'})];

            % Compute the AUC via trapezoid integration
            obj.AUC = zeros(1, obj.NumClasses, 'like', FPR);
            for i = 1:obj.NumClasses
                currRange = obj.ClassRangeIdx(i,1):obj.ClassRangeIdx(i,2);
                if numel(currRange) < 2
                    % Not enough data to integrate
                    obj.AUC(i) = 0;
                else
                    obj.AUC(i) = trapz(FPR(currRange), TPR(currRange));
                end
            end
        end

        function [obj, fixedMetricOutput] = computeROCWithPerfcurve(obj, scores)
            % This function computes the ROC metrics by relying directly on
            % perfcurve, rather than by using confusion metrics.
            % This is called when a user requests confidence intervals, as
            % those can't be derived from confusion metrics alone

            % Depending on which metric is fixed, we can get FPR and TPR in
            % a single call to perfcurve
            fixedMetricOutput = [];
            switch obj.FixedMetric.FullName
                case "Thresholds"
                    % Fixed threshold values, can get ROC in one call using
                    % perfcurve defaults, specified CI arguments, and fixed
                    % value
                    [obj, FPR, TPR, T, auc] = ...
                        obj.perfcurveWrapper(scores, 'TVals', obj.FixedMetric.Values, ...
                        obj.ConfidenceIntervalArgs{:});
                    obj.AUC = auc';
                case "FalsePositiveRate"
                    % Fixed FPR values, can also get ROC in one call with
                    % perfcurve defaults
                    [obj, FPR, TPR, T, auc] = ...
                        obj.perfcurveWrapper(scores, 'XVals', obj.FixedMetric.Values, ...
                        obj.ConfidenceIntervalArgs{:});
                    obj.AUC = auc';
                case "TruePositiveRate"
                    % 'Flipped' ROC curve - just swap the XCrit and YCrit
                    % values
                    [obj, TPR, FPR, T, fAUC] = ...
                        obj.perfcurveWrapper(scores, 'XVals', obj.FixedMetric.Values, 'XCrit', 'tpr',...
                        'YCrit', 'fpr', obj.ConfidenceIntervalArgs{:});
                    % Flip the AUC, and flip the bounds
                    obj.AUC = 1 - fAUC';
                    obj.AUC([2,3],:) = obj.AUC([3,2],:);
                otherwise
                    % Get ROC relative to some other fixed metric - have to
                    % get FPR and TPR in two calls, and numerically
                    % integrate to get the AUC
                    [obj, fixedMetricOutput, FPR, T] = ...
                        obj.perfcurveWrapper(scores, 'XVals', obj.FixedMetric.Values, ...
                        'XCrit', obj.FixedMetric.Metric, ...
                        'YCrit', 'FPR', obj.ConfidenceIntervalArgs{:});
                    [~, ~, TPR] = ...
                        obj.perfcurveWrapper(scores, 'XVals', obj.FixedMetric.Values, ...
                        'XCrit', obj.FixedMetric.Metric, ...
                        'YCrit', 'TPR', obj.ConfidenceIntervalArgs{:});
                    for i = 1:obj.NumClasses
                        % Numerically integrate to get the AUC
                        % In this setup, since the confidence bounds aren't
                        % simultaneous, we can't get the lower and upper
                        % AUC bounds, so just report the current value
                        curRange = obj.ClassRangeIdx(i,1):obj.ClassRangeIdx(i,2);
                        if numel(curRange) < 2
                            obj.AUC(1,i) = 0;
                        else
                            curFPR = FPR(curRange, :);
                            curTPR = TPR(curRange, :);
                            obj.AUC(1,i) = trapz(curFPR(:,1), curTPR(:,1));
                        end
                    end
            end

            % Generate metrics table
            classOutputSizes = (obj.ClassRangeIdx(:,2) - obj.ClassRangeIdx(:,1)) + 1;
            classNameCol = arrayfun(@(classInd, repSize) repmat(labels(obj.PrivClassNames(classInd)), repSize, 1), ...
                (1:numel(obj.PrivClassNames))', classOutputSizes, 'UniformOutput', false);
            obj.Metrics = table(vertcat(classNameCol{:}), T, FPR, TPR, 'VariableNames', ...
                {'ClassName', 'Threshold', 'FalsePositiveRate', 'TruePositiveRate'});
        end

        function obj = computeAddMetricsFromConfusionMetrics(obj, metrics, metricNames, fixedMetricOutput)
            % This function computes the value for any additional requested
            % metrics using the confusion metrics
            if nargin < 4
                % Don't have fixedMetricOutput, assume empty
                fixedMetricOutput = [];
            end
            tp = obj.getConfusionMetric('TP');
            fp = obj.getConfusionMetric('FP');
            tn = obj.getConfusionMetric('TN');
            fn = obj.getConfusionMetric('FN');
            numMetrics = numel(metrics);
            metricVals = ones(size(obj.Metrics, 1), numMetrics, 'like', tp);
            for i = 1:numMetrics
                if strcmpi(metricNames{i}, obj.FixedMetric.FullName)
                    % We already have computed the fixed metric
                    metricVals(:,i) = fixedMetricOutput;
                elseif isa(metrics{i}, 'function_handle')
                    % Custom metric - there is separate logic to handle
                    % calling the custom function in safe way
                    metricVals(:,i) = obj.evaluateCustomMetric(metrics{i}, tp, fp, tn, fn);
                else
                    % Known named parameter
                    switch metricNames{i}
                        % The first four metrics are already stored in
                        % ConfusionMetrics - add them directly, and remove
                        % them from the ConfusionMetrics table
                        case 'TruePositives'
                            metricVals(:,i) = tp;
                            obj.ConfusionMetrics.TP = [];
                        case 'FalseNegatives'
                            metricVals(:,i) = fn;
                            obj.ConfusionMetrics.FN = [];
                        case 'FalsePositives'
                            metricVals(:,i) = fp;
                            obj.ConfusionMetrics.FP = [];
                        case 'TrueNegatives'
                            metricVals(:,i) = tn;
                            obj.ConfusionMetrics.TN = [];

                            % The rest of the metrics are computed from the
                            % confusion metrics
                        case 'SumOfTrueAndFalsePositives'
                            metricVals(:,i) = tp+fp;
                        case 'RateOfPositivePredictions'
                            metricVals(:,i) = (tp + fp) ./ (tp + fn + fp + tn);
                        case 'RateOfNegativePredictions'
                            metricVals(:,i) = (tn + fn) ./ (tp + fn + fp + tn);
                        case 'Accuracy'
                            metricVals(:,i) = (tp + tn) ./ (tp + fn + fp + tn);
                        case 'FalseNegativeRate'
                            metricVals(:,i) = fn ./ (tp + fn);
                        case 'TrueNegativeRate'
                            metricVals(:,i) = tn ./ (tn + fp);
                        case 'PositivePredictiveValue'
                            metricVals(:,i) = tp ./ (tp + fp);
                        case 'NegativePredictiveValue'
                            metricVals(:,i) = tn ./ (tn + fn);
                        case 'ExpectedCost'
                            for j = 1:obj.NumClasses
                                curRange = obj.ClassRangeIdx(j,1):obj.ClassRangeIdx(j,2);
                                curTP = tp(curRange);
                                curFN = fn(curRange);
                                curFP = fp(curRange);
                                curTN = tn(curRange);
                                cost = obj.NormalizedCost{j};
                                metricVals(curRange,i) = ((curTP*cost(1,1) + curFN*cost(1,2)) + (curFP*cost(2,1) + curTN*cost(2,2))) ...
                                    ./ (curTP + curFN + curFP + curTN);
                            end
                    end
                end
            end

            if ~isempty(metricVals)
                obj.Metrics = [obj.Metrics, array2table(metricVals, 'VariableNames', metricNames)];
            end
        end

        function obj = computeAddMetricsFromPerfcurve(obj, scores, addMetrics, addMetricsFullNames, fixedMetricOutput)
            % This function computes additional metrics through perfcurve.
            % This is used when a user requests confidence intervals
            if nargin < 5
                % No fixed metric
                fixedMetricOutput = [];
            end

            % Set up initial metrics table to be added to obj.Metrics
            % Populate with 0s for now, re-fill with actual values as we
            % iterate through metrics
            numMetrics = numel(addMetrics);
            metricTable = array2table(zeros(size(obj.Metrics, 1), numMetrics), ...
                'VariableNames', addMetricsFullNames);

            if strcmpi(obj.FixedMetric.FullName, "Thresholds")
                % Fixed relative to a threshold - we can compute the
                % metrics in pairs
                for i = 1:2:numMetrics
                    sharedArgs = [{'TVals', obj.FixedMetric.Values, 'XCrit'}, ...
                        addMetrics(i), obj.ConfidenceIntervalArgs];
                    if i < numMetrics
                        % Haven't hit the end of the list - keep getting
                        % metrics in pairs
                        [~, xMetric, yMetric] = obj.perfcurveWrapper(scores, sharedArgs{:}, 'YCrit', addMetrics{i+1});
                        metricTable.(addMetricsFullNames{i+1}) = yMetric;
                    else
                        % Hit the end of the list - this happens with an
                        % odd number of metrics
                        [~, xMetric] = obj.perfcurveWrapper(scores, sharedArgs{:});
                    end
                    metricTable.(addMetricsFullNames{i}) = xMetric;
                end
            else
                % Fixed relative to some specific metric - compute metrics
                % one at a time
                % Remove the fixed metric from the list - don't need to
                % recompute it.
                isFixedMetric = ismember(obj.FixedMetric.FullName, addMetricsFullNames);
                if any(isFixedMetric)
                    metricTable.(addMetricsFullNames(isFixedMetric)) = fixedMetricOutput;
                    addMetricsFullNames(isFixedMetric) = [];
                    addMetrics(isFixedMetric) = [];
                end

                for i = 1:numel(addMetrics)
                    [~, ~, metric] = obj.perfcurveWrapper(scores, 'XCrit', obj.FixedMetric.Metric,...
                        'YCrit', addMetrics{i}, obj.ConfidenceIntervalArgs{:}, ...
                        'XVals', obj.FixedMetric.Values);
                    metricTable.(addMetricsFullNames{i}) = metric;
                end
            end
            obj.Metrics = [obj.Metrics, metricTable];
        end

        function obj = computeClassScales(obj)
            % This function computes and sets the class scales
            % wN = TP + FN, wP = TN + FP. The wN and wP values are
            % the weighted total of positive and negative classes, and
            % are constant for all confusion metric values
            wN = obj.ConfusionMetrics.TN(obj.ClassRangeIdx(:,1)) + ...
                obj.ConfusionMetrics.FP(obj.ClassRangeIdx(:,1));
            wP = obj.ConfusionMetrics.TP(obj.ClassRangeIdx(:,1)) + ...
                obj.ConfusionMetrics.FN(obj.ClassRangeIdx(:,1));

            obj.ClassScales = ones(obj.NumClasses, 2);
            if obj.NumClasses == 1
                prior = obj.NormalizedPrior{1};
                if strcmpi(prior, 'empirical')
                    % If the prior was 'empirical' then the class scales remain
                    % at [1 1].
                    return
                end
                % Otherwise, compute the prior from wN and wP
                if strcmpi(prior, 'uniform')
                    prior = [1 1];
                end
                obj.ClassScales = [wN * prior(1), wP * prior(2)];
            else
                for i = 1:obj.NumClasses
                    prior = obj.NormalizedPrior{i};
                    obj.ClassScales(i,:) = [wN(i) * prior(1), wP(i) * prior(2)];
                end
            end
            obj.ClassScales = obj.ClassScales ./ sum(obj.ClassScales,2);
        end

        function confMetric = getConfusionMetric(obj, metric)
            % This function serves as a wrapper to getting a specified
            % confusion metric. In order to save memory, the metric can
            % live in the internal ConfusionMetrics table, or it can live
            % in the public Metrics table (if a user has requested it).
            if ismember(metric, obj.ConfusionMetrics.Properties.VariableNames)
                confMetric = obj.ConfusionMetrics.(metric);
            else
                confMetric = obj.Metrics.(getNameFromMetric(metric));
            end
        end

        function metric = evaluateCustomMetric(obj, metricFcn, tp, fp, tn, fn)
            % This function computes a custom metric using a given function handle
            metric = ones(size(tp(:,1)));
            try
                for i = 1:obj.NumClasses
                    cost = obj.NormalizedCost{i};
                    currRange = obj.ClassRangeIdx(i,1):obj.ClassRangeIdx(i,2);
                    % TP etc are already scaled, but custom function expects unscaled,
                    % so unscale them
                    scales = obj.ClassScales(i,:);
                    metric(currRange) = arrayfun(@(a,b,c,d) metricFcn([a,b;c,d], obj.ClassScales(i,:), cost), ...
                        tp(currRange)/scales(1),fn(currRange)/scales(1),fp(currRange)/scales(2),tn(currRange)/scales(2));
                end
            catch ME
                % Trying to compute the custom function failed - add some extra
                % context, and then let the original error propagate
                fcnFailedError = MException(message('stats:rocmetrics:CustomMetricFcnFailed'));
                throw(addCause(fcnFailedError, ME));
            end
        end

        function [microX,microY,microT,microAUC] = microAverageCurve(obj, scores)
            % Compute micro-average curve
            % This is done by unraveling the scores and labels together,
            % and treating it as a binary class problem - either the
            % classifier gave the correct answer, or it did not

            % Rescale the scores, and unravel cross-validated input into a
            % single set of data
            if obj.IsCrossvalidated
                scores  = vertcat(scores{:});
                labels  = vertcat(obj.PrivLabels{:});
                weights = vertcat(obj.Weights{:});
            else
                labels  = obj.PrivLabels;
                weights = obj.Weights;
            end
            weights = repmat(weights, obj.NumClasses, 1);

            % Create micro-average labels. The label is a logical, with 1
            % for if the classifier gave the correct answer (regardless of
            % which class), and 0 otherwise
            microLabels = zeros(size(labels,1)*obj.NumClasses, 1);
            scoreSize = size(scores);
            for i = 1:obj.NumClasses
                idxRange = sub2ind(scoreSize, 1, i):sub2ind(scoreSize, scoreSize(1), i);
                microLabels(idxRange) = (labels == obj.PrivClassNames(i));
            end

            % Use perfcurve to compute the values for the new binary
            % ROC curves are not impacted by a given cost or prior, so we
            % do not pass those arguments to perfcurve
            [microX, microY, microT, microAUC] = ...
                perfcurve(microLabels, scores(:), true, 'Weights', weights,...
                'UseNearest', obj.UseNearest, 'ProcessNaN', obj.ProcessNaN);
        end

        function [fprMacro,tprMacro,tMacro,aucMacro] = macroAverageCurve(obj, weighted, scores)
            % This function computes the macro average curve
            % It takes in an additional argument, weighted, which indicates
            % if the average is weighted, and it takes in the rescaled
            % scores

            % For weighted curves, we take a mean weighted by the prior.
            % For unweighted curves, we take a normal mean
            if weighted
                weights = obj.Prior';
            else
                weights = ones(obj.NumClasses, 1) / obj.NumClasses;
            end

            fprFixed = strcmp(obj.FixedMetric.FullName, "FalsePositiveRate");
            tprFixed = strcmp(obj.FixedMetric.FullName, "TruePositiveRate");
            % Compute the curves by relying on perfcurve
            % We first need to determine which metric is the one which is
            % 'fixed', as the macro average is computed relative to that
            % fixed metric.
            if fprFixed
                basisData = unique(obj.Metrics.FalsePositiveRate(:,1));                     
            elseif tprFixed
                basisData = unique(obj.Metrics.TruePositiveRate(:,1));
            else
                % Either the thresholds are fixed, or some other metric is
                % held fixed. The distinction doesn't matter here, and we
                % treat either case as if the thresholds were fixed
                basisData = unique(obj.Metrics.Threshold(:,1));
                hasRejectAllThresh = (obj.Metrics.Threshold(1,1) == obj.Metrics.Threshold(2,1)) && ...
                                     (isequal(obj.Metrics.ClassName(1,1), obj.Metrics.ClassName(2,1)));

                if hasRejectAllThresh
                    % The curve has the reject-all point - ensure that the
                    % macro-average also has this reject-all point
                    basisData = [basisData; basisData(end) + eps(basisData(end))];
                end
            end

            % Set up data. interps are 2 of TPR, FPR, and Thresholds
            macroSize = numel(basisData);
            macroInterp1 = zeros(macroSize, obj.NumClasses, 'like', obj.Metrics.FalsePositiveRate);
            macroInterp2 = macroInterp1;

            obsWeights = obj.Weights;
            labels = obj.PrivLabels;
            if obj.IsCrossvalidated
                % Pull out the data from the cross-validation folds,
                % and stack it together - we don't want to get
                % cross-validation confidence intervals
                scores = vertcat(scores{:});
                labels = vertcat(labels{:});
                obsWeights = vertcat(obsWeights{:});
            end
            for i = 1:obj.NumClasses
                % Call perfcurve for each class
                % We are computing the FPR and TPR, which are invariant to
                % a specific cost and prior, so we don't need to watch for
                % those args
                curScores = scores(:,i);
                curClass = obj.PrivClassNames(i);

                % Set UseNearest to off always. This ensures that we get
                % curves of fixed size and at points exactly specified by
                % the basisData (e.g., we will get a TPR for all threshold
                % values, even if the TPR values are not unique)
                if fprFixed
                    % Ignore first X output
                    [~, macroInterp1(:,i), macroInterp2(:,i)] = ...
                        perfcurve(labels, curScores, curClass, 'XVals', ...
                        basisData, 'UseNearest', 'off', 'ProcessNaN', obj.ProcessNaN, ...
                        'Weights', obsWeights);
                elseif tprFixed
                    % Ignore first X output, which is now TPR since the
                    % XCrit and YCrit values are flipped
                    [~, macroInterp1(:,i), macroInterp2(:,i)] = ...
                        perfcurve(labels, curScores, curClass, 'Xvals', ...
                        basisData, 'UseNearest', 'off', 'Xcrit', ...
                        'tpr', 'YCrit', 'fpr', 'ProcessNaN', obj.ProcessNaN, ...
                        'Weights', obsWeights);
                else
                    [macroInterp1(:,i), macroInterp2(:,i)] = ...
                        perfcurve(labels, curScores, curClass, 'TVals', ...
                        basisData, 'UseNearest', 'off', 'ProcessNaN', obj.ProcessNaN, ...
                        'Weights', obsWeights);
                end
            end

            % We now have a set of curves of equal size
            % Multiply each set of curves against the weight to get the
            % average
            if fprFixed
                fprMacro = basisData;
                tprMacro = macroInterp1 * weights;
                tMacro = macroInterp2 * weights;
            elseif tprFixed
                fprMacro = macroInterp1 * weights;
                tprMacro = basisData;
                tMacro = macroInterp2 * weights;
            else
                fprMacro = macroInterp1 * weights;
                tprMacro = macroInterp2 * weights;
                tMacro = flipud(basisData); % Thresholds is stored in descending order
                if hasRejectAllThresh
                    % Set the thresholds equal like perfcurve does
                    tMacro(1) = tMacro(2);
                end
            end

            % Get the AUC by trapezoid integration
            if numel(tprMacro) < 2
                % Not enough points to compute an area
                aucMacro = 0;
            else
                aucMacro = trapz(fprMacro, tprMacro);
            end
        end
    end
end

%%% VALIDATION FUNCTIONS %%%
function [scores, labels, classNames, NVPairs, addFullNames, fixedMetric] = ...
    validateInputs(scores, labels, classNames, NVPairs)
% This function handles validation of all inputs. It dispatches to specific
% validation functions for some inputs, and returns the validated (and in
% some cases modified) inputs
NVPairs.UseNearestNeighbor = internal.stats.parseOnOff(NVPairs.UseNearestNeighbor, 'UseNearestNeighbor');
NVPairs.NaNFlag = validatestring(NVPairs.NaNFlag, ["omitnan", "includenan"]);
[scores, labels, NVPairs.Weights] = ...
    validateScoresLabelsAndWeights(scores, labels, NVPairs.Weights, NVPairs.NaNFlag);
classNames = validateClassNames(classNames);
numClasses = size(classNames, 1);
validateMultiClassSizes(scores, numClasses);
NVPairs.Cost = validateCost(NVPairs.Cost, numClasses);
NVPairs.Prior = validateAndConvertPrior(NVPairs.Prior, NVPairs.Weights, labels, classNames);
[NVPairs.AdditionalMetrics, addFullNames] = validateAdditionalMetrics(NVPairs.AdditionalMetrics, ...
    'AdditionalMetrics', mfilename);
[NVPairs.AdditionalMetrics, addFullNames] = removeROCMetrics(NVPairs.AdditionalMetrics, addFullNames);
[fullFixedName, origMetric] = validateFixedMetric(NVPairs.FixedMetric, NVPairs.AdditionalMetrics, addFullNames);
fixedMetric = struct('Metric', origMetric, 'Values', NVPairs.FixedMetricValues, ...
    'FullName', fullFixedName);
end

function validateStudSE(studSE)
% NumBootstrapsStudentizedSE must be a positive integer, representing the 
% number of bootstrap samples to take
if ~isempty(studSE)
    validateattributes(studSE, {'double', 'single'}, ...
        {'positive', 'real', 'integer', 'nonnan', 'scalar'}, mfilename, 'NumBootstrapsStudentizedSE')
end
end

function validateFixedValues(fixedVals)
% This function validates the fixed value input
% Fixed value can be:
% 1. An empty char/numeric
% 2. The string 'all'
% 3. A real numeric vector
fixedVals = convertStringsToChars(fixedVals);
if ~isempty(fixedVals)
    if ischar(fixedVals)
        % Only allow the char 'all'
        if ~strcmpi(fixedVals, 'all')
            error(message('stats:rocmetrics:BadFixedVals'))
        end
    else
        validateattributes(fixedVals, {'double', 'single'}, ...
            {'vector', 'real', 'nonsparse'}, mfilename, 'FixedMetricValues');
    end
end
end

function validateMultiClassSizes(scores, numClasses)
% This function validates that scores and classNames align in size. The
% number of classes should equal the number of columns of scores.
if iscell(scores)
    % Cross-validated, validate the number of columns within cells.
    numCols = cellfun(@(x) size(x, 2), scores);
else
    numCols = size(scores, 2);
end
if any(numCols(1) ~= numCols)
    % Relevant for cross-validated scores - ensure all folds have the same
    % number of columns
    error(message('stats:rocmetrics:InconsistentScoreSizes'))
elseif any(numCols ~= numClasses)
    error(message('stats:rocmetrics:ScoresSizeBad'));
end
end

function [scores, labels, weights] = validateScoresLabelsAndWeights(scores, labels, weights, useNaN)
% This function validates basic aspects of scores, labels, and weights, and
% prepares them to be passed to perfcurve. More validation is done in
% perfcurve

if iscell(scores)
    % Cross-validated input

    % Check size of scores
    if ~isvector(scores)
        error(message('stats:perfcurve:CVScoresNotVector'));
    end
    scores = scores(:);
    if numel(scores)<2
        error(message('stats:perfcurve:CVScoresTooShort'));
    end

    % Check that the other inputs are all cell arrays as well
    if ~iscell(labels)
        error(message('stats:perfcurve:CVLabelsNotCell'));
    end
    if ~isvector(labels)
        error(message('stats:perfcurve:CVLabelsNotVector'));
    end
    labels = labels(:);

    noWeights = isempty(weights);
    if ~noWeights
        if ~iscell(weights)
            error(message('stats:perfcurve:CVWeightsNotCell'));
        elseif ~isvector(weights)
            error(message('stats:perfcurve:CVWeightsNotVector'));
        end
    end
    weights = weights(:);

    % Validate labels and weights sizes
    numFolds = numel(scores);
    if noWeights
        weights = cell(numFolds, 1);
    elseif numel(weights) ~= numFolds
        error(message('stats:perfcurve:CVWeightsWithNonmatchingLength'));
    end

    if numel(labels) ~= numFolds
        error(message('stats:perfcurve:CVLabelsWithNonmatchingLength'));
    end

    % Validate label is the same type across all folds
    labelType = class(labels{1});
    hasSameType = cellfun(@(l) strcmp(class(l), labelType), labels);
    if ~all(hasSameType)
        error(message('stats:perfcurve:CVLabelsWithDifferentTypes'));
    end

    % Iterate through each fold, verify type and sizes within them
    for i = 1:numFolds
        % Scores
        curScores = scores{i};
        if ~isfloat(curScores) || ~isreal(curScores) || isempty(curScores) ...
                || ~ismatrix(curScores)
            error(message('stats:perfcurve:CVScoresWithBadElements'));
        end

        if size(curScores, 1) == 1
            curScores = curScores(:);
        end
        numScoreRows = size(curScores, 1);

        % Labels
        curLabels = validateLabelTypeInput(labels{i}, 'labels');
        if ~isequal(size(curLabels), [numScoreRows, 1])
            error(message('stats:perfcurve:CVLabelsNotMatchedToScores'));
        end

        % Weights
        if noWeights
            curWeights = ones(numScoreRows, 1, 'like', curScores);
        else
            curWeights = weights{i};
            if ~isfloat(curWeights) || ~isvector(curWeights) || ~isreal(curWeights) ...
                    || any(curWeights < 0)
                error(message('stats:perfcurve:CVWeightsNotNumeric'))
            end
            curWeights = curWeights(:);

            if numel(curWeights) ~= numScoreRows
                error(message('stats:perfcurve:CVWeightsNotMatchedToScores'));
            end
        end

        % Remove missing labels and NaN/0 weights
        % Set back values. Convert labels to ClassLabel format for easier
        % usage later on
        [curLabels, curScores, curWeights] = removeMissing(curLabels, curScores, ...
            curWeights, useNaN);
        scores{i} = curScores;
        weights{i} = curWeights;
        labels{i} = classreg.learning.internal.ClassLabel(curLabels);
    end
else
    % Validate scores
    validateattributes(scores, {'double', 'single'}, {'real', 'nonempty', '2d'}, ...
        mfilename, 'scores');
    if size(scores, 1) == 1
        scores = scores(:);
    end
    numScoreRows = size(scores,1);

    % Validate labels
    labels = validateLabelTypeInput(labels, 'labels');
    if numel(labels) ~= numScoreRows
        error(message('stats:perfcurve:ScoresAndLabelsDoNotMatch'));
    end

    % Validate weights
    validateattributes(weights, {'double', 'single'}, {'real', 'nonnegative', 'nonsparse'}, ...
        mfilename, 'Weights')
    if isempty(weights)
        % Generate default weights
        weights = ones(numScoreRows, 1, 'like', scores);
    else
        % Verify given weight size
        if ~isvector(weights) || (numel(weights) ~= numScoreRows)
            error(message('stats:perfcurve:WeightsNotMatchedToScores'));
        end
        weights = weights(:);
    end

    % Remove missing data and 0 weights
    % Convert labels to ClassLabel format for ease of use
    [labels, scores, weights] = removeMissing(labels, scores, weights, useNaN);
    labels = classreg.learning.internal.ClassLabel(labels);
end
end

function input = validateLabelTypeInput(input, paramName)
% This function validates the type of 'label-type' inputs, like classNames
% and labels. More detailed validation, beyond type, is done in perfcurve.
if ischar(input)
    input = cellstr(input);
end
validateattributes(input, {'cell', 'string', 'logical', 'numeric', 'categorical'},...
    {'vector', 'nonempty'}, mfilename, paramName);
if iscell(input) && ~iscellstr(input)
    % Given a cell array, but not a cellstr
    error(message('stats:classreg:learning:internal:ClassLabel:ClassLabel:UnknownType'));
end
input = input(:);
end

function classNames = validateClassNames(classNames)
% This function validates that the classNames input is of the correct type,
% and it is well formatted. classNames must be unique, and not contain any
% missing entries
classNames = validateLabelTypeInput(classNames, 'classNames');

% Validate there are no missing entries, and that the input is unique
if any(ismissing(classNames))
    error(message('stats:rocmetrics:MissingClassNames'));
elseif numel(unique(classNames)) ~= numel(classNames)
    error(message('stats:rocmetrics:DuplicateClassNames'));
end
classNames = classreg.learning.internal.ClassLabel(classNames);
end

function prior = validateAndConvertPrior(prior, weights, labels, classNames)
% This function validates the prior inputs, and converts it to a numeric
% vector, if it isn't numeric already

% Validate basic type - prior can only be a char/string, double, or single
validateattributes(prior, {'char', 'string', 'double', 'single'}, {}, mfilename, 'Prior');

% Min prior size is 2. In the case of a binary class problem, the prior
% represents the positive class and all other negative classes
priorSize = max(numel(classNames), 2);

% Determine which prior was given (one of the allowed strings, or a numeric
% prior)
if isnumeric(prior)
    % Validate the prior is properly specified
    validateattributes(prior, {'double', 'single'}, {'nonempty', 'real', ...
        'nonnan', 'positive', 'finite', 'vector', 'numel', priorSize}, ...
        mfilename, 'Prior');
    prior = prior(:) / sum(prior);
elseif strcmpi(prior, 'uniform')
    prior = ones(priorSize, 1) / priorSize;
elseif strcmpi(prior, 'empirical')
    % Unravel cross-validation folds, if necessary
    if iscell(weights)
        weights = vertcat(weights{:});
        labels = vertcat(labels{:});
    end

    % Get logical matrix of class membership, and weight with given weights
    [~, grps] = ismember(labels, classNames);
    if numel(classNames) == 1
        % Binary class setup - if a label isn't the positive class, it counts
        % towards the negative classes count
        classCounts = [grps, ~grps];
    else
        classCounts = (1:priorSize) == grps;
    end

    % Weight counts, then sum to get the total class counts, which then are
    % used to construct prior
    weightedClassCounts = classCounts .* weights;
    totalClassCounts = sum(weightedClassCounts, 1);
    prior = totalClassCounts(:) / sum(totalClassCounts);
else
    % Char, but not any of the valid options
    error(message('stats:perfcurve:BadPriorString'))
end
end

function cost = validateCost(cost, expSize)
% This function validates basic aspects of the Cost input
expSize = max(expSize, 2);
if isempty(cost)
    cost = ones(expSize) - eye(expSize);
else
    % Already given a cost - verify it is valid
    validateattributes(cost, {'double', 'single'}, ...
        {'real', 'size', [expSize, expSize], 'nonnan', ...
        'finite', 'nonnegative', 'nonsparse'}, ...
        mfilename, 'Cost');
    if any(diag(cost)~=0)
        error(message('stats:rocmetrics:NonzeroDiagCost'));
    end
end
end

function [addMetrics, fullNames] = validateAdditionalMetrics(addMetrics, paramName, funcName)
% This function validates the AdditionalMetrics NV pair. This can be a
% string, char, or cellstr of metric names, in addition to a cell array of
% combined metric names and function handles for custom metrics
fullNames = [];
addMetrics = convertStringsToChars(addMetrics);

if ~isempty(addMetrics)
    % If given just a single function handle or char, convert to a cell array
    if isa(addMetrics, 'function_handle')
        addMetrics = {addMetrics};
    elseif ischar(addMetrics)
        addMetrics = cellstr(addMetrics);
    end

    if ~iscell(addMetrics)
        % Given something other than a cellstr, char or func handle
        error(message('stats:rocmetrics:BadAdditionalMetrics', paramName))
    end
    cellfun(@(am) validateattributes(am, {'char', 'string', 'function_handle'},...
        {'vector'}, funcName, paramName), addMetrics);

    addMetrics = addMetrics(:);
    % Get the list of full names of each metric - ensures we don't have
    % duplicates in the form of different names, e.g., 'fpr' and 'fall'
    fullNames = cellfun(@getNameFromMetric, addMetrics);

    % For any custom metrics, add numeric suffix if indicated to do so
    isFcnHandle = cellfun(@(m) isa(m, 'function_handle'), addMetrics);
    fullNames(isFcnHandle) = fullNames(isFcnHandle) + (1:nnz(isFcnHandle))';

    % Remove duplicates
    [fullNames, inds] = unique(fullNames, 'stable');
    addMetrics = addMetrics(inds);
end
end

function [addMetrics, fullNames] = removeROCMetrics(addMetrics, fullNames)
% This function removes any ROC metrics given in 'AdditionalMetrics',
% as they get computed by default
if ~isempty(addMetrics)
    isROC = ismember(fullNames, ["FalsePositiveRate", "TruePositiveRate"]);
    addMetrics(isROC) = [];
    fullNames(isROC) = [];
    if any(isROC)
        warning(message('stats:rocmetrics:ROCAlreadyComputed'))
    end
end
end

function [fullName, origMetric] = validateFixedMetric(fixedMetric, addMetrics, addMetricsFullNames)
% This function validates that fixedMetric, if specified, is a valid metric
% to fix. It must be a metric that appears in addMetrics, tpr, fpr, or
% 'thresholds'.
fixedMetric = convertStringsToChars(fixedMetric);
if strcmpi(fixedMetric, 'Thresholds')
    % Fixed relative to some threshold
    fullName = "Thresholds";
elseif argMatchesCustomMetricString(fixedMetric)
    % User wants to hold a custom metric fixed.
    % The full metric name in this case is already given
    fullName = fixedMetric;
else
    % Some other metric. Get the full name
    fullName = getNameFromMetric(fixedMetric);
end

% Compare against all the specified metrics with AdditionalMetrics, along
% with Thresholds, TPR, and FPR, which can always be held fixed
allAllowedFixed = [addMetricsFullNames(:); "TruePositiveRate"; "FalsePositiveRate"; "Thresholds"];
foundInAllowedList = ismember(allAllowedFixed, fullName);
if ~any(foundInAllowedList)
    % Want to compute curves to a metric that isn't getting computed
    % Error with the supported metrics for this function call
    allowedString = strjoin(allAllowedFixed, ", ");
    error(message('stats:rocmetrics:BadFixedMetric', allowedString))
end

% Get the original name. This is relevant for custom metrics, where
% perfcurve will expect a function handle rather than the name of a custom
% metric
if isempty(addMetrics)
    origMetric = fixedMetric;
else
    allShort = [addMetrics(:); {'tpr';'fpr';'thresholds'}];
    origMetric = allShort{foundInAllowedList};
end
end

%%% GENERAL HELPER FUNCTIONS %%%
function [labels, scores, weights] = removeMissing(labels, scores, weights, useNaN)
% This function removes missing values from scores, labels, and weights
% A missing row in any of these arguments means the row needs to be deleted
% for all of them
% Remove missing labels and NaN/0 weights
toRemove = ismissing(labels);
toRemove = toRemove | isnan(weights) | (weights == 0);
if strcmpi(useNaN, 'omitnan')
    % Remove NaN scores
    toRemove = toRemove | any(isnan(scores), 2);
end
scores(toRemove,:) = [];
weights(toRemove) = [];
labels(toRemove) = [];

if isempty(scores)
    % All missing scores/labels/weights - no data to compute a curve for
    error(message('stats:rocmetrics:AllMissingData'));
end
end

function doesMatch = argMatchesCustomMetricString(arg)
% This function tests if the given arg matches a custom metric
% name.
% This regex matches CustomMetric#, where # is at least 1 digit, but it
% can be as many digits as the user likes
arg = convertCharsToStrings(arg);
doesMatch = isStringScalar(arg) && ~isempty(regexpi(arg, '^CustomMetric\d+$', 'once'));
end

function fullName = getNameFromMetric(metric)
% This function converts from the list of metrics allowed (e.g., tpr, fpr)
% into the full name of that metric (e.g.,TruePositiveRate, FalsePositiveRate)
if isa(metric, 'function_handle')
    fullName = "CustomMetric";
else
    switch lower(metric)
        case {"tp", "truepositives"}
            fullName = "TruePositives";
        case {"fn", "falsenegatives"}
            fullName = "FalseNegatives";
        case {"fp", "falsepositives"}
            fullName = "FalsePositives";
        case {"tn", "truenegatives"}
            fullName = "TrueNegatives";
        case {"tp+fp", "sumoftrueandfalsepositives"}
            fullName = "SumOfTrueAndFalsePositives";
        case {"rpp", "rateofpositivepredictions"}
            fullName = "RateOfPositivePredictions";
        case {"rnp", "rateofnegativepredictions"}
            fullName = "RateOfNegativePredictions";
        case {"accu", "accuracy"}
            fullName = "Accuracy";
        case {"tpr", "sens", "reca", "truepositiverate"}
            fullName = "TruePositiveRate";
        case {"fnr", "miss", "falsenegativerate"}
            fullName = "FalseNegativeRate";
        case {"fpr", "fall", "falsepositiverate"}
            fullName = "FalsePositiveRate";
        case {"tnr", "spec", "truenegativerate"}
            fullName = "TrueNegativeRate";
        case {"ppv","prec", "positivepredictivevalue"}
            fullName = "PositivePredictiveValue";
        case {"npv", "negativepredictivevalue"}
            fullName = "NegativePredictiveValue";
        case {"ecost", "expectedcost"}
            fullName = "ExpectedCost";
        otherwise
            error(message('stats:rocmetrics:BadMetric', metric))
    end
end
end

function transformedScores = rescaleScores(scores)
% This function transforms scores for multi-class problems. It breaks ties
% in a manner consistent with classification objects.
if iscell(scores)
    % Cross-validated data - map the rescale function across each fold
    transformedScores = cellfun(@(s) rescaleScores(s), scores, 'UniformOutput', false);
else
    if size(scores, 2) > 1
        transformedScores = scores;
        inds = 1:size(scores, 2);
        for i = inds
            % For each class, rescale the scores of that class relative to the
            % rest of the classes. This moves the decision boundary to 0, and
            % thresholds will range from -1 to 1
            isCurrInd = (inds == i);
            transformedScores(:,i) = scores(:,isCurrInd) - max(scores(:,~isCurrInd), [], 2);

            % Break any ties in a way consistent with how classification models break
            % ties. In the case of a tie, the class chosen is done so in order
            % of the classes.
            if i > 1
                ties = find(transformedScores(:,i) == 0);
                earlierFound = any(scores(ties,1:i-1) == scores(ties,i), 2);
                ties = ties(earlierFound);
                transformedScores(ties,i) = -realmin;
            end
        end
    else
        transformedScores = scores;
    end
end
end

%%% PLOT UTILITIES %%%
%%% Validation Functions %%%
function [obj, metricFull] = validatePlotMetrics(obj, metric, metricName)
% This function validates a metric that is to be plotted. The given
% metric must satisfy two criteria:
% 1. Must be a string/char/function handle
% 2. If string/char, must be a known named metric that lives in the obj.Metrics
% table or can be computed via addMetrics
validateattributes(metric, {'char','string','function_handle'}, {}, 'plot', metricName);

% Check if the input is of the form "CustomMetric#", which indicates
% wanting to plot a custom metric within the table.
if argMatchesCustomMetricString(metric)
    if ~ismember(metric, obj.Metrics.Properties.VariableNames(3:end))
        % Named 'CustomMetric#' but that doesn't actually appear in the
        % table
        error(message('stats:rocmetrics:BadMetric', metric))
    end
    metricFull = metric;
else
    % Otherwise pass the metric to addMetrics
    obj = obj.addMetrics(metric);

    % Get the name. If the metric is a function handle, it will live at the
    % end of the metrics table, since it was just added
    if isa(metric, 'function_handle')
        metricFull = obj.Metrics.Properties.VariableNames{end};
    else
        % Otherwise, convert to full name
        metricFull = getNameFromMetric(metric);
    end
end
end

function classNameInds = validateAndExtractClassNames(classNames, obj)
% This functions validates the input positive classes, if any are given,
% and returns the indices of the classes to use
if ~isempty(classNames)
    % Convert to ClassLabel, which will handle type validation
    if ischar(classNames)
        classNames = cellstr(classNames);
    end
    classNames = classreg.learning.internal.ClassLabel(classNames);

    % Get membership
    [~, classNameInds] = ismember(classNames, obj.PrivClassNames);
    notMember = find(~classNameInds,1);
    if ~isempty(notMember)
        badClass = char(classNames(notMember));
        error(message('stats:rocmetrics:ClassNameNotFoundPlot', badClass));
    end
    classNameInds = classNameInds';
else
    classNameInds = [];
end
end

%%% Other Plot Helpers %%%
function inRange = isInFixedRange(metric)
% This function determines if the given metric lives in a fixed [0 1]
% range.
fixedRangeMetrics = {'RateOfPositivePredictions', 'RateOfNegativePredictions', ...
    'Accuracy', 'TruePositiveRate', 'FalseNegativeRate', 'FalsePositiveRate', ...
    'TrueNegativeRate', 'PositivePredictiveValue', 'NegativePredictiveValue'};
inRange = any(strcmpi(metric, fixedRangeMetrics));
end

function [chartArgs, curveName] = makeChartArgs(ind, xMetric, yMetric, obj, isROC)
% This function constructs the arguments for rocmetrics from the
% rocmetrics object, and places them into a cell array.

% Get DisplayName
curveName = char(obj.PrivClassNames(ind));
dispName = curveName;
if isROC
    dispName = [dispName, ' (AUC = ' num2str(round(obj.AUC(1,ind), 4)) ')'];
end

% Get data table and indices for the current class
data = obj.Metrics;
curRange = obj.ClassRangeIdx(ind,1):obj.ClassRangeIdx(ind,2);
chartArgs = {
    'XData_I'           , data.(xMetric)(curRange,:), ...
    'XAxisMetric_I'     , xMetric, ...
    'YData_I'           , data.(yMetric)(curRange,:), ...
    'YAxisMetric_I'     , yMetric, ...
    'Thresholds_I'      , data.Threshold(curRange,:), ...
    'DisplayName'       , dispName};
end

function [parent, obj, plotArgs] = getObjAndAxesArgs(plotArgs)
% This function parses out the object (rocmetrics) and an axes, if one is
% given. If one is not given, a default is constructed.

% Check for plot(ax,obj)
[parent, args] = axescheck(plotArgs{:});
obj = args{1};
plotArgs = args(2:end);

% Check for plot(obj,ax)
if isempty(parent)
    [parent, plotArgs] = axescheck(plotArgs{:});
end
end

function cax = createAndValidateAxes(parent)
% This function generates the axes on which to plot (if one is not given).
% It also validates the axes

% Parent may not be an axes handle. If it is empty or is an axes, pass to
% newplot. Otherwise, find the axes in the graphics tree via ancestor.
if isempty(parent) || isgraphics(parent,'axes')
    parent = newplot(parent);
    cax = parent;
else
    cax = ancestor(parent,'axes');
end

% Ensure that supplied parent is valid
if isempty(cax)
    shortname = regexprep(class(parent), '.*\.', '');
    error(message('MATLAB:handle_graphics:exceptions:HandleGraphicsException',...
        getString(message("MATLAB:HandleGraphics:hgError", 'ROCCurve', shortname))));
end
end

function gObj = plotAverageROC(gObj, obj, averageROC, ax, graphicsNVPairs)
% This function computes and plots an average ROC curve, if requested by
% the user
if ~isempty(averageROC)
    % Call average, let it handle input validation
    [xAvg, yAvg, tAvg, aucAvg] = obj.average(averageROC);
    averageROC = lower(averageROC);
    averageROC(1) = upper(averageROC(1));
    dispName = [averageROC '-average (AUC = ' num2str(aucAvg) ')'];
    chartArgs = {
        'XData_I'       , xAvg, ...
        'XAxisMetric_I' , "FalsePositiveRate", ...
        'YData_I'       , yAvg, ...
        'YAxisMetric_I' , "TruePositiveRate", ...
        'Thresholds_I'  , tAvg, ...
        'DisplayName'   , dispName, ...
        'Parent'        , ax
        };

    gObj(end+1) = mlearnlib.graphics.chart.ROCCurve(chartArgs{:}, ...
        graphicsNVPairs{:});
end
end

function lineObj = plotUnityLine(ax, showLine)
% This function handles plotting the [0,0] - [1,1] unity line, if it is
% requested
lineObj = [];
if showLine
    lineObj = matlab.graphics.chart.primitive.Line('XData', [0 1], 'YData', [0 1],...
        'Color', 'k', 'LineStyle', '--');
    lineObj.Parent = ax;
    hB = hggetbehavior(lineObj, 'datacursor');
    hB.Enable = 0;
    lineObj.Annotation.LegendInformation.IconDisplayStyle = 'off';
end
end

function addAxesLabelsAndLegend(ax, canModifyAxes, xMetric, yMetric, isROC)
% This function adds extra things to the axes, like labels and a legend.
% This is only done if it is known that we have an axes we can edit (i.e.,
% it isn't an axes that was given with 'hold on' active)
if canModifyAxes
    if isInFixedRange(xMetric) && isInFixedRange(yMetric)
        % Criteria both live in [0,1] - safe to square axis
        axis(ax, 'square');
    end
    axis(ax, 'padded');

    % Add labels
    xlabel(ax, makeLabelWithSpaces(xMetric))
    ylabel(ax, makeLabelWithSpaces(yMetric))
    if isROC
        title(ax, getString(message('stats:rocmetrics:ROCCurve_title')))
    else
        title(ax, getString(message('stats:rocmetrics:PerformanceCurve_title')));
    end

    % Place the legend. For ROC curves, the southeast position should be
    % empty. For precision-recall curves, the southwest position should be
    % empty. Otherwise, choose the best location automatically
    if isROC
        legend(ax, 'Location', 'southeast');
    elseif strcmp(xMetric, "TruePositiveRate") && strcmp(yMetric, "PositivePredictiveValue")
        % Precision-Recall curve
        legend(ax, 'Location', 'southwest')
    else
        legend(ax, 'Location', 'best');
    end
end
end

function label = makeLabelWithSpaces(metric)
% This function takes a given metric used in a performance plot and adds
% spacing to the metric name. It also removes any numbering from custom
% metrics
% Add a space before all capital letters except the first
metric = convertStringsToChars(metric);
label = [metric(1) regexprep(metric(2:end), '([A-Z])', ' $1')];

% Remove any numbers
label = replace(label, digitsPattern, '');
end

function gObj = addModelOperatingPoint(rocObj, ax, showOperating, gObj, operatingThresh, className)
% This function adds a Scatter object for the model operating point, if it
% has been requested
if showOperating
    % Create scatter object
    operatingInd = find(rocObj.Thresholds >= operatingThresh, 1, 'last');
    dispName = [className, ' ', getString(message('stats:rocmetrics:OperatingPoint_display'))];
    nextPlot = ax.NextPlot;
    hold(ax, 'on');
    scatterObj = scatter(ax, rocObj.XData(operatingInd), rocObj.YData(operatingInd), ...
        'filled', 'MarkerEdgeColor', rocObj.Color,'MarkerFaceColor', rocObj.Color, ...
        'SeriesIndex', rocObj.SeriesIndex, 'DisplayName', dispName, 'Marker', 'o');
    ax.NextPlot = nextPlot;

    % Add datatips
    datatipRows = scatterObj.DataTipTemplate.DataTipRows;
    datatipRows(1).Label = "FalsePositiveRate";
    datatipRows(2).Label = "TruePositiveRate";
    datatipRows(3).Label = "Threshold";
    datatipRows(3).Value = rocObj.Thresholds(operatingInd);
    scatterObj.DataTipTemplate.DataTipRows = datatipRows;
    gObj(end+1) = scatterObj;
end
end

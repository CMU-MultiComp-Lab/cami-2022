"""Pipeline for running predictive models and associated feature analyses."""
import itertools

import numpy as np
import sklearn.model_selection
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network

import analysis.session_features

from utils.log import log

# features to include in the predictive models
SCORE_FEATURES = {'pos': ['percept', 'power', 'perplexity'],
                  'neg': ['time', 'edit', 'restart']}
SCORES = ['pos', 'neg']

# parameter grids for the different models
SVM_PARAM_GRID = {'kernel': ['linear', 'rbf', 'poly'],
                  'C': [0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [0.1, 0.01, 0.001, 0.0001]}
MLP_PARAM_GRID = {'hidden_layer_sizes': [1, 5, 10, 50, 100, 500],
                  'activation': ['logistic', 'tanh', 'relu']}

HIDDEN_LAYERS = {'pos': 10, 'neg': 50}


def run_svm(X, y):
    """Run an exhaustive search over the parameter values defined in
    SVM_PARAM_GRID for an MLP estimator.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): The target values.

    Returns:
        sklearn.model_selection.GridSearchCV: The final model results.
    """
    log('Fitting SVM...')
    model = sklearn.model_selection.GridSearchCV(
        sklearn.svm.SVR(),
        SVM_PARAM_GRID,
        n_jobs=32)
    model.fit(X, y.ravel())
    log('SVM performance: R^2 = {}'
        .format(np.mean(model.cv_results_['mean_test_score'])))
    log('Best SVM params: {}'.format(model.best_params_))
    return model


def rank_features(X, y, features, params):
    """Run a greedy stepwise feature selection process on the MLP model.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): The target values.
        features (list of str): The list of features to select from.
        params (dict): Parameter settings that provided the best results on
            the cross-validated hold-out validation.
    """
    selected_features = []
    best_scores = []

    while len(selected_features) < len(features):

        results = []
        for feature in np.setdiff1d(features, selected_features):
            model = sklearn.neural_network.MLPRegressor(
                activation=params['activation'],
                hidden_layer_sizes=params['hidden_layer_sizes'],
                max_iter=50000)
            features_idx = [features.index(f)
                            for f in selected_features + [feature]]
            model.fit(X[:, features_idx], y.ravel())
            score = model.score(X[:, features_idx], y)
            results.append((feature, score))

        best_feature, best_score = max(results, key=lambda x: x[1])
        selected_features.append(best_feature)
        best_scores.append(best_score)

    log('Feature ordering: {}'.format(selected_features))
    diff_scores = np.insert(np.diff(best_scores), 0, best_scores[0])
    log('Incremental Pearson\'s r: {}'.format(diff_scores))


def run_mlp(X, y):
    """Run an exhaustive search over the parameter values defined in
    MLP_PARAM_GRID for an MLP estimator.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): The target values.

    Returns:
        sklearn.model_selection.GridSearchCV: The final model results.
    """
    log('Fitting MLP...')
    model = sklearn.model_selection.GridSearchCV(
        sklearn.neural_network.MLPRegressor(max_iter=50000),
        MLP_PARAM_GRID,
        n_jobs=32)
    model.fit(X, y.ravel())
    score = np.mean(model.cv_results_['mean_test_score'])
    log('MLP performance: R^2 = {}'
        .format(score))
    log('Best MLP params: {}'.format(model.best_params_))
    return model


def scale_data(data):
    """Scale the data by removing the mean and scaling to a unit variance.

    Args:
        data (np.ndarray): The data to scale.

    Returns:
        np.ndarray: The scaled data.
    """
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


def run_predictive_models(transcripts):
    """Primary pipeline through which to run all predictive models.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.
    """
    data = analysis.session_features.get_session_data(transcripts)
    for scale in SCORES:

        log('Fitting {} score results...'.format(scale))

        X = scale_data(data[SCORE_FEATURES[scale]].as_matrix())
        y = scale_data(data[scale].astype(float).as_matrix().reshape(-1, 1))

        run_svm(X, y)
        mlp_model = run_mlp(X, y)

        rank_features(X, y, SCORE_FEATURES[scale], mlp_model.best_params_)

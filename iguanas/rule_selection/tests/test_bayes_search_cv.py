import pytest
import pandas as pd
import numpy as np
from hyperopt import tpe, anneal
from sklearn.ensemble import RandomForestClassifier

from iguanas.rule_generation import RuleGeneratorDT, RuleGeneratorOpt
from iguanas.rule_optimisation import BayesianOptimiser
from iguanas.rules import Rules
from iguanas.metrics import FScore, JaccardSimilarity, Precision
from iguanas.rule_selection import SimpleFilter, CorrelatedFilter, GreedyFilter, BayesSearchCV
from iguanas.correlation_reduction import AgglomerativeClusteringReducer
from iguanas.rbs import RBSOptimiser, RBSPipeline
from iguanas.pipeline import LinearPipeline, ClassAccessor
from iguanas.rule_generation import RuleGeneratorDT
from iguanas.space import UniformFloat, UniformInteger, Choice

f1 = FScore(1)
js = JaccardSimilarity()
p = Precision()


@pytest.fixture
def _create_data():
    np.random.seed(0)
    X = pd.DataFrame({
        'A': np.random.randint(0, 2, 100),
        'B': np.random.randint(0, 10, 100),
        'C': np.random.normal(0.7, 0.2, 100),
        'D': (np.random.uniform(0, 1, 100) > 0.6).astype(int)
    })
    y = pd.Series((np.random.uniform(0, 1, 100) >
                  0.9).astype(int), name='label')
    sample_weight = (y+1)*10
    return X, y, sample_weight


@pytest.fixture
def _instantiate_classes():
    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    rg_dt = RuleGeneratorDT(
        metric=f1.fit,
        n_total_conditions=4,
        tree_ensemble=rf
    )
    rg_opt = RuleGeneratorOpt(
        metric=f1.fit,
        n_total_conditions=4,
        num_rules_keep=10,
    )
    rule_strings = {
        'Rule1': "(X['A']>0)&(X['C']>0)",
        'Rule2': "(X['B']>0)&(X['D']>0)",
        'Rule3': "(X['D']>0)",
        'Rule4': "(X['C']>0)"
    }
    rules = Rules(rule_strings=rule_strings)
    rule_lambdas = rules.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    ro = BayesianOptimiser(
        rule_lambdas=rule_lambdas,
        lambda_kwargs=rules.lambda_kwargs,
        metric=f1.fit,
        n_iter=5
    )
    sf = SimpleFilter(
        threshold=0.05,
        operator='>=',
        metric=f1.fit
    )
    cf = CorrelatedFilter(
        correlation_reduction_class=AgglomerativeClusteringReducer(
            threshold=0.9,
            strategy='bottom_up',
            similarity_function=js.fit
        )
    )
    gf = GreedyFilter(
        metric=f1.fit,
        sorting_metric=p.fit
    )
    rbs = RBSOptimiser(
        RBSPipeline(
            config=[],
            final_decision=0,
        ),
        metric=f1.fit,
        pos_pred_rules=ClassAccessor('gf', 'rules_to_keep'),
        neg_pred_rules=[],
        n_iter=10
    )
    return rg_dt, rg_opt, ro, sf, cf, gf, rbs


def test_fit_predict_rule_gen_dt(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    rg_dt, _, _, sf, cf, gf, rbs = _instantiate_classes
    search_spaces = {
        'rg_dt': {
            'n_total_conditions': UniformInteger(2, 5),
            'target_feat_corr_types': Choice([None])
        },
        'sf': {
            'threshold': UniformFloat(0, 1),
        },
        'cf': {
            'threshold': UniformFloat(0, 1)
        },
        'gf': {
            'metric': Choice([p.fit, f1.fit])
        },
        'rbs': {
            'n_iter': UniformInteger(10, 15)
        }
    }
    steps = [
        ('rg_dt', rg_dt),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    bs = BayesSearchCV(
        pipeline=lp,
        search_spaces=search_spaces,
        metric=f1.fit,
        cv=3,
        n_iter=5,
        error_score=0,
        verbose=1
    )
    with pytest.warns(UserWarning):
        # Test fit/predict/fit_predict, no sample_weight
        bs.fit(X, y)
        assert bs.best_score == 0.22857142857142856
        assert bs.best_index == 2
        assert bs.best_params['cf']['threshold'] == 0.08174388306233471
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method Precision.fit of Precision>'
        assert bs.best_params['rbs']['n_iter'] == 15.0
        assert bs.best_params['rg_dt']['n_total_conditions'] == 5.0
        assert bs.best_params['rg_dt']['target_feat_corr_types'] is None
        assert bs.best_params['sf']['threshold'] == 0.2583716619727481
        assert bs.cv_results.shape == (5, 11)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.02
        assert f1.fit(y_pred, y) == 0.33333333333333337
        y_pred = bs.fit_predict(X, y)
        assert y_pred.mean() == 0.02
        assert f1.fit(y_pred, y) == 0.33333333333333337
        # Test fit/predict/fit_predict, sample_weight given
        bs.fit(X, y, sample_weight)
        assert bs.best_score == 0.3015873015873016
        assert bs.best_index == 4
        assert bs.best_params['cf']['threshold'] == 0.1342082233571794
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 12.0
        assert bs.best_params['rg_dt']['n_total_conditions'] == 4.0
        assert bs.best_params['rg_dt']['target_feat_corr_types'] is None
        assert bs.best_params['sf']['threshold'] == 0.1474198119511717
        assert bs.cv_results.shape == (5, 11)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.1
        assert f1.fit(y_pred, y, sample_weight) == 0.8421052631578948
        y_pred = bs.fit_predict(X, y, sample_weight)
        assert y_pred.mean() == 0.1
        assert f1.fit(y_pred, y, sample_weight) == 0.8421052631578948


def test_fit_predict_rule_gen_opt(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    _, rg_opt, _, sf, cf, gf, rbs = _instantiate_classes
    search_spaces = {
        'rg_opt': {
            'n_total_conditions': UniformInteger(2, 5),
        },
        'sf': {
            'threshold': UniformFloat(0, 1),
        },
        'cf': {
            'threshold': UniformFloat(0, 1)
        },
        'gf': {
            'metric': Choice([p.fit, f1.fit])
        },
        'rbs': {
            'n_iter': UniformInteger(10, 15)
        }
    }
    steps = [
        ('rg_opt', rg_opt),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    bs = BayesSearchCV(
        pipeline=lp,
        search_spaces=search_spaces,
        metric=f1.fit,
        cv=3,
        n_iter=5,
        error_score=0,
        verbose=1
    )
    with pytest.warns(UserWarning):
        # Test fit/predict/fit_predict, no sample_weight
        bs.fit(X, y)
        assert bs.best_score == 0.08333333333333333
        assert bs.best_index == 4
        assert bs.best_params['cf']['threshold'] == 0.1342082233571794
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 12.0
        assert bs.best_params['rg_opt']['n_total_conditions'] == 4.0
        assert bs.best_params['sf']['threshold'] == 0.1474198119511717
        assert bs.cv_results.shape == (5, 10)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.11
        assert f1.fit(y_pred, y) == 0.5714285714285713
        y_pred = bs.fit_predict(X, y)
        assert y_pred.mean() == 0.11
        assert f1.fit(y_pred, y) == 0.5714285714285713
        # Test fit/predict/fit_predict, sample_weight given
        bs.fit(X, y, sample_weight)
        assert bs.best_score == 0.3619909502262444
        assert bs.best_index == 4
        assert bs.best_params['cf']['threshold'] == 0.1342082233571794
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 12.0
        assert bs.best_params['rg_opt']['n_total_conditions'] == 4.0
        assert bs.best_params['sf']['threshold'] == 0.1474198119511717
        assert bs.cv_results.shape == (5, 10)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.13
        assert f1.fit(y_pred, y, sample_weight) == 0.6153846153846154
        y_pred = bs.fit_predict(X, y, sample_weight)
        assert y_pred.mean() == 0.13
        assert f1.fit(y_pred, y, sample_weight) == 0.6153846153846154


def test_fit_predict_rule_opt(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    _, _, ro, sf, cf, gf, rbs = _instantiate_classes
    search_spaces = {
        'ro': {
            'metric': Choice([p.fit, f1.fit])
        },
        'sf': {
            'threshold': UniformFloat(0, 1),
        },
        'cf': {
            'threshold': UniformFloat(0, 1)
        },
        'gf': {
            'metric': Choice([p.fit, f1.fit])
        },
        'rbs': {
            'n_iter': UniformInteger(10, 15)
        }
    }
    steps = [
        ('ro', ro),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    bs = BayesSearchCV(
        pipeline=lp,
        search_spaces=search_spaces,
        metric=f1.fit,
        cv=3,
        n_iter=10,
        error_score=0,
        verbose=0
    )
    with pytest.warns(UserWarning):
        # Test fit/predict/fit_predict, no sample_weight
        bs.fit(X, y)
        assert bs.best_score == 0.09042145593869733
        assert bs.best_index == 9
        assert bs.best_params['cf']['threshold'] == 0.933434993006217
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 13.0
        assert str(bs.best_params['ro']['metric']
                   ) == '<bound method Precision.fit of Precision>'
        assert bs.best_params['sf']['threshold'] == 0.09205449690521583
        assert bs.cv_results.shape == (10, 10)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.82
        assert f1.fit(y_pred, y) == 0.1956521739130435
        y_pred = bs.fit_predict(X, y)
        assert y_pred.mean() == 0.82
        assert f1.fit(y_pred, y) == 0.1956521739130435
        # Test fit/predict/fit_predict, sample_weight given
        bs.fit(X, y, sample_weight)
        assert bs.best_score == 0.2756485534322622
        assert bs.best_index == 9
        assert bs.best_params['cf']['threshold'] == 0.933434993006217
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 13.0
        assert str(bs.best_params['ro']['metric']
                   ) == '<bound method Precision.fit of Precision>'
        assert bs.best_params['sf']['threshold'] == 0.09205449690521583
        assert bs.cv_results.shape == (10, 10)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.82
        assert f1.fit(y_pred, y, sample_weight) == 0.3243243243243243
        y_pred = bs.fit_predict(X, y, sample_weight)
        assert y_pred.mean() == 0.82
        assert f1.fit(y_pred, y, sample_weight) == 0.3243243243243243


def test_error(_create_data, _instantiate_classes):
    X, y, _ = _create_data
    _, _, ro, sf, cf, gf, rbs = _instantiate_classes
    search_spaces = {
        'ro': {
            'metric': Choice([p.fit, f1.fit])
        },
        'sf': {
            'threshold': UniformFloat(0, 1),
        },
        'cf': {
            'threshold': UniformFloat(0, 1)
        },
        'gf': {
            'metric': Choice([p.fit, f1.fit])
        },
        'rbs': {
            'n_iter': UniformInteger(10, 15)
        }
    }
    steps = [
        ('ro', ro),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    bs = BayesSearchCV(
        pipeline=lp,
        search_spaces=search_spaces,
        metric=f1.fit,
        cv=3,
        n_iter=10,
        error_score='raise',
        verbose=0
    )
    with pytest.raises(Exception, match="No rules remaining for: Pipeline parameter set = {'cf': {'threshold': 0.1955964101622225}, 'gf': {'metric': <bound method Precision.fit of Precision>}, 'rbs': {'n_iter': 12.0}, 'ro': {'metric': <bound method Precision.fit of Precision>}, 'sf': {'threshold': 0.4860473230215504}}; Fold index = 0."):
        bs.fit(X, y)

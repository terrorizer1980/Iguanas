"""
Generates scores for each rule in a set. Scaling functions can also be applied
to the scores.
"""
from iguanas.rule_scoring.rule_scoring_methods import PerformanceScorer, LogRegScorer,\
    RandomForestScorer
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
from iguanas.rule_scoring.rule_score_scalers import MinMaxScaler, ConstantScaler
from typing import Union


class RuleScorer:
    """
    Generates rule scores using the rule binary columns and the target column.

    Parameters
    ----------
    scoring_class : Union[PerformanceScorer, LogRegScorer, RandomForestScorer]
        The instantiated scoring class - this defines the method for
        generating the scores. Scoring classes are available in the 
        `rule_scoring_methods` module.
    scaling_class : Union[MinMaxScaler, ConstantScaler], optional
        The instantiated scaling class - this defines the method for 
        scaling the raw scores from the scoring class. Scaling classes are
        available in the `rule_score_scalers` module. Defaults to None.

    Attributes
    ----------
    rule_scores : Dict[str, int]
        Contains the generated score (values) for each rule (keys).
    """

    def __init__(self, scoring_class: Union[PerformanceScorer, LogRegScorer,
                                            RandomForestScorer],
                 scaling_class=None):

        self.scoring_class = scoring_class
        self.scaling_class = scaling_class

    def fit(self, X_rules: PandasDataFrameType, y: PandasSeriesType,
            sample_weight=None) -> None:
        """
        Generates rule scores using the rule binary columns and the binary 
        target column.

        Parameters
        ----------
        X_rules : PandasDataFrameType
            The rule binary columns.
        y : PandasSeriesType
            The binary target column.
        sample_weight : PandasSeriesType, optional
            Row-wise weights to apply in the `scoring_class`. Defaults to None.
        """

        self.rule_scores = self.scoring_class.fit(
            X_rules=X_rules, y=y, sample_weight=sample_weight)
        if self.scaling_class is not None:
            self.rule_scores = self.scaling_class.fit(
                rule_scores=self.rule_scores)

    def transform(self, X_rules: PandasDataFrameType) -> PandasDataFrameType:
        """
        Transforms the rule binary columns to show the generated scores applied
        to the dataset (i.e. replaces the 1 in `X_rules` with the generated 
        score).

        Parameters
        ----------
        X_rules : PandasDataFrameType
            The rule binary columns.

        Returns
        -------
        PandasDataFrameType
            The generated scores applied to the dataset.
        """

        X_scores = self.rule_scores * X_rules
        return X_scores

    def fit_transform(self, X_rules: PandasDataFrameType, y: PandasSeriesType,
                      sample_weight=None) -> PandasDataFrameType:
        """
        Generates rule scores using the rule binary columns and the binary 
        target column, then transforms the rule binary columns to show the 
        generated scores applied to the dataset (i.e. replaces the 1 in 
        `X_rules` with the generated score).

        Parameters
        ----------
        X_rules : PandasDataFrameType
            The rule binary columns.
        y : PandasSeriesType
            The binary target column.
        sample_weight : PandasSeriesType, optional
            Row-wise weights to apply in  the `scoring_class`. Defaults to 
            None.

        Returns
        -------
        PandasDataFrameType
            The generated scores applied to the dataset.
        """

        self.fit(X_rules=X_rules, y=y, sample_weight=sample_weight)
        X_scores = self.transform(X_rules=X_rules)
        return X_scores

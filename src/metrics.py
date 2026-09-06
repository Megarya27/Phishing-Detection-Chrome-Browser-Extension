#This program calculates evaluation metrics for model performance, including precision, recall, F1 score, and ROC-AUC. 
# It also provides functions for bootstrapping to estimate confidence intervals and performing statistical significance tests between models. 
# also includes paired significance tests to determine if one model statistically outperforms another.

#imports
from dataclasses import dataclass
import numpy as np
from scipy import stats
from sklearn.metrics import ( #performance metric calculators from scikit-learn
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score)

from global_config import N_BOOTSTRAP, RANDOM_SEED

#evaluation metrics container
@dataclass
class PointMetrics:
    precision: float
    recall: float
    f1: float
    auc_roc: float
    def as_dict(self): #convert metrics to standard Python dictionary
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc }

#computes precision, recall, F1, and AUC-ROC on given predictions
#y_true: actual labels (0 or 1)
#y_pred: models prediction(0 or 1)
#y_prob: model raw probability scores for class 1 (0.0 to 1.0)
def compute_point_metrics(y_true, y_pred, y_prob) -> PointMetrics:
    precision = precision_score(y_true, y_pred, zero_division=0) #precision: how many were correct?, out of all predicted positives
    recall = recall_score(y_true, y_pred, zero_division=0) #recall: how many were caught?, out of all actual positives
    f1 = f1_score(y_true, y_pred, zero_division=0) #f1: balances precision and recall, useful for imbalanced datasets

    # ROC-AUC: measures how well probabilities rank positive samples above negative samples.
    unique_classes = set(y_true)
    if len(unique_classes) > 1:
        auc_roc = roc_auc_score(y_true, y_prob) # requires at least two distinct classes present in y_true
    else:
        auc_roc = float("nan") #if test split contains only 0s or only 1s then AUC is undefined

    return PointMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
    )


#generates an empirical score distribution using bootstrapping.
#randomly draws N samples with replacement from test set, recalculates chosen metric on that resampled slice
#and repeats this process `n_bootstrap` times.
def bootstrap_metric_distribution(y_true, y_pred, y_prob, metric="f1", n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    #initialise reproducible random number generator
    rng = np.random.default_rng(seed)

    #ensure all inputs are NumPy arrays so it can be indexed with arrays of integers
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)
    total_samples = len(y_true_arr)
    scores = np.empty(n_bootstrap)#new empty array to store score from each repetition

    for i in range(n_bootstrap):
        #pick random row indices from 0 (including) to total_samples with replacement
        sample_indices = rng.integers(0, total_samples, size=total_samples)

        # slice true labels, predictions and probabilities using those indices
        resampled_true = y_true_arr[sample_indices]
        resampled_pred = y_pred_arr[sample_indices]
        resampled_prob = y_prob_arr[sample_indices]

        #calculate requested metric for this random slice
        if metric == "f1": score = f1_score(resampled_true, resampled_pred, zero_division=0)

        elif metric == "precision":
            score = precision_score(resampled_true, resampled_pred, zero_division=0)

        elif metric == "recall":
            score = recall_score(resampled_true, resampled_pred, zero_division=0)

        elif metric == "auc_roc":
            # ROC-AUC cant be calculated if resample drew only one class
            if len(set(resampled_true)) > 1:
                score = roc_auc_score(resampled_true, resampled_prob)
            else:
                score = np.nan
        else:
            raise ValueError(f"Unknown metric requested: '{metric}'")
        scores[i] = score
    return scores


#computes a percentile based confidence interval (default: 95% CI so alpha=0.05) for given array of scores.
def confidence_interval(scores, alpha=0.05):
    lower_percentile = 100 * (alpha / 2.0)          # 2.5%
    upper_percentile = 100 * (1.0 - (alpha / 2.0))  # 97.5%

    # np.nanpercentile ignores failed NaN runs if  occurred
    lower_bound = np.nanpercentile(scores, lower_percentile)
    upper_bound = np.nanpercentile(scores, upper_percentile)
    return float(lower_bound), float(upper_bound)


#runs a paired bootstrap t-test between two matched bootstrap score distributions.
#checks if Model A's improvement over Model B is statistically significant or random noise.
def paired_bootstrap_ttest(scores_a, scores_b):
    arr_a = np.asarray(scores_a)
    arr_b = np.asarray(scores_b)
    differences = arr_a - arr_b
    t_stat, p_value = stats.ttest_1samp(differences, popmean=0.0) #null hypothesis: true average difference is 0.0
    return float(t_stat), float(p_value)

# adjusts the significance threshold (alpha) when making multiple comparisons
# to avoid false discoveries (Family-Wise Error Rate control).
# Formula: corrected_alpha = alpha / number_of_tests
def bonferroni_correct(p_values, alpha=0.05):
    number_of_tests = len(p_values)
    corrected_alpha = alpha / number_of_tests
    # Check if each test is significant under the stricter threshold
    significance_flags = []
    for p in p_values:
        is_significant = p < corrected_alpha
        significance_flags.append(is_significant)
    return corrected_alpha, significance_flags


# generates a full summary containing:
#direct point metrics.
#95% bootstrap confidence interval bounds for F1.
#the full bootstrap distribution (for downstream statistical testing).
def full_report(name, y_true, y_pred, y_prob):
    point_metrics = compute_point_metrics(y_true, y_pred, y_prob)
    f1_distribution = bootstrap_metric_distribution(
        y_true,
        y_pred,
        y_prob,
        metric="f1",
    )
    ci_low, ci_high = confidence_interval(f1_distribution)
    report = point_metrics.as_dict()
    report["model"] = name
    report["f1_ci_low"] = ci_low
    report["f1_ci_high"] = ci_high

    return report, f1_distribution
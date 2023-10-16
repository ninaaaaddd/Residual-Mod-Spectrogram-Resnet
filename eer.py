import numpy as np

def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. 
    
    :param target_scores: The values for the target class. For binary, outputs for any one of the output neurons
    :type target_scores: list or array
    :param nontarget_scores: The values for the non target class. For binary, outputs for the other class same neuron as targetscore
    :type nontarget_scores: list or array
    :return: (EER, threshold)
    :rtype: (float, float)
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def getEER(score_path):
    """ Returns equal error rate (EER) and the corresponding threshold.
    :param score_path: path to score file
    :type score_path: str
    :return: EER
    :rtype: float
    """
    cm_data = np.genfromtxt(score_path, dtype=str)
    cm_keys = cm_data[:, 1]
    cm_scores = cm_data[:, 2].astype(float)
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    eer_cm, _ = compute_eer(bona_cm, spoof_cm)[0]
    return eer_cm * 100

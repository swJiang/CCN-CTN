import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve


def compute_valid_roc(labels, distance):
    '''
    Note:this a metric learning so match pairs have more little distance
         non-match pairs have more bigger distance,but standard roc compute
         assume better match have better scores
         so we need to modify our socre using maximum of score subtract all
         score
    '''
    labels = np.uint8(labels)
    pos_index = np.where(labels == 1)
    pos_distance = np.array(distance)[pos_index[0]]
    neg_index = np.where(labels == 0)
    neg_distance = np.array(distance)[neg_index[0]]
    plt.figure()
    plt.plot(list(range(50000)), pos_distance, 'r*', label='match-distance')
    plt.plot(list(range(50000)), neg_distance, 'gp', label='nonmatch-distance')
    plt.legend()
    plot_npy = np.concatenate((pos_distance, neg_distance),axis=0)
    np.save("./ss.npy",plot_npy)
    # reverse distance to reassure match pair have bigger score
    # non-macth pair have little score
    # reverse_dist = np.max(distance) - distance
    auc_ = roc_auc_score(labels, distance)
    fpr, tpr, thresholds = roc_curve(labels, distance)
    index = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[index]
    print("\n")
    print('>>>@fpr95:%.5f' % fpr95, 'auc:%.3f' % auc_)
    plt.figure()
    plt.plot(fpr, tpr, 'r-')
    plt.title('ROC-curve')
    plt.show()
    plt.savefig("./roc.jpg")
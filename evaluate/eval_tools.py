from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.integrate import simps

def dist(gtLandmark, dist_type='centers', left_pt=44, right_pt=52, num_eye_pts=8):
    if dist_type=='centers':
        normDist = np.linalg.norm(np.mean(gtLandmark[left_pt:left_pt+num_eye_pts], axis=0) -
                                  np.mean(gtLandmark[right_pt:right_pt+num_eye_pts], axis=0))
    elif dist_type=='corners':
        normDist = np.linalg.norm(gtLandmark[left_pt] - gtLandmark[right_pt+num_eye_pts/2])
    elif dist_type=='diagonal':
        height, width = np.max(gtLandmark, axis=0) - np.min(gtLandmark, axis=0)
        normDist = np.sqrt(width**2 + height**2)
    return normDist

def landmark_error(gtLandmarks, predict_Landmarks, dist_type='centers', show_results=False, verbose=False):
    norm_errors = []
    errors = []
    for i in range(len(gtLandmarks)):
        norm_dist = dist(gtLandmarks[i], dist_type=dist_type)
        error = np.mean(np.sqrt(np.sum((gtLandmarks[i] - predict_Landmarks[i])**2, axis=1)))
        norm_error = error/norm_dist
        errors.append(error)
        norm_errors.append(norm_error)
        if verbose:
            print('{0}: {1}'.format(i, error))

    if verbose:
        print("Image idxs sorted by error")
        print(np.argsort(errors))
    avg_error = np.mean(errors)
    avg_norm_error = np.mean(norm_errors)
    print("Average error: {0}".format(avg_error))
    print("Average norm error: {0}".format(avg_norm_error))
    return norm_errors, errors

def auc_error(errors, failure_threshold, step=0.0001, save_path='', showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failure_threshold+step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]
    auc = simps(ced, x=xAxis) / failure_threshold
    failure_rate = 1. - ced[-1]
    print("AUC @ {0}: {1}".format(failure_threshold, auc))
    print("Failure rate: {0}".format(failure_rate))

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()
        plt.savefig(save_path)

    return auc, failure_rate

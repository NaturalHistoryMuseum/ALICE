import itertools

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import match_descriptors, ORB


def pairs(iterable, cycle=False):
    first = iter(iterable)
    second = iter(iterable)
    if cycle:
        second = itertools.cycle(second)
    next(second)

    return zip(first, second)


def visualise(images, feature_detectors, matches):
    all_images = np.concatenate(images, axis=1)
    plt.imshow(all_images)
    for m in global_matches:
        points = zip(*[detector.keypoints[k, ::-1] + np.array([i * images[0].shape[1], 0])
                       for i, k, detector in zip(range(4), m, feature_detectors)])
        plt.plot(*points)
    plt.show()


def global_matches(images):
    feature_detectors = [ORB(n_keypoints=3000) for i in range(4)]
    for detector, image in zip(feature_detectors, images):
        detector.detect_and_extract(image[..., 1])

    matches = [{first: second for first, second
                in match_descriptors(detector1.descriptors, detector2.descriptors, cross_check=True)}
               for detector1, detector2 in pairs(feature_detectors)]

    global_matches = []
    for first in matches[0]:
        point = [first]
        try:
            for pair in matches:
                point.append(pair[point[-1]])
            global_matches.append(point)
        except KeyError:
            pass

    points = np.zeros((len(global_matches), 4, 2))
    for i in range(len(global_matches)):
        for j in range(4):
            points[i, j] = feature_detectors[j].keypoints[global_matches[i][j], ::-1]

    return points

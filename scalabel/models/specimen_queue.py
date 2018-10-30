import itertools

import imghdr
import json
import os
import re

from scalabel.models import Specimen
from scalabel.models.logger import logger


class SpecimenQueue(object):
    """
    A list of specimens and their associated views.
    :param specimens: a list of Specimen objects

    """

    def __init__(self, specimens):
        self.specimens = specimens

    def __getitem__(self, item):
        return self.specimens[item]

    def __setitem__(self, key, value):
        self.specimens[key] = value

    def __delitem__(self, key):
        del self.specimens[key]

    @classmethod
    def load(cls, calibrator, root, specimen_sep='_', lookup_file='.lookup'):
        """
        Load specimens from a folder using the given calibration.
        :param calibrator: the calibrator instance for this batch
        :param root: the folder where the images are stored
        :param specimen_sep: the character separating the specimen ID from the camera
                             ID in the filename, e.g. in SPECIMEN_ALICE2.jpg the sep
                             is _ (Default value = '_')
        :param lookup_file: a json config file linking filenames with camera IDs using
                            regex (Default value = '.lookup')
        :returns: SpecimenQueue

        """
        img_files = [f for f in os.listdir(root) if
                     os.path.isfile(os.path.join(root, f)) and imghdr.what(
                         os.path.join(root, f)) is not None]
        specimen_files = {k: list(v) for k, v in itertools.groupby(sorted(img_files),
                                                                   lambda x:
                                                                   x.split(specimen_sep)[
                                                                       0])}
        specimens = []

        with open(lookup_file, 'r') as f:
            camera_lookup = json.load(f)

        for ix, (specimen, filenames) in enumerate(specimen_files.items()):
            images = []
            for rgx, camera_id in camera_lookup.items():
                try:
                    f = next(i for i in filenames if
                             re.search(rgx, i))
                    try:
                        vp = calibrator[camera_id].position
                        images.append((vp, os.path.join(root, f)))
                    except KeyError:
                        continue
                except StopIteration:
                    continue
            s = Specimen.from_images(specimen, images)
            specimens.append(s)
            logger.debug(f'loaded {s.id} ({ix + 1}/{len(specimen_files)})')
        return cls(specimens)

    def limit(self, n):
        """
        Return a queue with only the top n specimens.
        :param n: the number of specimens in the new queue

        """
        return SpecimenQueue(self.specimens[:min(n, len(self.specimens))])

    def try_process(self, specimen_transform, log_stage=None):
        """
        Try to apply the specified function/transform to each specimen in the queue.
        :param specimen_transform: a function taking a specimen object as the argument
                                   and outputting a specimen object
        :param log_stage: a description of the stage, e.g. 'find features' (optional)
        :returns: a new specimen queue

        """
        specimens = []
        q = self.specimens.copy()
        attempts = {s.id: 0 for s in self.specimens}
        log_stage = '' if log_stage is None else f' at {log_stage} stage'
        while len(q) > 0 and all([v < 3 for v in attempts.values()]):
            s = q.pop(0)
            try:
                specimens.append(specimen_transform(s))
                logger.debug(f'processed {s.id}{log_stage}')
                del attempts[s.id]
            except Exception as e:
                logger.debug(f'{s.id} failed{log_stage}: {e}')
                attempts[s.id] += 1
                q.append(s)
                continue
        return SpecimenQueue(specimens)

from scalabel.models.viewsets import Specimen
from ._models import PostLabel


def specimen_contrast_labels(specimen):
    """
    Increase the contrast of the labels in the specimen.
    :param specimen: the specimen with labels
    :return: a specimen with PostLabel objects

    """
    new_specimen = Specimen.from_specimen(specimen)
    new_specimen.labels = [PostLabel(specimen.id, l.views, l.image) for l in
                           specimen.labels]
    for l in new_specimen.labels:
        l.contrast()
    return new_specimen


def queue_contrast_labels(queue):
    """
    Run the specimen_contrast_labels method for each specimen in the queue.
    :param queue: a queue of specimens
    :return: a queue of transformed specimens

    """
    return queue.try_process(specimen_contrast_labels, 'label contrast')

from ._features import FeaturesSpecimen
from ._labels import LabelSpecimen


def specimen_find_features(specimen):
    """
    Transforms the given specimen into a FeaturesSpecimen, which locates common
    features in the views.
    :param specimen: the specimen to detect features in
    :return: a FeaturesSpecimen

    """
    return FeaturesSpecimen.from_specimen(specimen)


def queue_find_features(queue):
    """
    Run the specimen_find_features method for each specimen in the queue.
    :param queue: a queue of specimens
    :return: a queue of transformed specimens (FeaturesSpecimen objects)

    """
    return queue.try_process(specimen_find_features, 'find features')


def specimen_find_labels(specimen):
    """
    Transforms a FeaturesSpecimen into a LabelsSpecimen, which uses the detected
    features to find labels in the views.
    :param specimen: a FeaturesSpecimen
    :return: a LabelsSpecimen

    """
    return LabelSpecimen.from_specimen(specimen)


def queue_find_labels(queue):
    """
    Run the specimen_find_labels method for each specimen in the queue.
    :param queue: a queue of FeaturesSpecimen objects
    :return: a queue of transformed specimens (LabelsSpecimen objects)

    """
    return queue.try_process(specimen_find_labels, 'find labels')

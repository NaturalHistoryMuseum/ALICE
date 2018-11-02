from ALICE.models import Specimen, SpecimenQueue
from ._models import AlignedLabel


def specimen_align_labels(specimen):
    """
    Align the parts of each label in the given specimen.
    :param specimen: a specimen with Label objects
    :return: a specimen with AlignedLabel objects

    """
    specimen_with_labels = Specimen.from_specimen(specimen)
    specimen_with_labels.labels = [AlignedLabel.from_label(l) for l in specimen.labels]
    return specimen_with_labels


def queue_align_labels(queue: SpecimenQueue):
    """
    Run the specimen_align_labels method for each specimen in the queue.
    :param queue: a queue of specimens
    :return: a queue of transformed specimens

    """
    return queue.try_process(specimen_align_labels, 'label alignment')

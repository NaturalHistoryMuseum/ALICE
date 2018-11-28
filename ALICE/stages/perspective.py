from ALICE.models.specimen_queue import SpecimenQueue
from ALICE.models.views import WarpedView, RecognitionView
from ALICE.models.viewsets import Specimen


def specimen_warp(specimen):
    """
    Warps the image for each view in the given specimen, using the view position
    transform in order to standardise the perspectives.
    :param specimen: the specimen to warp
    :return: a specimen with WarpedView objects

    """
    return Specimen(specimen.id, [WarpedView.from_view(v) for v in specimen.views])


def queue_warp(queue: SpecimenQueue):
    """
    Run the specimen_warp method for each specimen in the queue.
    :param queue: a queue of specimens
    :return: a queue of transformed specimens

    """
    return queue.try_process(specimen_warp, 'perspective correction')


def specimen_crop(specimen):
    """
    Tries to find the labels in each view of the given specimen, and crops to that area.
    :param specimen: the specimen
    :return: a specimen with cropped RecognitionView objects

    """
    return Specimen(specimen.id, [RecognitionView.from_view(v) for v in specimen.views])


def queue_crop(queue: SpecimenQueue):
    """
    Run the specimen_crop method for each specimen in the queue.
    :param queue: a queue of specimens
    :return: a queue of specimens with views cropped to the labels

    """
    return queue.try_process(specimen_crop, 'object recognition')

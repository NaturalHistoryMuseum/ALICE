from ALICE.models import SpecimenQueue
from ALICE.models.views import WarpedView
from ALICE.models.viewsets import Specimen


def specimen_warp(specimen):
    """
    Warps the image for each view in the given specimen, using the view position
    transform in order to standardise the perspectives.
    :param specimen: the specimen to warp

    """
    return Specimen(specimen.id, [WarpedView.from_view(v) for v in specimen.views])


def queue_warp(queue: SpecimenQueue):
    """
    Run the specimen_warp method for each specimen in the queue.
    :param queue: a queue of specimens
    :return: a queue of transformed specimens

    """
    return queue.try_process(specimen_warp, 'perspective correction')

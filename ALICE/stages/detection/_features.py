from ALICE.models.viewsets import FeatureComparer, Specimen


class FeaturesSpecimen(Specimen):
    """
    Instantiates a FeatureComparer, which tries to find common features in the images
    of the specimen, and stores it in the 'comparer' attribute.
    :param specimen_id: the ID of the associated specimen
    :param views: a list of FeaturesView objects (views of the specimen)

    """

    def __init__(self, specimen_id, views):
        super(FeaturesSpecimen, self).__init__(specimen_id, views)
        self.comparer = FeatureComparer.ensure_minimum(specimen_id, views)

    @property
    def display(self):
        return self.comparer.display

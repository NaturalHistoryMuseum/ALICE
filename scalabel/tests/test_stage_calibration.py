import numpy as np

from scalabel.calibrate import Calibrator

calibration_csv = 'data/square.csv'


class TestStageCalibration(object):
    """
    Testing the calibration stage of the pipeline. Needs to numerically define the
    distortion for each camera's view relative to the others.
    """

    def test_load_serialised_calibration(self):
        calibration = Calibrator.from_csv(calibration_csv)
        assert isinstance(calibration.coordinates, np.ndarray)
        assert len(calibration.coordinates.shape) == 3
        assert calibration.coordinates.shape[-1] == 2

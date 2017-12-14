import csv
import numpy as np
from pathlib import Path
import skimage.draw
import skimage.io
from skimage.transform import AffineTransform, estimate_transform, rescale, warp


def load_coordinates(path):
    points = np.zeros((4, 4, 2), dtype=np.float)

    with open(path, 'r+') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
        header = next(reader)
        for camera, index, *coordinate in reader:
            points[int(camera), int(index)] = coordinate
        return points


def register_images(images, scale=0.25, coordinates='square.csv'):
    images = [rescale(image, scale) for image in images]
    # points representing squares on the calibration grid
    point_list = load_coordinates(coordinates) * scale

    box_normal = np.array([[0, 100], [0, 0], [100, 0], [100, 100]])
    trans = [estimate_transform('projective', box_normal, points) for points in point_list]
    height, width = images[0].shape[:2]
    box_image = np.array([[0, height], [0, 0], [width, 0], [width, height]])
    bounds = [transformation.inverse(box_image) for transformation in trans]
    offset = np.stack(bounds).min(axis=0).min(axis=0)
    shape = np.stack(bounds).max(axis=0).max(axis=0) - offset
    scale = shape / np.array([4000, 4000])

    warped_images = []
    display_images = []
    for i, (transformation, image) in enumerate(zip(trans, images)):
        homography = AffineTransform(scale=scale) + AffineTransform(translation=offset) + transformation
        warped_images.append(warp(image, homography, output_shape=(4000, 4000))[1000:-1000, 1000:-1000])

        display_images.append(image.copy())
        rr, cc = skimage.draw.polygon_perimeter(point_list[i, :, 1], point_list[i, :, 0])
        display_images[-1][rr, cc] = 1

    return warped_images, display_images


if __name__ == '__main__':
    data_dir = Path('data', 'Test 3_Flies_GreyFoam')
    specimen_prefix = "0003"

    filenames = sorted(filename for filename in data_dir.iterdir() if specimen_prefix in filename.stem)[2:]
    images = [skimage.io.imread(name) for name in filenames]

    warped, display = register_images(images)

# scalabel
### Software for ALICE

![](https://github.com/NaturalHistoryMuseum/scalabel/blob/master/example.png)

## Pipeline:
1) **Calibration:** for each of the four camera angles, image coordinates for a flat square in world space must be given in `square.csv`. This can be done in two ways:
     * *Manually:* by determining the `(x,y)` pixel coordinates for each *corresponding* corner and creating a `square.csv` file by hand. This must fit the format in the example [`square.csv`](https://github.com/NaturalHistoryMuseum/scalabel/blob/d50877eb1c85eae676a12ae59e43105436188dfc/square.csv)
     * *Automatically:* by using four views of the calibration pattern and the `calibration.py` script
1) **Alignment/Perspective correction:** using the corresponding points from the calibration step, find the homography (perspective transformation) that aligns all four images. This can be fit exactly and will result in everything in the *same flat plane* as the calibration square being in alignment. For anything above or below this will only *approximately* align them.
1) **Feature matching:** features can be detected in *each* image and matched across *all* corrected images, such that we know for each detected point what its coordinates will be in another image. 
1) **Cluster feature points:** given the correspondences between feature points, it is possible to cluster the points such that each cluster represents points on the same label in the image. This enables cropping segments from each image to give separate images that each contain the same label and nothing else.
1) **Label alignment:** since the perspective correction of the images is only approximate, the cropped label images need to be aligned more closely in order to overlay them correctly.
1) **Label reconstruction:** with labels aligned pixel-wise, a median filter is applied per-pixel so that anything only visible in a minority of views (such as a pin) will no longer be visible

## Note on licenses
Both pyflow (https://github.com/pathak22/pyflow) and gco_python (https://github.com/amueller/gco_python; http://vision.csd.uwo.ca/code/) are included but have their own licenses. If making the repository public this needs to be accounted for

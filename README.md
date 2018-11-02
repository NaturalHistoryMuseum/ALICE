# ALICE
## Angular Label Image Capture and Extraction

![](https://github.com/NaturalHistoryMuseum/scalabel/blob/master/example.png)

This project attempts to locate and extract images of labels attached to pinned insects. Given views of the specimen from multiple angles, it can isolate the labels.

## Pipeline:
1) **Calibration:** for each camera angle, image coordinates for a flat square in world space must be given in a CSV calibration file. This must fit the format in this [example CSV](data/cooper01/calibration/calibration.csv). The coordinates are translated to the centre of the image and scaled to fit.
2) **Perspective:** using the corresponding points from the calibration step, transform each view of the specimen so that the 'square' is actually flat and oriented the same in each image.
3) **Detection:**
  a) _Features:_ detect points of interest in each image and find the features that are common to all views.
  b) _Labels:_ cluster the feature points to find possible labels, and isolate these regions from the specimen images.
4) **Align:** since the perspective correction of the images is only approximate, the cropped label images need to be aligned more closely in order to overlay them correctly. With labels aligned pixel-wise, a median filter is then applied per-pixel so that anything only visible in a minority of views (such as a pin) will no longer be visible.
5) **Postprocess:** apply filters to improve contrast, etc.

## Note on licenses
Both pyflow (https://github.com/pathak22/pyflow) and gco_python (https://github.com/amueller/gco_python; http://vision.csd.uwo.ca/code/) are included but have their own licenses. If making the repository public this needs to be accounted for.

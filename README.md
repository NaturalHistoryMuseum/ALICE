# scalabel

## Current procedure:
1) Mark a square manually in each data/test_patterns calibration files. Hardcode the coordinatees in ```point_list``` in fourpoint.py
2) Run fourpoint.py to generate perspective corrected images
3) Run test_features.py to group label features and align labels

## Recommended Todo:
1) Label clustering. At the moment a label is selected manually, but matches are distinct enough to cluster labels according to match vectors.
2) Auto callibration system. Given calibration images of all cameras, generate loadable coordinates of well centred squares. This will help if camera configuration changes in the future.

## Note on licenses
Both pyflow (https://github.com/pathak22/pyflow) and gco_python (https://github.com/amueller/gco_python; http://vision.csd.uwo.ca/code/) are included but have their own licenses. If making the repository public this needs to be accounted for

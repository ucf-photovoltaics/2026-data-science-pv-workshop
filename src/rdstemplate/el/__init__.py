"""EL image-processing pipeline for the workshop case study.

Five stages turn a raw electroluminescence (EL) module image into per-module
defect features that merge back onto the tidy dataframe:

    01_input            raw EL PNGs (as captured)
    02_module_rectified rectify.py  -> straight-on module rectangle
    03_cells            cells.py    -> labeled individual cell crops
    04_cell_masks       segment.py  -> cells + semitransparent defect masks
    05_module_restitched stitch.py  -> masked cells reassembled into a module
    (features)          features.py -> per-class defect fractions -> metadata.csv

Each stage reads the previous stage's folder and writes the next, so students
can inspect the intermediate artifacts between steps.
"""

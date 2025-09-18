- Delete Rasterio (only used in imgTo3031, which is never used - yet)
- Make our own Preprocessing + CFAR following SNAP model (to avoid overly large files: the SNAP outputs are really large, some files may not ba processable) - https://github.com/senbox-org/s1tbx/blob/1daae60d572e3ad0ee98c0ce3538c61ad71d7fa1/s1tbx-op-feature-extraction/src/main/java/org/esa/s1tbx/fex/gpf/oceantools/AdaptiveThresholdingOp.java + https://github.com/senbox-org/s1tbx/blob/1daae60d572e3ad0ee98c0ce3538c61ad71d7fa1/s1tbx-op-feature-extraction/src/main/java/org/esa/s1tbx/fex/gpf/oceantools/ObjectDiscriminationOp.java
- Merge cnn_class.py and cnn_predictor.py 
- Add example notebook
- Make list of dependencies






- FILE STRUCTURE:
  
  ================================== PART OF REPO ==========================================
  - Generated shp files (output of main notebook)
    - 15D5
      - CNN
      - ResNet
      - CFAR
      - Ensemble
    - A890
      - CNN
      - ResNet
      - CFAR
      - Ensemble
  - Models
    - CNN.py (class module)
    - ResNet.py (class module)
    - CFAR.py
    - Ensemble.py
    - CNN.pt (need to initialize class first)
    - ResNet.pt (need to initialize class first)
  - Preprocessing methods
    - lee_filter.py
    - ...
  - Models Training (CFAR + Ensemble don't need training)
    - CNN (contains all necessary files for retraining the model)
      - files.extension
      - ...
    - ResNet
      - files.extension
      - ...
  - Utils
    - exportShpFromContours.py
    - getPoints.py
    - Maubat_Utils (adapted from https://github.com/benjaminfouquet/IcebergDetectionMaubat/tree/main/libs)
      - myGeoTools.py
  - README.md (includes doc for using SNAP + this repo)
  - TODO.md
  - main.ipynb (notebook is better for simplicity)
  ==========================================================================================

  ================================== NOT PART OF REPO ======================================
  (should be in gitignore)
  - Input images (ex: from polarview)
    - 1D5D.tif
    - A890.tif
  ==========================================================================================
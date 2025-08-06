import os
import sys

version = os.environ.get("SAM2_VERSION_TRACK", "official").lower()

if version in ["dam4sam", "dam4sam2"]:
    from .sam2_video_predictor_dam4sam import *

elif version in ["grounded", "grounded_sam", "groundedsam2"]:
    from .sam2_video_predictor_grounded import *

else:
    from .sam2_video_predictor_official import *

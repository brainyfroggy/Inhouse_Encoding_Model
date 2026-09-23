from __future__ import annotations

from pathlib import Path


HPG_PROJECT = Path("/blue/mzding/yujunchen/projects/TRIBEv2_emotion_modeling")
HPG_RESPONSES = Path("/blue/mzding/yujunchen/projects/TRIBEv2/outputs_ckvideo")
HPG_METADATA = Path(
    "/blue/mzding/yujunchen/data/original_ckvideo_data/CowenKeltnerEmotionalVideos.csv"
)

LOCAL_RESPONSES = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\TRIBEv2\ckvideos\outputs_ckvideo"
)
LOCAL_METADATA = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\original_ckvideo_data\CowenKeltnerEmotionalVideos.csv"
)

CLASS_NAMES = ["low", "mid", "high"]
RANDOM_SEED = 42

import os
from pathlib import Path

from paths import CE_CSL_ROOT, DATA_ROOT, MT5_ROOT, PROJECT_ROOT

mt5_path = str(MT5_ROOT)

project_root = str(PROJECT_ROOT)
ce_csl_root = str(CE_CSL_ROOT)
data_root = Path(DATA_ROOT)

# label paths
train_label_paths = {
                    "CSL_News": str(data_root / "CSL_News" / "CSL_News_Labels.json"),
                    "CSL_Daily": str(data_root / "CSL_Daily" / "labels.train"),
                    "CE_CSL": str(data_root / "CE_CSL" / "labels.train"),
                    "WLASL": str(data_root / "WLASL" / "labels-2000.train"),
                    "How2Sign": str(data_root / "How2Sign" / "labels.train"),
                    "OpenASL": str(data_root / "OpenASL" / "labels.train"),
                    }

dev_label_paths = {
                    "CSL_News": str(data_root / "CSL_News" / "CSL_News_Labels.json"),
                    "CSL_Daily": str(data_root / "CSL_Daily" / "labels.dev"),
                    "CE_CSL": str(data_root / "CE_CSL" / "labels.dev"),
                    "WLASL": str(data_root / "WLASL" / "labels-2000.dev"),
                    "How2Sign": "",
                    "OpenASL": str(data_root / "OpenASL" / "labels.dev"),
                    }

test_label_paths = {
                    "CSL_News": str(data_root / "CSL_News" / "CSL_News_Labels.json"),
                    "CSL_Daily": str(data_root / "CSL_Daily" / "labels.test"),
                    "CE_CSL": str(data_root / "CE_CSL" / "labels.test"),
                    "WLASL": str(data_root / "WLASL" / "labels-2000.test"),
                    "How2Sign": str(data_root / "How2Sign" / "labels.test"),
                    "OpenASL": str(data_root / "OpenASL" / "labels.test"),
}


# video paths
rgb_dirs = {
            "CSL_News": str(data_root / "CSL_News" / "rgb_format"),
            "CSL_Daily": str(data_root / "CSL_Daily" / "sentence-crop"),
            "CE_CSL": os.path.join(ce_csl_root, "video"),
            "WLASL": str(data_root / "WLASL" / "rgb_format"),
            "How2Sign": str(data_root / "How2Sign" / "rgb_format"),
            "OpenASL": str(data_root / "OpenASL" / "rgb_format"),
            }

# pose paths
pose_dirs = {
            "CSL_News": str(data_root / "CSL_News" / "pose_format"),
            "CSL_Daily": str(data_root / "CSL_Daily" / "pose_format"),
            "CE_CSL": os.path.join(ce_csl_root, "pose_format"),
            "WLASL": str(data_root / "WLASL" / "pose_format"),
            "How2Sign": str(data_root / "How2Sign" / "pose_format"),
            "OpenASL": str(data_root / "OpenASL" / "pose_format"),
}

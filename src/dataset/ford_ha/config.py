from pathlib import Path

# Base data directory
DATA_DIR = Path.home() / "Datasets" / "FordAV"

# Camera name mapping
CAM_MAP = {
    "FL": "FrontLeft",
    "FR": "FrontRight",
    "RL": "RearLeft",
    "RR": "RearRight",
    "SR": "SideRight",
    "SL": "SideLeft",
    # "C": "Center",  # Not synchronized with others
}

LOGS = {
    "train": {
        1: {
            "path": DATA_DIR / "2017-10-26/V2/Log1",
            "indices": list(range(4500, 8500)),
        },
        2: {
            "path": DATA_DIR / "2017-10-26/V2/Log2",
            "indices": list(range(3150))
            + list(range(6000, 9200))
            + list(range(11000, 15000)),
        },
        3: {
            "path": DATA_DIR / "2017-08-04/V2/Log3",
            "indices": list(range(1500)),
        },
        4: {
            "path": DATA_DIR / "2017-10-26/V2/Log4",
            "indices": list(range(7466)),
        },
        5: {
            "path": DATA_DIR / "2017-08-04/V2/Log5",
            "indices": list(range(3200))
            + list(range(5300, 9900))
            + list(range(10500, 11130)),
        },
        6: {
            "path": DATA_DIR / "2017-08-04/V2/Log6",
            "indices": list(range(1000, 3500))
            + list(range(4500, 5000))
            + list(range(7000, 7857)),
        },
    },
    "test": {
        1: {
            "path": DATA_DIR / "2017-08-04/V2/Log1",
            "indices": list(range(100, 200))
            + list(range(5000, 5500))
            + list(range(7000, 8500)),
        },
        2: {
            "path": DATA_DIR / "2017-08-04/V2/Log2",
            "indices": list(range(2500, 3000))
            + list(range(8500, 10500))
            + list(range(12500, 13727)),
        },
        3: {
            "path": DATA_DIR / "2017-08-04/V2/Log3",
            "indices": list(range(3500, 5000)),
        },
        4: {
            "path": DATA_DIR / "2017-08-04/V2/Log4",
            "indices": list(range(1500, 2500))
            + list(range(4000, 4500))
            + list(range(7000, 9011)),
        },
        5: {
            "path": DATA_DIR / "2017-10-26/V2/Log5",
            "indices": list(range(3500)),
        },
        6: {
            "path": DATA_DIR / "2017-10-26/V2/Log6",
            "indices": list(range(2000, 2500)) + list(range(3500, 4000)),
        },
    },
}

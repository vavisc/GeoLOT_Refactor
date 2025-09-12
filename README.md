## Project Setup

This project uses [`uv`](https://github.com/astral-sh/uv) to manage Python environments and dependencies. 
Clone the repository and install dependencies in a local virtual environment:

```bash
git clone https://github.com/vavisc/GeoLOT_Refactor.git
cd GeoLOT_Refactor
uv venv
uv sync
```

## Data

This project uses two well-known datasets: [FordAV Multiseasonal](https://avdata.ford.com/home/default.aspx) and [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php).
We provide download scripts for both (note: downloading can take quite some time).

In addition, we rely on satellite imagery provided by the authors of [HighlyAccurate](https://github.com/YujiaoShi/HighlyAccurate) and [PureACL](https://github.com/ShanWang-Shan/PureACL).
Please check their repositories for instructions on how to obtain the data. We thank both teams for making their data available, enabling reproducibility and fair comparison.

### Ford-AV
*TODO: Improve Downloadscript asserting desired structure and unzipping the folders.*
Download with:

```bash
bash scripts/setup/download_fordav.sh
```

### KITTI
*TODO: Improve Downloadscript by letting downloads run in parallel*
Download with:

```bash
bash scripts/setup/download_kitti.sh
```

After downloading, add the corresponding satellite imagery into the respective dataset log folders.
*TODO: Provide additional instructions and scripts for automated setup.*
*TODO: Add overview of the expected Datastructure*



#!/bin/bash

# Base output directory (everything goes under $HOME/Datasets)
BASEDIR="$HOME/Datasets"
OUTDIR="$BASEDIR/FordAV"
mkdir -p "$OUTDIR"

# Download calibration (placed alongside V2 logs)
CALIB_URL="https://ford-multi-av-seasonal.s3-us-west-2.amazonaws.com/Calibration/Calibration-V2.tar.gz"
echo "==> Downloading calibration..."
wget -c "$CALIB_URL" -P "$OUTDIR" &

# Available dates
DATES=("2017-08-04" "2017-10-26")

# Camera views
VIEWS=("FL" "FR" "SL" "SR" "RL" "RR")

# Logs 1–6 for each date and each view
for DATE in "${DATES[@]}"; do
    for LOG in {1..6}; do
        LOGDIR="$OUTDIR/V2/Log${LOG}"
        mkdir -p "$LOGDIR"
        for VIEW in "${VIEWS[@]}"; do
            FILE="${DATE}-V2-Log${LOG}-${VIEW}.tar.gz"
            URL="https://ford-multi-av-seasonal.s3-us-west-2.amazonaws.com/${DATE}/V2/Log${LOG}/${FILE}"
            echo "==> Downloading $FILE ..."
            wget -c "$URL" -P "$LOGDIR" &
        done
    done
done

# Wait for all parallel jobs to finish
wait

echo "==> Done! All files are in $OUTDIR"

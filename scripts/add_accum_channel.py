import argparse
import cv2
import datetime
import json
import numpy as np
import pandas as pd
import pathlib
import silhouettefile as sf
import sys

from PIL import Image, ImageSequence


def main(silhouette_main: pathlib.Path, tiff_source: pathlib.Path, remove_border: bool, output: pathlib.Path, overwrite: bool):
    f = sf.SilhouetteFile(silhouette_main)
    
    tiff_im = Image.open(tiff_source)
    ims = []
    for page in ImageSequence.Iterator(tiff_im):
        im = np.array(page, dtype=np.uint16) / 65535.0
        ims.append(im)
    im_shape = ims[0].shape

    new_columns = {}
    for channel_name in "rgb":
        new_columns[f"sum_{channel_name}"] = []
        new_columns[f"tmp_avg_{channel_name}"] = []
        new_columns[f"tmp_std_{channel_name}"] = []
    columns_order = ["sum_r", "sum_g", "sum_b", "tmp_avg_r", "tmp_avg_g", "tmp_avg_b", "tmp_std_r", "tmp_std_g", "tmp_std_b"]

    begin_time = datetime.datetime.now()
    df = f.data_frame
    for _, row in df.iterrows():
        layer_id = row.layer_id
        contour_id = row.contour_id

        count = len(new_columns[f"sum_r"]) + 1
        if count % 100 == 0:
            current = datetime.datetime.now()
            td = current-begin_time
            remain = (td / count) * (df.shape[0] - count)
            print(f"{count} of {df.shape[0]}, elapsed: {td}, remain: {remain}")

        channel_ims = {}
        for channel_id, channel_name in enumerate("rgb"):
            im = ims[layer_id * 3 + channel_id]
            channel_ims[channel_name] = im
        
        points = np.array(json.loads(row.points))
        tmp_im = np.zeros(im_shape, dtype=np.uint16)
        cv2.drawContours(tmp_im, [points], 0, contour_id, -1)
        if remove_border:
            cv2.drawContours(tmp_im, [points], 0, 0, 1)

        for channel_name, channel_im in channel_ims.items():
            ma = np.ma.array(channel_im, mask=tmp_im!=contour_id)
            new_columns[f"sum_{channel_name}"].append(ma.sum())
            new_columns[f"tmp_avg_{channel_name}"].append(ma.mean())
            new_columns[f"tmp_std_{channel_name}"].append(ma.std())
    
    df = pd.concat([df, pd.DataFrame(new_columns)[columns_order]], axis=1)

    if output.exists() and not overwrite:
        print(f"E: The file {output} exists. Remove it or use the --overwrite flag.", file=sys.stderr)
        return

    df.to_csv(output, index=False)
    print(f"I: the result has been written in {output} file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates the channel accumulation of a silhouette file")
    parser.add_argument("--main", type=pathlib.Path, help="Silhouette file", required=True)
    parser.add_argument("--tiff-source", type=pathlib.Path, help="TIFF file containing the stack source", required=True)
    parser.add_argument("--remove-border", help="remove the contour border before calculating the avg, std, and sum", action="store_true", default=False)
    parser.add_argument("--output", type=pathlib.Path, help="output CSV file", required=True)
    parser.add_argument("--overwrite", help="overwrite the output file if exists", action="store_true", default=False)
    args = parser.parse_args()

    main(args.main, args.tiff_source, args.remove_border, args.output, args.overwrite)
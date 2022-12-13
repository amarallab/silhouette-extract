import json
import pandas as pd
import pathlib


class SilhouetteFile:
    filename: pathlib.Path
    full_data: dict
    data_frame: pd.DataFrame

    def __init__(self, filename: pathlib.Path|str):
        if isinstance(filename, pathlib.Path):
            self.filename = filename
        elif type(filename) is str:
            self.filename = pathlib.Path(filename)
        else:
            raise ValueError(f"filename must be str or pathlib.Path")

        with open(self.filename / "feed.json") as f:
            feed = json.loads(f.read())
    
        layers = {}
        for layer_id in feed["layer_ids"]:
            with open(self.filename / f"{layer_id}.json") as f:
                layer = json.loads(f.read())
                layers[layer_id] = layer

        with open(self.filename / "feud.json") as f:
            feud = json.loads(f.read())

        feud_layers = {}
        for current in feud["layers"]:
            feud_layers[current["id"]] = current

        full_data = {}
        for layer_id, layer in layers.items():
            layer_data = {}
            for contour in layer["contours"]:
                layer_data[contour["id"]] = contour
            full_data[layer_id] = layer_data

        for layer_id, data in feud_layers.items():
            for contour in data.get("contours", []):
                contour_id = contour["id"]
                full_data[layer_id][contour_id]["label"] = contour["label"]

        data_frame = []
        for layer_id, layer in full_data.items():
            for contour_id, contour in layer.items():
                cx, cy = contour["centroid"]
                pixel_count = contour["pixel_count"]
                ar, ag, ab = contour["color_avg"]["r"], contour["color_avg"]["g"], contour["color_avg"]["b"]
                sr, sg, sb = contour["color_std"]["r"], contour["color_std"]["g"], contour["color_std"]["b"]
                label = contour.get("label", "")
                spoints = json.dumps(contour["points"]).strip()
                data_frame.append((layer_id, contour_id, cx, cy, pixel_count, ar, ag, ab, sr, sg, sb, spoints, label))

        self.full_data = full_data
        self.data_frame = pd.DataFrame(data_frame, columns=[
            "layer_id", "contour_id", 
            "centroid_x", "centroid_y", "pixel_count", 
            "avg_r", "avg_g", "avg_b", 
            "std_r", "std_g", "std_b", 
            "points", "label"])
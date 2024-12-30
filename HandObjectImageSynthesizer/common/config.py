import json
from easydict import EasyDict as edict


class Config():

    def __init__(self, json_path):
        with open(json_path) as f:
            self.cfg = json.load(f)
            self.cfg = edict(self.cfg)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.cfg, f, indent=4)

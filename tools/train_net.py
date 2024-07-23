import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.data import DatasetMapper
from detectron2.data.datasets import load_coco_json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.events import get_event_storage
import os
import sys
import pickle
import argparse
import torch
import time
import io
import numpy as np


_DATASETS = {
    'nucoco_mini_val': {
        'img_dir': '/home/live/RRPNv2/RRPN/data/nucoco/val',
        'ann_file': '/home/live/RRPNv2/RRPN/data/nucoco/annotations/instances_val.json',
    },
    'nucoco_mini_train': {
        'img_dir': '/home/live/RRPNv2/RRPN/data/nucoco/train',
        'ann_file': '/home/live/RRPNv2/RRPN/data/nucoco/annotations/instances_train.json',
    },
}

category_id_to_name = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck"}

def register_datasets():
    for dataset_name, dataset_info in _DATASETS.items():
        DatasetCatalog.register(dataset_name, lambda dataset_info=dataset_info: load_coco_json(dataset_info['ann_file'], dataset_info['img_dir']))
        MetadataCatalog.get(dataset_name).set(thing_classes=list(category_id_to_name.values()))

# Register the datasets
register_datasets()

def add_proposal_cfg(cfg):
    cfg.DATASETS.PROPOSAL_FILES_TRAIN = "/home/live/RRPNv2/RRPN/data/nucoco/proposals/proposals_mini_train.pkl"
    cfg.DATASETS.PROPOSAL_FILES_TEST = "/home/live/RRPNv2/RRPN/data/nucoco/proposals/proposals_mini_val.pkl"

class RadarDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True, proposal_files=None):
        super().__init__(cfg, is_train)
        self.proposal_files = proposal_files
        with open(self.proposal_files, 'rb') as f:
            self.proposals = pickle.load(f)
        
        self.proposal_ids = set(self.proposals['ids'])
        self.id_to_index = {img_id: idx for idx, img_id in enumerate(self.proposals['ids'])}

    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        image_id = dataset_dict["image_id"]
        if image_id in self.proposal_ids:
            idx = self.id_to_index[image_id]
            proposals = self.proposals['boxes'][idx]
            if np.isnan(proposals).any() or np.isinf(proposals).any():
                print(f"Proposals for image_id {image_id} contain NaN or Inf values")
                print(f"Proposals: {proposals}")
                raise ValueError(f"Proposals contain NaN or Inf for image_id {image_id}")
            dataset_dict["proposals"] = {
                "boxes": proposals,
                "scores": self.proposals['scores'][idx]
            }
        else:
            print(f"No proposals found for image_id {image_id}")
            raise ValueError(f"No proposals found for image_id {image_id}")

        return dataset_dict

# Verify Datasets
def verify_datasets():
    for dataset_name in _DATASETS.keys():
        dataset_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        print(f"Dataset: {dataset_name}, Number of samples: {len(dataset_dicts)}")
        print(f"Metadata: {metadata}")

        # Inspect some annotations
        for i, d in enumerate(dataset_dicts[:3]):
            print(f"Sample {i}: {d}")

train_proposals_path = "/home/live/RRPNv2/RRPN/data/nucoco/proposals/proposals_mini_train.pkl"
val_proposals_path = "/home/live/RRPNv2/RRPN/data/nucoco/proposals/proposals_mini_val.pkl"

def verify_proposals(proposals_path):
    with open(proposals_path, 'rb') as f:
        proposals = pickle.load(f)
        print(f"Proposals Keys: {proposals.keys()}")
        print(f"Number of Proposals: {len(proposals['ids'])}")
        print(f"Example Proposal: {proposals['boxes'][0]}")

# Validate Data Loader
def data_loader_test(cfg):
    print("Testing Train Data Loader")
    train_loader = build_detection_train_loader(cfg)
    for batch in train_loader:
        print(f"Train Data Batch: {batch}")
        break

    print("Testing Test Data Loader")
    test_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    for batch in test_loader:
        print(f"Test Data Batch: {batch}")
        break

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        proposal_files_train = cfg.DATASETS.PROPOSAL_FILES_TRAIN
        mapper = RadarDatasetMapper(cfg, is_train=True, proposal_files=proposal_files_train)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def run_step(self):
        assert self.model.training, "[CustomTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # print(f"Data Batch: {data}")

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        if torch.isnan(losses).any() or torch.isinf(losses).any():
            print(f"NaN or Inf detected in losses: {loss_dict}")
            raise ValueError("NaN or Inf detected in loss computation")

        self.optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def _write_metrics(self, loss_dict, data_time):
        metrics_dict = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time
        storage = get_event_storage()
        storage.put_scalars(**metrics_dict, smoothing_hint=False)

def setup(args):
    cfg = get_cfg()
    add_proposal_cfg(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def main(args):
    cfg = setup(args)
    
    # Verify datasets and proposals
    print("Verifying Datasets...")
    verify_datasets()
    print("Verifying Proposals...")
    verify_proposals(train_proposals_path)
    verify_proposals(val_proposals_path)
    
    # Test Data Loader
    print("Testing Data Loaders...")
    data_loader_test(cfg)
    
    # Train with Detailed Logging
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Detectron2 model")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)

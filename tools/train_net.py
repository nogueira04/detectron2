import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.data import DatasetMapper
from detectron2.data.datasets import load_coco_json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes, Instances
import optuna
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import os
import sys
import pickle
import argparse
import torch
import torch.multiprocessing
import time
import io
import numpy as np
import resource
import json
import cv2

torch.multiprocessing.set_sharing_strategy('file_system')

# Monitor File Descriptors
def check_open_fds():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Soft limit: {soft}, Hard limit: {hard}")
    print(f"Number of open file descriptors: {len(os.listdir('/proc/self/fd'))}")

check_open_fds()

_DATASETS = {
    'nucoco_val': {
        'img_dir': '/clusterlivenfs/gnmp/RRPN/data/nucoco/val',
        'ann_file': '/clusterlivenfs/gnmp/RRPN/data/nucoco/annotations/instances_val.json',
    },
    'nucoco_train': {
        'img_dir': '/clusterlivenfs/gnmp/RRPN/data/nucoco/train',
        'ann_file': '/clusterlivenfs/gnmp/RRPN/data/nucoco/annotations/instances_train.json',
    },
}

category_id_to_name = {0: "_", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "bus", 6: "truck"}

def register_datasets():
    for dataset_name, dataset_info in _DATASETS.items():
        # Register the dataset
        DatasetCatalog.register(dataset_name, lambda dataset_info=dataset_info: load_coco_json(dataset_info['ann_file'], dataset_info['img_dir']))
        # Set the metadata for the dataset (e.g., class names)
        MetadataCatalog.get(dataset_name).set(thing_classes=list(category_id_to_name.values()))

DatasetCatalog.clear()  # Clear any existing dataset registration
MetadataCatalog.clear()  # Clear existing metadata
register_datasets()

def add_proposal_cfg(cfg):
    cfg.DATASETS.PROPOSAL_FILES_TRAIN = ("/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_train.pkl", )
    cfg.DATASETS.PROPOSAL_FILES_TEST = ("/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_val.pkl", )

class RadarDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True, proposal_files=None, device=None):
        super().__init__(cfg, is_train)
        self.proposal_files = proposal_files
        self.device = device  # Store the device
        with open(self.proposal_files[0], 'rb') as f:
            self.proposals = pickle.load(f)
        
        self.proposal_ids = set(self.proposals['ids'])
        self.id_to_index = {img_id: idx for idx, img_id in enumerate(self.proposals['ids'])}

    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        image_id = dataset_dict["image_id"]
        if image_id in self.proposal_ids:
            idx = self.id_to_index[image_id]
            proposals = self.proposals['boxes'][idx]  # This should be a numpy array of shape [N, 4]
            scores = self.proposals['scores'][idx]     # This should be a corresponding numpy array of shape [N]

            # Check for NaN or Inf
            if np.isnan(proposals).any() or np.isinf(proposals).any():
                print(f"Proposals for image_id {image_id} contain NaN or Inf values")
                print(f"Proposals: {proposals}")
                raise ValueError(f"Proposals contain NaN or Inf for image_id {image_id}")

            # Convert proposals to a format Detectron2 expects
            dataset_dict["proposals"] = {
                "boxes": torch.tensor(proposals, dtype=torch.float32).to(self.device),
                "objectness_logits": torch.tensor(scores, dtype=torch.float32).to(self.device)
            }
            dataset_dict['proposals'] = Instances(dataset_dict['image'].shape[1:], proposal_boxes=Boxes(dataset_dict['proposals']['boxes']), objectness_logits=dataset_dict['proposals']['objectness_logits'])

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

train_proposals_path = ("/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_train.pkl", )
val_proposals_path = ("/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_val.pkl", )

def verify_proposals(proposals_path):
    check_open_fds()
    with open(proposals_path, 'rb') as f:
        proposals = pickle.load(f)
        print(f"Proposals Keys: {proposals.keys()}")
        print(f"Number of Proposals: {len(proposals['ids'])}")
        print(f"Example Proposal: {proposals['boxes'][0]}")
    check_open_fds()

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
    def __init__(self, cfg, debug=False, debug_interval=50):
        super().__init__(cfg)
        self.debug = False
        self.debug_dir = os.path.join(cfg.OUTPUT_DIR, "debug_data")
        self.debug_interval = debug_interval  # Interval for saving debug information
        self.iteration_count = 0  # Track iterations
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    @classmethod
    def build_train_loader(cls, cfg):
        proposal_files_train = cfg.DATASETS.PROPOSAL_FILES_TRAIN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mapper = RadarDatasetMapper(cfg, is_train=True, proposal_files=proposal_files_train, device=device)
        return build_detection_train_loader(cfg, mapper=mapper, num_workers=1)

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

        # Save debug information at specific intervals
        if self.debug and (self.iteration_count % self.debug_interval == 0):
            self._save_debug_info(data)

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

        # Increment the iteration counter
        self.iteration_count += 1

    def _write_metrics(self, loss_dict, data_time):
        metrics_dict = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time
        storage = get_event_storage()
        storage.put_scalars(**metrics_dict, smoothing_hint=False)

    def _save_debug_info(self, data, limit_per_batch=2):
        saved_count = 0  

        for i, batch in enumerate(data):
            if saved_count >= limit_per_batch:
                break  

            image_id = batch["image_id"]
            image = batch["image"].permute(1, 2, 0).cpu().numpy()  
            image = (image - image.min()) / (image.max() - image.min())  
            image = (image * 255).astype(np.uint8)  
            
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            image_folder = os.path.join(self.debug_dir, f"image_{image_id}_{i}")
            os.makedirs(image_folder, exist_ok=True)

            original_image_file = os.path.join(image_folder, "original_image.png")
            cv2.imwrite(original_image_file, image)

            # Save ground truth boxes and classes
            gt_boxes = batch["instances"].gt_boxes.tensor.cpu().numpy()
            gt_classes = batch["instances"].gt_classes.cpu().numpy()

            for box, cls in zip(gt_boxes, gt_classes):
                x0, y0, x1, y1 = box.astype(int)
                class_name = self.metadata.thing_classes[cls]
                color = (0, 255, 0)  # Green for ground truth
                cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
                cv2.putText(image, class_name, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            gt_image_file = os.path.join(image_folder, "gt_image.png")
            cv2.imwrite(gt_image_file, image)

            gt_data = {
                "image_id": image_id,
                "gt_boxes": gt_boxes.tolist(),
                "gt_classes": gt_classes.tolist(),
                "gt_class_names": [self.metadata.thing_classes[cls] for cls in gt_classes],
            }
            gt_file = os.path.join(image_folder, "debug_data.json")
            with open(gt_file, 'w') as f:
                json.dump(gt_data, f, indent=4)

            if "proposals" in batch:
                proposals_boxes = batch["proposals"].proposal_boxes.tensor
                proposals_scores = batch["proposals"].objectness_logits

                if isinstance(proposals_boxes, torch.Tensor):
                    proposals_boxes = proposals_boxes.cpu().numpy()
                
                if isinstance(proposals_scores, torch.Tensor):
                    proposals_scores = proposals_scores.cpu().numpy()

                for box, score in zip(proposals_boxes, proposals_scores):
                    x0, y0, x1, y1 = box.astype(int)
                    color = (255, 0, 0)  # Blue for proposals
                    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(image, f"{score:.2f}", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                proposals_image_file = os.path.join(image_folder, "proposals_image.png")
                cv2.imwrite(proposals_image_file, image)    

                proposals_data = {
                    "image_id": image_id,
                    "proposal_boxes": proposals_boxes.tolist(),
                    "proposal_scores": proposals_scores.tolist(),
                }
                proposals_file = os.path.join(image_folder, "proposals_data.json")
                with open(proposals_file, 'w') as f:
                    json.dump(proposals_data, f, indent=4)

            saved_count += 1

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
    
    debug = 'DEBUG' in args.opts

    trainer = CustomTrainer(cfg, debug=debug)
    trainer.resume_or_load(resume=False)
    trainer.train()

# def objective(trial):
#     cfg = get_cfg()
#     add_proposal_cfg(cfg)
#     cfg.set_new_allowed(True)
#     cfg.merge_from_file(args.config_file)
    
#     cfg.SOLVER.BASE_LR = trial.suggest_loguniform("lr   ", 1e-5, 1e-3)
#     cfg.SOLVER.WEIGHT_DECAY = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
#     cfg.SOLVER.MOMENTUM = trial.suggest_float("momentum", 0.8, 0.99)
#     # cfg.SOLVER.IMS_PER_BATCH = trial.suggest_categorical("batch_size", [2, 4, 8])
#     cfg.MODEL.BACKBONE.FREEZE_AT = trial.suggest_categorical("freeze_at", [0, 2, 4])
    
#     cfg.DATASETS.TRAIN = ("nucoco_train",)
#     cfg.DATASETS.TEST = ("nucoco_val",)
#     cfg.DATALOADER.NUM_WORKERS = 4
#     cfg.SOLVER.MAX_ITER = 40000
#     cfg.OUTPUT_DIR = f"./output_trial_{trial.number}"
    
#     trainer = CustomTrainer(cfg)
#     trainer.resume_or_load(resume=False)
    
#     try:
#         trainer.train()
#         evaluator = COCOEvaluator("nucoco_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
#         val_loader = build_detection_test_loader(cfg, "nucoco_val")
#         eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)

#         print(f"Trial {trial.number}, {eval_results}")
        
#         return eval_results["bbox"]["AP"]
    
#     except Exception as e:
#         print(f"Error during training: {e}")
#         return float("nan")

def objective(trial):
    cfg = get_cfg()
    add_proposal_cfg(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    
    # Learning rate and weight decay
    cfg.SOLVER.BASE_LR = trial.suggest_loguniform("base_lr", 1e-5, 1e-3)
    cfg.SOLVER.WEIGHT_DECAY = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    cfg.SOLVER.WEIGHT_DECAY_NORM = trial.suggest_loguniform("weight_decay_norm", 1e-6, 1e-3)
    
    # Momentum (only used if SGD is chosen, but we can still tune it)
    cfg.SOLVER.MOMENTUM = trial.suggest_float("momentum", 0.8, 0.99)
    
    # Steps, Max Iter, Scheduler Steps
    # For steps, we can choose a few tuples as candidates. Make sure it aligns with MAX_ITER.
    # cfg.SOLVER.STEPS = trial.suggest_categorical("steps", [(5000, 20000), (10000, 15000), (20000, 30000)])
    cfg.SOLVER.SCHEDULER_STEPS = cfg.SOLVER.STEPS  # To keep them aligned, or you could suggest separately.
    
    # Learning rate scheduler and warmup parameters
    cfg.SOLVER.LR_SCHEDULER_NAME = trial.suggest_categorical("lr_scheduler_name", ["WarmupMultiStepLR", "WarmupCosineLR"])
    cfg.SOLVER.WARMUP_FACTOR = trial.suggest_float("warmup_factor", 1e-4, 1e-1, log=True)
    cfg.SOLVER.WARMUP_ITERS = trial.suggest_int("warmup_iters", 100, 2000, step=100)
    cfg.SOLVER.WARMUP_METHOD = trial.suggest_categorical("warmup_method", ["linear", "constant"])
    
    # Gradient clipping
    clip_grad_enabled = trial.suggest_categorical("clip_grad_enabled", [True, False])
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = clip_grad_enabled
    if clip_grad_enabled:
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = trial.suggest_float("clip_value", 0.1, 10.0, log=True)
    else:
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    
    # Optimizer selection
    cfg.SOLVER.OPTIMIZER = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    
    # Gamma (used for step-based LR decay if applicable)
    cfg.SOLVER.GAMMA = trial.suggest_float("gamma", 0.05, 0.5)
    
    # Freeze at layers (existing from your original code)
    cfg.MODEL.BACKBONE.FREEZE_AT = trial.suggest_categorical("freeze_at", [0, 2, 4])
    
    # Dataset configuration (fixed)
    cfg.DATASETS.TRAIN = ("nucoco_train",)
    cfg.DATASETS.TEST = ("nucoco_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Output directory
    cfg.OUTPUT_DIR = f"./output_trial_{trial.number}"
    
    # Train the model
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    try:
        trainer.train()
        evaluator = COCOEvaluator("nucoco_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, "nucoco_val")
        eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)

        print(f"Trial {trial.number}, {eval_results}")
        
        # Return the primary metric (assuming bbox AP is what we want)
        return eval_results["bbox"]["AP"]
    
    except Exception as e:
        print(f"Error during training: {e}")
        return float("nan")


def main_with_optuna(args):
    storage = "sqlite:///optuna_study.db"
    
    study = optuna.create_study(direction="maximize", study_name="detectron_hpo", storage=storage, load_if_exists=True)
    
    study.optimize(objective, n_trials=15)
    
    print("Best Trial:")
    print(study.best_trial)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Train a Detectron2 model")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print("Command Line Args:", args)
    # main(args)
    main_with_optuna(args)

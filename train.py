import argparse
import getpass
import contextlib
import importlib
import logging
import os
import sys
import time
import re
import datetime
import traceback
import pytorch_lightning as pl
import torch

# Fix PyTorch 2.6+ weights_only=True default breaking OmegaConf in checkpoints
try:
    import omegaconf
    torch.serialization.add_safe_globals([
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
        omegaconf.omegaconf.OmegaConf,
        omegaconf.base.ContainerMetadata,
        omegaconf.base.Node,
        omegaconf._utils._get_value,
        omegaconf.base.Container,
        omegaconf.nodes.AnyNode,
        omegaconf.nodes.StringNode,
        omegaconf.nodes.IntegerNode,
        omegaconf.nodes.FloatNode,
        omegaconf.nodes.BooleanNode,
    ])
except (ImportError, AttributeError):
    pass
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import step1x3d_geometry
import step1x3d_geometry.datamodules.mesh_motion
import step1x3d_geometry.systems.mesh_motion_autoencoder
from step1x3d_geometry.systems.base import BaseSystem
from step1x3d_geometry.utils.callbacks import (
    EarlyEnvironmentSetter,
    CodeSnapshotCallback,
    ConfigSnapshotCallback,
    CustomProgressBar,
    ProgressCallback,
    S3SyncCallback,
)
from step1x3d_geometry.utils.ema import EMA, EMAModelCheckpoint
from step1x3d_geometry.utils.config import ExperimentConfig, load_config
from step1x3d_geometry.utils.misc import get_rank
from step1x3d_geometry.utils.typing import Optional


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"

    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def load_custom_module(module_path):
    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
    try:
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(
                module_name, module_path
            )
        else:
            module_spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(module_path, "__init__.py")
            )

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(traceback.format_exc())
        print(f"Cannot import {module_path} module for custom nodes:", e)
        return False


def load_custom_modules():
    node_paths = ["custom"]
    node_import_times = []
    if not os.path.exists("node_paths"):
        return
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if (
                os.path.isfile(module_path)
                and os.path.splitext(module_path)[1] != ".py"
            ):
                continue
            if module_path.endswith(".disabled"):
                continue
            time_before = time.perf_counter()
            success = load_custom_module(module_path)
            node_import_times.append(
                (time.perf_counter() - time_before, module_path, success)
            )

    if len(node_import_times) > 0:
        print("\nImport times for custom modules:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()


def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True  # enable cudnn benchmark for optimal performance

    # 🔧 多节点CUDA设备配置修复
    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    
    if len(env_gpus) > 0:
        n_gpus = len(env_gpus)
        devices = n_gpus  # 使用实际GPU数量而不是-1
        print(f"🔧 Using CUDA_VISIBLE_DEVICES: {env_gpus_str} ({n_gpus} GPUs)")
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        devices = n_gpus  # 使用实际GPU数量而不是-1
        print(f"🔧 Using {n_gpus} GPUs: {selected_gpus}")
        # 为多节点训练设置CUDA_VISIBLE_DEVICES
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    load_custom_modules()

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus, resume=args.resume)

    # set a different seed for each device
    rank = get_rank()
    pl.seed_everything(cfg.seed + rank, workers=True)

    # 🚀 多节点环境变量设置
    if cfg.trainer.get("num_nodes", 1) > 1:
        # 多节点训练时设置环境变量
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", n_gpus * cfg.trainer.num_nodes))
        node_rank = int(os.environ.get("NODE_RANK", 0))
        
        print(f"🌐 Multi-node training detected:")
        print(f"  - Nodes: {cfg.trainer.num_nodes}")
        print(f"  - GPUs per node: {cfg.trainer.get('devices', n_gpus)}")
        print(f"  - Total GPUs: {cfg.trainer.num_nodes * cfg.trainer.get('devices', n_gpus)}")
        print(f"  - Current node rank: {node_rank}")
        print(f"  - Local rank: {local_rank}")
        print(f"  - Global rank: {os.environ.get('RANK', 'unknown')}")
        
        # 确保PyTorch能正确识别分布式环境
        if "MASTER_ADDR" not in os.environ:
            print("⚠️  Warning: MASTER_ADDR not set, using localhost")
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            print("⚠️  Warning: MASTER_PORT not set, using 29500")
            os.environ["MASTER_PORT"] = "29500"
    else:
        print(f"🖥️  Single-node training with {n_gpus} GPUs")

    dm = step1x3d_geometry.find(cfg.data_type)(
        overfit_single=args.overfit_single,
        **cfg.data
    )
    system: BaseSystem = step1x3d_geometry.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    if args.gradio:
        fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs"))
        fh.setLevel(logging.INFO)
        if args.verbose:
            fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    callbacks = []
    if args.train:
        if not args.use_ema:
            callbacks += [
                EarlyEnvironmentSetter(),
                ModelCheckpoint(
                    dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
                ),
                LearningRateMonitor(logging_interval="step"),
                CodeSnapshotCallback(
                    os.path.join(cfg.trial_dir, "code"), use_version=False
                ),
                ConfigSnapshotCallback(
                    args.config,
                    cfg,
                    os.path.join(cfg.trial_dir, "configs"),
                    use_version=False,
                ),
            ]
        else:
            callbacks += [
                EarlyEnvironmentSetter(),
                EMAModelCheckpoint(
                    dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
                ),
                EMA(decay=0.9999),
                LearningRateMonitor(logging_interval="step"),
                CodeSnapshotCallback(
                    os.path.join(cfg.trial_dir, "code"), use_version=False
                ),
                ConfigSnapshotCallback(
                    args.config,
                    cfg,
                    os.path.join(cfg.trial_dir, "configs"),
                    use_version=False,
                ),
            ]

        if cfg.s3_trial_dir:
            callbacks.append(S3SyncCallback(
                local_dir=cfg.trial_dir,
                s3_path=cfg.s3_trial_dir,
            ))
            rank_zero_only(lambda: print(f"S3 sync enabled: {cfg.trial_dir} -> {cfg.s3_trial_dir}"))()

        if args.use_swa:
            assert args.use_ema is not True
            print("enable SWA callback")
            callbacks += [StochasticWeightAveraging(swa_lrs=1e-2)]

        if args.gradio:
            callbacks += [
                ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))
            ]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            CSVLogger(cfg.trial_dir, name="csv_logs"),
        ] + system.get_loggers()
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()

    # 🚀 多节点trainer配置
    # 🔧 多节点trainer配置 - 避免设备冲突
    trainer_config = dict(cfg.trainer)
    
    # 多节点训练时，使用"auto"让PyTorch Lightning自动检测可用GPU
    if trainer_config.get("num_nodes", 1) > 1:
        final_devices = "auto"  # 让Lightning自动检测
        print(f"🔧 Multi-node mode: using devices='auto' for automatic GPU detection")
    else:
        # 单节点时使用配置文件中的设备数或代码计算的devices
        final_devices = trainer_config.get("devices", devices)
        print(f"🔧 Single-node mode: using devices={final_devices}")
    
    trainer_kwargs = dict(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=final_devices,
        use_distributed_sampler=True,  # 🔥 Lightning 2.x参数：自动管理DistributedSampler
        **{k: v for k, v in trainer_config.items() if k != "devices"},
        # profiler="pytorch",
    )
    
    # 如果是多节点训练，配置DDP策略
    if cfg.trainer.get("num_nodes", 1) > 1:
        print(f"🚀 Configuring multi-node DDP strategy for {cfg.trainer.num_nodes} nodes")
        # 使用DDP策略，默认不查找未使用参数（可根据需要调整）
        if "strategy" not in cfg.trainer or cfg.trainer.strategy == "ddp":
            trainer_kwargs["strategy"] = DDPStrategy(
                find_unused_parameters=False,  # 设为True如果有未使用参数
                static_graph=False,  # 设为True如果模型图结构固定
                broadcast_buffers=False,  # 关闭缓冲区广播，避免与梯度allreduce交错
            )
        elif cfg.trainer.strategy == "ddp_find_unused_parameters_true":
            trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)
    
    trainer = Trainer(**trainer_kwargs)

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        # PyTorch 2.6 defaults weights_only=True which breaks OmegaConf objects in ckpt
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch without weights_only arg
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            print(f"[WARN] Failed to fully load checkpoint metadata: {e}. Skipping resume status set.")
            return
        try:
            system.set_resume_status(ckpt.get("epoch", 0), ckpt.get("global_step", 0))
        except Exception as e:
            print(f"[WARN] Could not set resume status from checkpoint: {e}")

    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
        if args.gradio:
            # also export assets if in gradio mode
            trainer.predict(system, datamodule=dm)
    elif args.validate:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        # Ensure datamodule is set up before building custom loaders
        try:
            dm.setup(None)
        except Exception:
            pass

        # 1) Evaluate on train split
        try:
            if hasattr(system, 'set_eval_split'):
                system.set_eval_split('train')
            trainer.test(system, dataloaders=dm.train_dataloader(), ckpt_path=cfg.resume)
        except Exception as e:
            print(f"[WARN] Train-split testing failed: {e}")

        # 2) Evaluate on val split
        try:
            if hasattr(system, 'set_eval_split'):
                system.set_eval_split('val')
            trainer.test(system, dataloaders=dm.val_dataloader(), ckpt_path=cfg.resume)
        except Exception as e:
            print(f"[WARN] Val-split testing failed: {e}")
    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument(
        "--overfit_single", action="store_true", help="if true, overfit on single sequence"
    )

    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser.add_argument(
        "--use_ema", action="store_true", help="if true, use EMA during training"
    )
    parser.add_argument(
        "--use_swa", action="store_true", help="if true, use SWA during training"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to checkpoint to resume from"
    )
    args, extras = parser.parse_known_args()

    if args.gradio:
        with contextlib.redirect_stdout(sys.stderr):
            main(args, extras)
    else:
        main(args, extras)

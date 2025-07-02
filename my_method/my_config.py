"""my_method/my_config.py"""

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from .custom_model import CustomModelConfig

MyMethod = MethodSpecification(
    config=TrainerConfig(
        method_name="my-method",
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=ColmapDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=CustomModelConfig(
                num_layers=8,
                hidden_dim=256,
                out_dim=4,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5,
                    max_steps=30000,
                ),
            }
        },
        max_num_iterations=30000,
        mixed_precision=True,
        use_grad_scaler=True,
    ),
    description="A simple custom NeRF implementation",
)
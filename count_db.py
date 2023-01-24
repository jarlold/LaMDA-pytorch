import colossalai
import torch
import wandb

from colossalai.amp import AMP_TYPE, convert_to_amp
from colossalai.core import global_context as gpc
from colossalai.trainer import Trainer, hooks
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import MultiTimer, save_checkpoint
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer

from lamda_pytorch.config.config import CFG
from lamda_pytorch.build_dataloader import build_dataloaders
from lamda_pytorch.lamda_pytorch import lamda_model
from lamda_pytorch.utils.utils import LaMDA_Loss, AutoregressiveWrapper

train_dataloader, eval_dataloader = build_dataloaders(cfg, tokenizer)
print("Train length ", len(train_dataloader))
print("Eval length ", len(eval_dataloader))

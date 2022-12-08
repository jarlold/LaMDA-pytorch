from colossalai.zero.shard_utils import TensorShardStrategy

zero = dict(
    model_config = dict(
        shard_strategy = TensorShardStrategy(),
        tensor_placement_policy = 'cpu',
        reuse_fp16_shard = False
    )
)

EPOCHS = 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-2
gradient_accumulation = 4
clip_grad_norm = 1.0
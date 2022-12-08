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

def LaMDA_Trainer(cfg: CFG):
    assert torch.cuda.is_available()
    disable_existing_loggers()

    parser = colossalai.get_default_parser()

    parser.add_argument(
        '--use_trainer',
        action='store_true',
        help='whether to use trainer'
    )

    args = parser.parse_args()

    # Defining ZeRO in the config is what is meant to be done when working at a "high-level" as the ColossalAI docs put it
    # Using Weights & Biases however requires the training to be done at "low-level" - that is, manually define the training loop.
    # ZeRO can still be used in this case, but if W&B is enabled, it will have to be implemented at the low-level.
    if cfg.use_zero and not cfg.use_wandb:
        colossalai.launch_from_torch(
            config='./lamda_pytorch/config/zero_config.py', 
            seed = cfg.seed
        )
    else:
        colossalai.launch_from_torch(
            config='./lamda_pytorch/config/colossal_config.py', 
            seed = cfg.seed
        )

    assert hasattr(gpc.config, "EPOCHS"), "Please provide NUM_EPOCHS in your configuration"

    # Colossal logger
    logger = get_dist_logger()
    logger.info("Initialized environment", ranks=[0])

    # LaMDA model
    if cfg.use_zero:
        with ZeroInitContext(target_device = torch.cuda.current_device(),
                            shard_strategy=gpc.config.zero.model_config.shard_strategy,
                            shard_param=True):
            model = AutoregressiveWrapper(lamda_model())
            # If this is being done at low level, model needs to be manually sharded
            if cfg.use_wandb:
                model = ShardedModelV2(model, TensorShardStrategy(), tensor_placement_policy='cpu', reuse_fp16_shard = True)
            
    else:
        model = lamda_model()
        model = AutoregressiveWrapper(model)

    # setup dataloaders
    
    # Honestly, setting cfg.use_huggingface to False would literally break everything, so there's really no point in even checking at this point.
    # if cfg.use_huggingface == True:
    if cfg.tokenizer_name == "sentencepiece":
        tokenizer = SentencePieceProcessor()
        tokenizer.load('wikipedia_32k_tokenizer.model')
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    
    train_dataloader, eval_dataloader = build_dataloaders(cfg, tokenizer)

    # loss function
    loss_fn = LaMDA_Loss()

    # optimizer function, apply FP16 computation if wanted
    if cfg.use_fp16:
        optimizer = HybridAdam(
            model.parameters(),
            lr = gpc.config.LEARNING_RATE,
            weight_decay=gpc.config.WEIGHT_DECAY
        )
        # once again, if using ZeRO, shard optimizer
        if cfg.use_zero:
            optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)
        
        model, _, loss_fn = convert_to_amp(model, optimizer, loss_fn, AMP_TYPE.NAIVE) # do not return 2nd parameter since HybridAdam is already optimized for FP16 (?)
        
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr = gpc.config.LEARNING_RATE,
            weight_decay=gpc.config.WEIGHT_DECAY
        )
    

    # initialize model, optimizer, criterion, and data loaders

    engine, train_dataloader, _, _ = colossalai.initialize(
        model,
        optimizer,
        loss_fn,
        train_dataloader = train_dataloader
    )

    def batch_data_process_func(batch_data):
        data = batch_data["input_ids"]
        labels = batch_data["labels"]
        return data, labels

    engine.schedule.data_process_func = batch_data_process_func

    if cfg.use_wandb:
        # wandb docs suggested to make a config dict, so here's a big fuck-off config dict
        # probably has some use idk
        wandb_config = {
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.lr,
            "weight_decay": gpc.config.WEIGHT_DECAY,
            "clip_grad_norm": gpc.config.clip_grad_norm,
            "grad_accumulation": gpc.config.gradient_accumulation,
            "vocab_size": cfg.num_tokens,
            "embed_dim": cfg.dim,
            "num_layers": cfg.depth,
            "num_attention_heads": cfg.heads,
            "dim_head": cfg.dim_head,
            "tokenizer": cfg.tokenizer_name,
        }
        
        # initialize Weights and Biases Logging
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=wandb_config)
        
        print(f"Number of parameters in current LaMDA model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        for epoch in range(gpc.config.EPOCHS):
            print(f"\nBeginning epoch {epoch} of training...")
            
            for step, batch in enumerate(train_dataloader):
                engine.train()
                inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()
            
                engine.zero_grad()
                outputs = engine(inputs)

                train_loss = engine.criterion(outputs, labels)
                wandb.log({"train_loss": train_loss})

                engine.backward(train_loss)
                engine.step()
                wandb.log({"steps": (step * (epoch + 1))})
                
                # After 1 iter of training, we do 1 iter of testing.
                # Temporarily moved here because 1 training run through the dataloader takes too long.
                # As long as the length of the dataloaders are the same there shouldn't be any problems
                engine.eval()
                batch = next(iter(eval_dataloader))
                inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()
                
                with torch.no_grad():
                    outputs = engine(inputs)
                    test_loss = engine.criterion(outputs, labels)
                    wandb.log({"test_loss": test_loss})
                    # Calculate perplexity
                    perplexity = torch.exp(test_loss)
                    wandb.log({"perplexity": perplexity})
            
            # This will be uncommented when we actually have to train stuff properly
            #print("Entering testing stage...")
            #engine.eval()
            #for step, batch in enumerate(eval_dataloader):
            #    inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()

            #    with torch.no_grad():
            #        outputs = engine(inputs)
            #        test_loss = engine.criterion(outputs, labels)
            #        wandb.log({"test_loss": test_loss})
            #        # Calculate perplexity
            #        perplexity = torch.exp(test_loss)
            #        wandb.log({"perplexity": perplexity})
                
                    # engine.backward(test_loss)
                    # engine.step()
                    
            # Save model
            if cfg.save_model and epoch % cfg.save_every_n_epoches == 0:
                save_checkpoint(f'LaMDA_EPOCH_{epoch}.pt', epoch, model)

        wandb.alert(
            title = 'Training Complete',
            text = "Training complete."
        )

    else:
        # Time session with ColossalAI
        timer = MultiTimer()

        # trainer
        trainer = Trainer(
            engine = engine,
            timer =  timer,
            logger = logger
        )

        hook_list = [
            hooks.LogMetricByStepHook(),
            hooks.LossHook(),
            hooks.LogMetricByEpochHook(logger)
        ]
        
        # save checkpoint
        if cfg.save_model:
            hook_list.append(hooks.SaveCheckpointHook(cfg.save_every_n_epoches, f'LaMDA_CURRENT_CKPT.pt', model))

        trainer.fit(
            train_dataloader = train_dataloader,
            epochs = gpc.config.EPOCHS,
            hooks = hook_list,
            display_progress = True
        )

if __name__ == "__main__":

    cfg = CFG()

    LaMDA_Trainer(cfg)
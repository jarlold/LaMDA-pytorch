from dataclasses import dataclass, field
from typing import Optional, ClassVar, List

@dataclass
class CFG:

    """
    Configuration for ZeRO
    """

    use_zero: bool = field(
        default = False,
        metadata = {'help': 'whether to use zero'}
    )

    """
    Configuration for optimizer
    """

    lr: float = field(
        default = 0.0001,
        metadata = {'help': 'learning rate'}
    )

    """
    Configuration class for LaMDA model.
    """

    num_tokens: int = field(
        default = 32000,
        metadata = {'help': 'number of tokens'}
    )

    dim: int = field(
        default = 256,
        metadata = {'help': 'dimension of the embedding'}
    )

    depth: int = field(
        default = 5,
        metadata = {'help': 'depth of the transformer'}
    )

    heads: int = field(
        default = 4,
        metadata = {'help': 'number of heads in the transformer'}
    )

    dim_head: int = field(
        default = 64,
        metadata = {'help': 'dimension of the head'}
    )

    """
    Configuration for data loader.
    """

    use_huggingface: bool = field(
        default = True,
        metadata = {'help': 'Whether to use huggingface datasets'}
    )

    train_dataset_path: Optional[str] = field(
        default="the_pile", 
        metadata={"help": "Path to Hugging Face training dataset."}
    )

    eval_dataset_path: Optional[str] = field(
        default="conceptofmind/pile_enron_emails", 
        metadata={"help": "Path to Hugging Face validation dataset."}
    )
    
    train_dataset_name: Optional[str] = field(
        default="hacker_news", 
        metadata={"help": "Path to Hugging Face training dataset directory."}
    )

    eval_dataset_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to Hugging Face validation dataset directory."}
    )

    choose_train_split: Optional[str] = field(
        default="train", 
        metadata={"help": "Choose Hugging Face training dataset split."}
    )

    choose_eval_split: Optional[str] = field(
        default="train", 
        metadata={"help": "Choose Hugging Face validation dataset split."}
    )

    remove_train_columns: ClassVar[List[str]] = field(
        default = ['meta'], 
        metadata={"help": "Train dataset columns to remove."}
    )

    remove_eval_columns: ClassVar[List[str]] = field(
        default = ['meta'], 
        metadata={"help": "Validation dataset columns to remove."}
    )

    seed: Optional[int] = field(
        default=42, 
        metadata={"help": "Random seed used for reproducibility."}
    )

    tokenizer_name: Optional[str] = field(
        default="sentencepiece",
        metadata={"help": "Tokenizer name."}
    )

    tokenizer_seq_length: Optional[int] = field(
        default=512, 
        metadata={"help": "Sequence lengths used for tokenizing examples."}
    )

    select_input_string: Optional[str] = field(
        default="text", 
        metadata={"help": "Select the key to used as the input string column."}
    )
    
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Batch size for training and validation."}
    )
    # Setting this to True will break everything.
    # Do not set this to True if you do not want to break everything.
    stream_data: bool = field(
        default=False,
        metadata={"help": "Use HuggingFace's dataset streaming feature."}
    )

    save_to_path: Optional[str] = field(
        default="''",
        metadata={"help": "Save the dataset to local disk."}
    )

    """
    Configuration for Weights and Biases
    """

    use_wandb: bool = field(
        default = False,
        metadata = {'help': 'Whether to use Weights and Biases for logging'}
    )

    project_name: str = field(
        default="lamda-enron-trainingtest",
        metadata = {'help': 'Name of the project'}
    )
    
    run_name: str = field(
        default="default-run",
        metadata = {'help': 'Name of the current run (please change before use)'}
    )
    
    """
    Configuration for training
    """
    
    save_model: bool = field(
        default = True,
        metadata = {'help': 'Save model during training'}
    )
    
    save_every_n_epoches: int = field(
        default = 1,
        metadata = {'help': 'If save_model is enabled, saves the model to a checkpoint after this many epoches since the last save'}
    )
    
    use_fp16: bool = field(
        default = False,
        metadata = {'help': 'Use FP16 for training'}
    )

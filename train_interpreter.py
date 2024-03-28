import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, AutoTokenizer

from datamodule import SarcasmDataModule
from models import Model
import data_utils

torch.set_float32_matmul_precision('medium')

pl.seed_everything(42, workers=True)

data = pd.read_excel('/blue/cai6307/n.kolla/data/PreFinalTranslation.xlsx', names=['sarcasm', 'interpretation', 'translation'])
interpreter_data = pd.DataFrame()

interpreter_input_prefix = 'interpret sarcasm: ' # Needed as the models are pretrained for multiple tasks
interpreter_data['inputs'] = data.sarcasm.apply(lambda text: data_utils.preprocess(text, interpreter_input_prefix))
interpreter_data['targets'] = data.interpretation.apply(data_utils.preprocess)

interpreter_model_name = 'google-t5/t5-large'

interpreter_tokenizer = AutoTokenizer.from_pretrained(interpreter_model_name, model_max_length=512)

interpreter_collator = DataCollatorForSeq2Seq(tokenizer=interpreter_tokenizer, model=interpreter_model_name)

interpreter_datamodule = SarcasmDataModule(interpreter_data, interpreter_tokenizer, interpreter_collator)

interpreter_model = Model(interpreter_model_name, lr=1e-5)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)

interpreter_checkpoint = ModelCheckpoint(
    dirpath='/blue/cai6307/n.kolla/finetune_ckpts/interpreters',
    filename='{val_loss:.3f}_{epoch}_{step}_model=' + interpreter_model_name.split("/")[-1],
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=False,
    every_n_epochs=1,
    verbose=False,
    save_last=False,
    enable_version_counter=True
)

interpreter_trainer = pl.Trainer(
    max_epochs=30, 
    accelerator='gpu', 
    devices=-1, 
    callbacks=[interpreter_checkpoint, early_stopping],
    default_root_dir="/blue/cai6307/n.kolla/logs",
    deterministic=True, # To ensure reproducability
)

interpreter_trainer.fit(interpreter_model, datamodule=interpreter_datamodule)

# interpreter_trainer.test(model=interpreter_model, datamodule=interpreter_datamodule, ckpt_path='best')
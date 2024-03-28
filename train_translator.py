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
translator_data = pd.DataFrame()

translator_input_prefix = 'translate English to Telugu: ' # Needed as the models are pretrained for multiple tasks
translator_data['inputs'] = data.interpretation.apply(lambda text: data_utils.preprocess(text, translator_input_prefix))
translator_data['targets'] = data.translation.apply(data_utils.preprocess)

translator_model_name = 'facebook/mbart-large-50-many-to-many-mmt'

translator_tokenizer = AutoTokenizer.from_pretrained(translator_model_name, model_max_length=512)

if 'mbart' in translator_model_name:
    translator_tokenizer.src_lang = 'en_XX'
    translator_tokenizer.tgt_lang = 'te_IN'

translator_collator = DataCollatorForSeq2Seq(tokenizer=translator_tokenizer, model=translator_model_name)

translator_datamodule = SarcasmDataModule(translator_data, translator_tokenizer, translator_collator)

translator_model = Model(translator_model_name, lr=1e-5)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)

translator_checkpoint = ModelCheckpoint(
    dirpath='/blue/cai6307/n.kolla/finetune_ckpts/translators',
    filename='{val_loss:.3f}_{epoch}_{step}_model=' + translator_model_name.split("/")[-1],
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=False,
    every_n_epochs=1,
    verbose=False,
    save_last=False,
    enable_version_counter=True
)

translator_trainer = pl.Trainer(
    max_epochs=50, 
    accelerator='gpu', 
    devices=-1, 
    callbacks=[translator_checkpoint, early_stopping],
    default_root_dir="/blue/cai6307/n.kolla/logs",
    deterministic=True, # To ensure reproducability
)

translator_trainer.fit(translator_model, datamodule=translator_datamodule)

# translator_trainer.test(model=translator_model, datamodule=translator_datamodule, ckpt_path='/blue/cai6307/n.kolla/logs/lightning_logs/version_5/checkpoints/val_loss=1.499_epoch=14_step=1410_model=mbart-large-50-many-to-many-mmt.ckpt')
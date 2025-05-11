import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Optional, List
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel, AdapterFusionConfig
from omegaconf import OmegaConf

class TextDataset(Dataset):
    def __init__(self, data: list, tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["input"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = self.tokenizer(
            item["target"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        return enc

class AdapterTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for adapter (LoRA/AdapterFusion) fine-tuning.
    """
    def __init__(self, config_path: str, train_data: list, val_data: Optional[list] = None):
        super().__init__()
        self.save_hyperparameters()
        self.config = OmegaConf.load(config_path)
        self.model_name = self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if getattr(self.config, "use_adapter_fusion", False):
            fusion_config = AdapterFusionConfig()
            self.model = PeftModel.from_pretrained(self.model, self.config.fusion_adapter_names, adapter_fusion_config=fusion_config)
        else:
            lora_cfg = LoraConfig(**self.config.lora)
            self.model = get_peft_model(self.model, lora_cfg)
        self.train_dataset = TextDataset(train_data, self.tokenizer, self.config.max_length)
        self.val_dataset = TextDataset(val_data, self.tokenizer, self.config.max_length) if val_data else None

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(self.val_dataset, batch_size=self.config.batch_size)
        return None

    def save_adapter(self, path: str):
        self.model.save_pretrained(path) 
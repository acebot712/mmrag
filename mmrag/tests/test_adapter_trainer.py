import pytest
from mmrag.trainers.adapter_trainer import AdapterTrainer

def test_adapter_trainer_smoke(tmp_path):
    config_path = "mmrag/configs/mmrag.yaml"
    train_data = [{"input": "Hello", "target": "Hi"}]
    val_data = [{"input": "Bye", "target": "Goodbye"}]
    trainer = AdapterTrainer(config_path, train_data, val_data)
    batch = next(iter(trainer.train_dataloader()))
    loss = trainer.training_step(batch, 0)
    assert loss is not None 
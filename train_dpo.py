import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from base_model.mygpt import MyGPT
from dpo_model.config import Config, gpt_config
from dpo_model.dpo import DPO
from utils.dpo_data_load import CustomDataset


class TrainDpo:
    def __init__(self):
        self.config = Config()
        # 演员和评论家模型
        self.model = MyGPT(
            local_rank=gpt_config["local_rank"],
            vocab=gpt_config["vocab"],
            embed_dim=gpt_config["embed_dim"],
            ff_embed_dim=gpt_config["ff_embed_dim"],
            num_heads=gpt_config["num_heads"],
            dropout=gpt_config["dropout"],
            layers=gpt_config["layers"],
        ).to(self.config.device)
        # 获得策略模型优化器, 这里使用的是lora, 不优化全量数据
        self.model_opt = Adam(self.model.parameters(), lr=self.config.lr)
        # 参考模型
        self.reference_model = MyGPT(
            local_rank=gpt_config["local_rank"],
            vocab=gpt_config["vocab"],
            embed_dim=gpt_config["embed_dim"],
            ff_embed_dim=gpt_config["ff_embed_dim"],
            num_heads=gpt_config["num_heads"],
            dropout=gpt_config["dropout"],
            layers=gpt_config["layers"],
        ).to(self.config.device)

        self.tokenizer = gpt_config["vocab"]

        # 训练数据
        dataset = CustomDataset(self.config.data_path, self.tokenizer)

        self.data_loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=dataset.collate_fn
        )
        self.dpo = DPO(self.model, self.model_opt, self.config)

    def train_dpo(self):
        for _epoch in range(self.config.epochs):
            for batch_data in self.data_loader:
                ref_logits = self.reference_model.generate_logits(
                    batch_data["inputs_ids"].to(self.config.device), batch_data["inputs_masks"].to(self.config.device)
                )  # 获得参考模型的logit
                self.dpo.train(
                    batch_data["inputs_ids"].to(self.config.device),
                    batch_data["inputs_masks"].to(self.config.device),
                    ref_logits,
                    batch_data["labels_mask"].to(self.config.device),
                )

        self.save_model()

    def save_model(self):
        # 保存lora参数
        torch.save(self.model.state_dict(), self.config.save_lora_path)


if __name__ == "__main__":
    train_dpo = TrainDpo()
    train_dpo.train_dpo()

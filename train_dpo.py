from torch.optim import Adam
from torch.utils.data import DataLoader

from base_model.tokenizer import Tokenizer
from dpo_model.config import Config
from dpo_model.dpo import DPO
from dpo_model.model import Model
from dpo_model.reference_model import ReferenceModel
from utils.dpo_data_load import CustomDataset


class TrainDpo:
    def __init__(self, tokenizer):
        self.config = Config()
        # 演员和评论家模型
        self.model = Model(self.config)
        if tokenizer is None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self.model.tokenizer
        # 获得策略模型优化器, 这里使用的是lora, 不优化全量数据
        self.model_opt = Adam(self.model.parameters(), lr=self.config.lr)
        # 参考模型
        self.reference_model = ReferenceModel(self.config)
        # 训练数据
        dataset = CustomDataset(self.config.data_path, self.tokenizer)
        self.data_loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=dataset.collate_fn
        )
        self.dpo = DPO(self.model, self.model_opt, self.config)

    def train_dpo(self):
        for _epoch in range(self.config.epochs):
            for batch_data in self.data_loader:
                ref_logits = self.reference_model(
                    batch_data["inputs_ids"], batch_data["inputs_masks"]
                )  # 获得参考模型的logit
                self.dpo.train(
                    batch_data["inputs_ids"], batch_data["inputs_masks"], ref_logits, batch_data["labels_mask"]
                )

        self.save_model()

    def save_model(self):
        # 保存lora参数
        self.model.model.save_pretrained(self.config.save_lora_path, safe_serialization=False)


if __name__ == "__main__":
    tokenizer = Tokenizer(filename=Config.tokenizer_path, min_occur_cnt=10)
    train_dpo = TrainDpo(tokenizer=tokenizer)
    train_dpo.train_dpo()

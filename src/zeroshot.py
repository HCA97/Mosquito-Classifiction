from typing import List
import open_clip
import torch as th


PROMPTS = [
    "mosquito species of Aedes albopictus",
    "mosquito species of Culex",
    "mosquito species of Aedes japonicus/koreicus",
    "mosquito species of Culiseta",
    "mosquito species of Anopheles",
    "mosquito species of Aedes aegypti",
]


class MosquitoZeroShotClassifier:
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        dataset: str = "datacomp_xl_s13b_b90k",
        prompts: List[str] = PROMPTS,
        device: str = "cuda",
    ):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=dataset, device=device
        )
        self.model.eval()
        self.prompts = prompts
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.text = self.tokenizer(self.prompts).to(device)

    def __call__(self, x: th.Tensor):
        return self.forward(x)

    def forward(self, x: th.Tensor) -> th.Tensor:
        image_features = self.model.encode_image(x)
        text_features = self.model.encode_text(self.text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs

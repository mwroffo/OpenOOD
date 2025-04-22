import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip





class CLIPZeroshot(nn.Module):
    def __init__(self, classnames, templates, backbone='ViT-B/16'):
        super().__init__()
        # Determine best available device (prefers CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        assert backbone in clip.available_models()
        self.model, self.preprocess = clip.load(backbone, device=self.device)
        self.zeroshot_weights = self.zeroshot_classifier(self.model, classnames,
                                                    templates)
    
    # https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    def zeroshot_classifier(self, model, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname)
                        for template in templates]  # format with class
                texts = clip.tokenize(texts).to(self.device)  # tokenize
                class_embeddings = model.encode_text(
                    texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights
    def forward(self, x):
        image_features = self.model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.zeroshot_weights
        return logits

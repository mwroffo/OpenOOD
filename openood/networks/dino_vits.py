import torch
import torch.nn as nn

class DINO_ViT_S_16(nn.Module):
    def __init__(self, num_classes=200, pretrained_path=None, return_feature=False):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.model.heads = nn.Identity()  # remove classifier
        self.fc = nn.Linear(384, num_classes)
        self.return_feature = return_feature

        if pretrained_path:
            print(f"Loading DINO ViT-S/16 from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = state_dict.get("teacher", state_dict)
            keys_to_remove = [k for k in state_dict if "head" in k]
            for k in keys_to_remove:
                del state_dict[k]
            msg = self.model.load_state_dict(state_dict, strict=False)
            print("Loaded DINO weights:", msg)

    def forward(self, x, return_feature=False):
        features = self.model(x)
        logits = self.fc(features)
        return (logits, features) if return_feature else logits
    
    def forward_threshold(self, x, threshold):
        features = self.model(x)  # Get ViT [CLS] token representation
        clipped_features = features.clip(max=threshold)  # Apply thresholding
        logits = self.fc(clipped_features)
        return logits
    
    def get_fc(self):
        return self.fc.weight.cpu().detach().numpy(), self.fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.heads[0]
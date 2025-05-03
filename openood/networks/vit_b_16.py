import torch
from torchvision.models.vision_transformer import VisionTransformer
import torch.nn.functional as F

class ViT_B_16(VisionTransformer):
    """
    The model from "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    by Dosovitskiy et al., 2020
    """
    def __init__(self,
                 mode,
                 image_size=384,
                 patch_size=16,
                 num_layers=12,
                 num_heads=12,
                 hidden_dim=768,
                 mlp_dim=3072,
                 num_classes=150,
                 pretrained_path='results/cub150_seed1_vit-b-16_base_e100_lr0.1_default/s0/best_epoch68_acc0.7664.ckpt', #'vit_b_16_hf.pth',
                 freeze_backbone=True):
        super(ViT_B_16, self).__init__(image_size=image_size,
                                       patch_size=patch_size,
                                       num_layers=num_layers,
                                       num_heads=num_heads,
                                       hidden_dim=hidden_dim,
                                       mlp_dim=mlp_dim,
                                       num_classes=num_classes)

        self.feature_size = hidden_dim

        # Load pretrained weights if provided
        if pretrained_path:
            print(f"Loading ViT weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")

            # Hugging Face / TorchVision-compatible .pth checkpoint
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            if mode == 'train':
                # Resize position embeddings if they don't match
                for posemb_key in ['pos_embed', 'pos_embedding', 'encoder.pos_embedding']:
                    if posemb_key in state_dict:
                        old_posemb = state_dict[posemb_key]
                        new_posemb = self.encoder.pos_embedding
                        if old_posemb.shape != new_posemb.shape:
                            print(f" Resizing pos_embedding from {old_posemb.shape} to {new_posemb.shape}")
                            cls_token = old_posemb[:, :1]
                            old_grid = old_posemb[:, 1:]
                            gs_old = int(old_grid.shape[1] ** 0.5)
                            gs_new = int((new_posemb.shape[1] - 1) ** 0.5)

                            # interpolate
                            old_grid = old_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
                            new_grid = F.interpolate(old_grid, size=(gs_new, gs_new), mode='bilinear', align_corners=False)
                            new_grid = new_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
                            state_dict[posemb_key] = torch.cat([cls_token, new_grid], dim=1)
                        
                #  Filter out classification head (1000 → 150 mismatch)
                keys_to_remove = [k for k in state_dict if "head" in k or "heads.head" in k]
                for k in keys_to_remove:
                    print(f"Removing incompatible key from checkpoint: {k}")
                    del state_dict[k]

                missing, unexpected = self.load_state_dict(state_dict, strict=False)

                print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
                if missing:
                    print("Missing keys:")
                    for k in missing:
                        print(f"  - {k}")
                if unexpected:
                    print("Unexpected keys:")
                    for k in unexpected:
                        print(f"  - {k}")

            # Freeze all layers except last block and head (for fine-tuning)
            if freeze_backbone:
                for name, param in self.named_parameters():
                    if "encoder.layers.11" in name or "heads" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # 🔍 Debug: print which params will be trained
            print("Trainable parameters:")
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(f"  - {name}")

        # Confirm classifier head config
        print("Classifier head:", self.heads)

    def forward(self, x, return_feature_list=False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if return_feature_list:
            # print("[DEBUG] CLS feat stats:", x.mean().item(), x.std().item())
            return self.heads(x), x
        else:
            return self.heads(x)

    def forward_threshold(self, x, threshold):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        feature = x.clip(max=threshold)
        logits_cls = self.heads(feature)

        return logits_cls

    def get_fc(self):
        fc = self.heads[0]
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.heads[0]

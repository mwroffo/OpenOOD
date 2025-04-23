import numpy as np
from .base_postprocessor import BasePostprocessor
from sklearn.covariance import EmpiricalCovariance
import torch
from tqdm import tqdm


class ViTMahalanobisPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.means = None
        self.inv_cov = None

    def setup(self, net, id_loader, ood_loader_dict):
        """Compute per-class mean and shared covariance from training features."""
        features_by_class = {}
        print("🔍 Extracting ViT features for Mahalanobis distance...")
        
        with torch.no_grad():
            for batch in tqdm(id_loader['train'],desc='Setup: ', position=0, leave=True):
                images, labels = batch['data'].cuda(), batch['label']

            with torch.no_grad():
                # ViT CLS token
                logits, feats = net(images, return_feature=True)
                feats = feats.cpu().numpy()

            for f, lbl in zip(feats, labels):
                features_by_class.setdefault(lbl, []).append(f)

        # Compute means and global covariance
        all_features = []
        all_labels = []

        self.means = {}
        for cls, feats in features_by_class.items():
            feats = np.stack(feats)
            self.means[cls] = feats.mean(axis=0)
            all_features.append(feats)
            all_labels.extend([cls] * feats.shape[0])

        all_features = np.vstack(all_features)
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(all_features)
        self.inv_cov = np.linalg.inv(cov_estimator.covariance_)

        print("Mahalanobis setup complete")

    def postprocess(self, net, data):
        """Batch Mahalanobis scoring."""
        with torch.no_grad():
            logits, feats = net(data, return_feature=True)  # feats: [B, D]
            preds = logits.argmax(dim=1)  # shape: [B]

            feats_np = feats.cpu().numpy()  # shape: [B, D]

        scores = []
        for feat in feats_np:
            distances = [
                (feat - mu).T @ self.inv_cov @ (feat - mu)
                for mu in self.means.values()
            ]
            scores.append(-min(distances))  # flip sign: higher = more ID

        scores_tensor = torch.tensor(scores)  # shape: [B]

        return preds.cpu(), scores_tensor
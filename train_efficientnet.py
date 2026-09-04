import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from dataset import DRDataset, get_train_transforms, get_eval_transforms
from train import train_one_stage, evaluate, get_class_weights, device
import model as M

CKPT = os.getcwd()

# reuse the already-built split
train_ds = DRDataset('labels/combined_train_split.csv', transform=get_train_transforms())
val_ds   = DRDataset('labels/combined_val_split.csv',   transform=get_eval_transforms())
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

class_weights = get_class_weights('labels/combined_train_split.csv').to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

net = M.build_model_efficientnet(num_classes=5, pretrained=True).to(device)

# Stage 1
net = M.freeze_backbone_efficientnet(net)
opt1 = torch.optim.Adam(net.classifier[1].parameters(), lr=1e-3, weight_decay=5e-4)
net, _ = train_one_stage(net, train_loader, val_loader, opt1, criterion,
                         num_epochs=5, stage_name='eff_stage1',
                         resume_path='checkpoint_eff_stage1.pt')

# Stage 2
net = M.unfreeze_backbone_efficientnet(net, 'features.5')
opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-5, weight_decay=5e-4)
net, best = train_one_stage(net, train_loader, val_loader, opt2, criterion,
                            num_epochs=60, stage_name='eff_stage2', patience=8,
                            resume_path='checkpoint_eff_stage2.pt')

torch.save(net.state_dict(), os.path.join(CKPT, 'best_efficientnet.pt'))
print(f"DONE — best EfficientNet val QWK: {best:.4f}")
print(f"Saved to {CKPT}/best_efficientnet.pt")

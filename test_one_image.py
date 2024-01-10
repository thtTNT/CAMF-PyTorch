from train_autoencoder import AutoEncoder
from train_classifier import Classifier
import torch.nn.functional as F
import torchvision
import torch
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_ir", required=True, help="input directory of images")
parser.add_argument("--input_vis", required=True, help="input directory of images")
# IR output image and VIS output image
parser.add_argument("--output", required=True, help="output directory of images")

args = parser.parse_args()

auto_encoder = AutoEncoder()
auto_encoder.load("model/")
auto_encoder.eval()
classifier = Classifier()
classifier.load("model_tno/model_42_1/")
classifier.eval()

image_ir = Image.open(args.input_ir)
image_vis = Image.open(args.input_vis)
image_ir = np.array(image_ir)
image_vis = np.array(image_vis)
image_ir = image_ir / 255.0 * 2 - 1
image_vis = image_vis / 255.0 * 2 - 1

image_ir = np.expand_dims(image_ir, axis=0)
image_vis = np.expand_dims(image_vis, axis=0)
image_ir = torch.from_numpy(image_ir).float()
image_vis = torch.from_numpy(image_vis).float()


features_ir = auto_encoder.encoder(image_ir)
features_ir = features_ir.detach()
features_vis = auto_encoder.encoder(image_vis)
features_vis = features_vis.detach()

weight_ir = 0.5 + classifier.linear.weight[0].tanh() * 0.5
weight_vis = 0.5 + classifier.linear.weight[1].tanh() * 0.5
print(weight_ir / torch.mean(weight_ir, dim=-1))
weight_ir = (weight_ir / torch.mean(weight_ir, dim=-1)).view(128,1,1)
weight_vis = (weight_vis / torch.mean(weight_vis, dim=-1)).view(128,1,1)

output_ir = features_ir * weight_ir
output_vis = features_vis * weight_vis

fused_features = output_ir + output_vis
fused_features = fused_features.unsqueeze(0)

fused_image = auto_encoder.decoder(fused_features)
fused_image = fused_image.detach()
torchvision.utils.save_image((fused_image + 1) / 2, args.output)
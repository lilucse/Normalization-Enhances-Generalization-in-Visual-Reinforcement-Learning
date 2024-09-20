import torch
import drqv2_cnsn
agent = torch.load('./dmcontrol-generalization-benchmark/drqv2_cn5sn_model/cartpole_swingup/snapshot.pt', map_location='cuda:0')# TODO
print('finish')
import torch
from torchvision.models.resnet import resnet101
from archs import LW_DualNet

from thop import profile
import torch
import torchvision.models as models

# ----------------
import time

start_time = time.time()

# 在这里执行训练代码

end_time = time.time()
training_time = end_time - start_time

print("训练时间为: ", training_time, "秒")

# ------------------
iterations = 300   # 重复计算的轮次

model = LW_DualNet()
device = torch.device("cuda:0")
model.to(device)

random_input = torch.randn(3, 3, 256, 256).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))

flops, params = profile(model.to(device), inputs=(random_input,))

print("FLOPs：", flops)
print("参数量：", params)

print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


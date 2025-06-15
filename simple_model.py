import torch
import torch.nn as nn

x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,4,6,8],dtype=torch.float32)

w=torch.tensor([0.0],dtype=torch.float32,requires_grad=True)

def forward(x):
    return x*w

loss=nn.MSELoss()
optimizer=torch.optim.SGD([w],lr=0.02)

print(f"before backpropagation, pred for 5 : {forward(5).item():.3f}")

for epoch in range(35):
    y_hat=forward(x)
    l=loss(y_hat,y)
    l.backward()
    optimizer.step()

    print(f"epoch{epoch+1},loss:{l:.8f},w:{w.item():.2f},grad:{w.grad.item():.2f}")
    optimizer.zero_grad()

print(f"After backpropagation, pred for 5 : {forward(5).item():.3f}")
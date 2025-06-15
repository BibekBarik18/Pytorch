import torch

x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,4,6,8],dtype=torch.float32)

w=torch.tensor([0.0],dtype=torch.float32,requires_grad=True)

def forward(x):
    return x*w

def loss(y_hat,y):
    return ((y_hat-y)**2).mean()

print(f"before backpropagation, pred for 5 : {forward(5).item():.3f}")

lr=0.02
for epoch in range(35):
    y_hat=forward(x)
    l=loss(y_hat,y)
    l.backward()
    with torch.no_grad():
        w-=lr*w.grad

    print(f"epoch{epoch+1},loss:{l:.8f},w:{w.item():.2f},grad:{w.grad.item():.2f}")
    w.grad.zero_()

print(f"After backpropagation, pred for 5 : {forward(5).item():.3f}")
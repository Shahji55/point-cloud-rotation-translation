import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create synthetic training data
def make_data(samples=100, points=50):
    src_list = []   # source point clouds (X)
    tgt_list = []   # target point clouds (Y)
    cos_list = []   # ground truth cos(angle)
    sin_list = []   # ground truth sin(angle)
    t_list   = []   # ground truth translation

    for _ in range(samples):
        # Random source point cloud
        X = np.random.rand(points, 2)

        # Random rotation
        angle = np.random.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])

        # Random translation
        t = np.random.uniform(-0.5, 0.5, size=2)

        # Apply transform to get output
        Y = X @ R.T + t

        src_list.append(X)
        tgt_list.append(Y)
        cos_list.append(np.cos(angle))
        sin_list.append(np.sin(angle))
        t_list.append(t)

    return np.array(src_list), np.array(tgt_list), np.array(cos_list), np.array(sin_list), np.array(t_list)

X_s, Y_s, cos_gt, sin_gt, t_gt = make_data()
N = X_s.shape[1]
batch_size = 16

# Neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(N*4,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,4)  # cos(angle), sin(angle), tx, ty
        )
    def forward(self, x):
        return self.fc(x)

model = Net()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# Network training
for epoch in range(50):
    for start in range(0, len(X_s), batch_size):

        # Select batch range
        end = start + batch_size

        # Input to the model
        inp = np.hstack([X_s[start:end], Y_s[start:end]])
        inp = inp.reshape(len(inp), -1)  # flatten into rows
        inp = torch.tensor(inp, dtype=torch.float32)

        # cos, sin, and translation ground truth
        tgt = np.hstack([cos_gt[start:end, None],
                         sin_gt[start:end, None],
                         t_gt[start:end]])
        tgt = torch.tensor(tgt, dtype=torch.float32)

        # Prediction
        pred = model(inp)

        # Mean squared error loss
        loss = ((pred - tgt) ** 2).mean()

        # Update model weights
        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.5f}")


# Calculate baseline + NN pred and visualize
X = X_s[0]
Y = Y_s[0]

# Centroids
cX = X.mean(axis=0)
cY = Y.mean(axis=0)

# Baseline angle
src = X[0] - cX
tgt = Y[0] - cY

ang_src = np.arctan2(src[1], src[0])
ang_tgt = np.arctan2(tgt[1], tgt[0])

base_angle = ang_tgt - ang_src

base_R = np.array([
    [np.cos(base_angle), -np.sin(base_angle)],
    [np.sin(base_angle),  np.cos(base_angle)]
])

# Baseline translation
base_t = cY - cX @ base_R.T

test_input = np.concatenate([X.flatten(), Y.flatten()]).astype(np.float32)
test_input = torch.tensor(test_input)[None, :]

out = model(test_input).detach().numpy()[0]

cos_p, sin_p = out[0], out[1]
t_p = out[2:]  # predicted translation 

print("\nBaseline translation: ", base_t)
print("Predicted translation: ", t_p)

angle_p = np.arctan2(sin_p, cos_p)  # predicted angle

print("\nBaseline angle: ", np.degrees(base_angle))
print("Predicted angle: ", np.degrees(angle_p))

R_pred = np.array([[np.cos(angle_p), -np.sin(angle_p)],
                   [np.sin(angle_p),  np.cos(angle_p)]])
Y_pred = X @ R_pred.T + t_p

plt.scatter(X[:,0], X[:,1], label="Source")
plt.scatter(Y[:,0], Y[:,1], label="Target")
plt.scatter(Y_pred[:,0], Y_pred[:,1], label="NN Pred")
plt.legend(); plt.axis('equal'); plt.title("Ground truth vs NN Pred")
plt.show()

import numpy as np
import matplotlib.pyplot as plt


x = np.array([[-2], [-1], [1], [2]], dtype=float)
y = np.array([[4], [2], [-2], [-4]], dtype=float)

lr = 0.01
epochs = 1000
losses = []
np.random.seed(0)

w1 = np.random.randn()
b1 = np.random.randn()
w2 = np.random.randn()
b2 = np.random.randn()

for i in range(epochs):
    # forward pass
    hidden = w1 * x + b1
    y_pred = w2 * hidden + b2

    # calculate loss 
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)

    # backprop 
    d_loss = 2 * (y_pred - y) / len(y)

    dw2 = np.sum(d_loss * hidden)
    db2 = np.sum(d_loss)

    dw1 = np.sum(d_loss * w2 * x)
    db1 = np.sum(d_loss * w2)

    # update weights
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

# final predictions
hidden = w1 * x + b1
y_pred = w2 * hidden + b2

# plot results
plt.figure(figsize=(6,6))
plt.scatter(x, y, label="data")
plt.plot(x, y_pred, label="prediction")
plt.title("Fit on Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
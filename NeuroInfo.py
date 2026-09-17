import random
import numpy as np
import matplotlib.pyplot as plt
from dataset import dataset


def relu(t):
    return np.maximum(t, 0)


def relu_deriv(t):
    return (t >= 0).astype(float)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)


def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))


def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full


def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)

    return z


def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc


input_layer = 10
output_layer = 4
first_layer = 100

W1 = np.random.rand(input_layer, first_layer)
b1 = np.random.rand(1, first_layer)
W2 = np.random.rand(first_layer, output_layer)
b2 = np.random.rand(1, output_layer)

W1 = (W1 - 0.5) * 2 * np.sqrt(1/input_layer)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/input_layer)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/first_layer)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/first_layer)

ALPHA = 0.00001
NUM_EPOCHS = 100000
BATCH_SIZE = 50

loss_arr = []

for ep in range(NUM_EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset) // BATCH_SIZE):

        batch_x, batch_y = zip(*dataset[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE])
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        # Forward

        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)

        E = np.sum(sparse_cross_entropy_batch(z, y))
        loss_arr.append(E)

        # Backward

        y_full = to_full_batch(y, output_layer)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # Update

        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

accuracy = calc_accuracy()
print("Точность:", accuracy)

plt.plot(loss_arr)
plt.show()

while True:
    print("-" * 200)
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = map(float, input("Введите характеристики: ").split())
    x = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10])
    probs = predict(x)
    pred_class = np.argmax(probs)
    class_names = ['Ambiance', 'Cara Cara', 'Hamlin', 'Blood Orange']
    print('Полученный класс:', class_names[pred_class])

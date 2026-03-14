from src.nn import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])

for i in range(100):

    # Forward pass
    ypred = [n(x) for x in xs]

    # Compute mean squared error
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

    # Print total parameters and loss
    total_params = len(n.parameters())
    print(f"Iteration {i+1:03d} | Loss: {loss.data:.4f} | Total Parameters: {total_params}")

    # Reset gradients
    for p in n.parameters():
        p.grad = 0.0

    # Backward pass
    loss.backward()

    # Update parameters
    for p in n.parameters():
        p.data += -0.01 * p.grad
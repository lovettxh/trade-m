import torch
import math


dtype = torch.float
device = torch.device("cpu")


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 3, 5, 2

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    # torch.nn.Linear(D_in,D_out)
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Tanh()
)
model.to(device)
# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(100):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()
    #print(loss)

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            
           

outputs = model(torch.tensor([1.,2.5,3.5], requires_grad=True))[1]
grad = torch.autograd.grad(outputs, model.parameters(), create_graph=True, only_inputs=True)
print("Grad:", grad)
eta =  [torch.rand_like(t, requires_grad=True) for t in model.parameters()]
print("eta:",eta)
mult = torch.sum(torch.stack([torch.dot(torch.flatten(x),torch.flatten(y)) for x,y in zip(grad,eta)]))

grad1 = torch.autograd.grad(mult, model.parameters(), create_graph=True, only_inputs=True)

print("Second Grad:", grad1)
s = sum([torch.sum(x) for x in grad1])
print(s)
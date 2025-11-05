# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Hari Prasath P

### Register Number: 212223230070

```python
class Model(nn.Module):
      def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features,out_features)

    def forward(self, x):
        return self.linear(x)
# Initialize the Model, Loss Function, and Optimizer

torch.manual_seed(59)  # Ensure same initial weights
model = Model(1, 1)
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("\nName: Hari Prasath P")
print("Register No: 212223230070")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n')
# Define Loss Function & Optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
# Train the Model
epochs = 50
losses = []
for epoch in range(1, epochs + 1):  # Loop over epochs
    y_pred = model(X)
    loss = loss_function(y_pred,y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')


```

### Dataset Information
<img width="1271" height="193" alt="Screenshot 2025-11-05 102321" src="https://github.com/user-attachments/assets/6155352d-eb3a-49f6-80c7-fd33d5e2e393" />
<img width="1272" height="176" alt="Screenshot 2025-11-05 102354" src="https://github.com/user-attachments/assets/72641cd5-d25f-443e-a92d-480eba71e1b7" />
<img width="1264" height="581" alt="Screenshot 2025-11-05 102418" src="https://github.com/user-attachments/assets/dcf886d2-8d53-46b1-9ac5-6484f04aff47" />

### OUTPUT
#### Training Loss Vs Iteration Plot
<img width="1225" height="574" alt="Screenshot 2025-11-05 102849" src="https://github.com/user-attachments/assets/3170d449-8592-462e-9e66-52bc599590be" />

#### Best Fit line plot
<img width="1212" height="581" alt="Screenshot 2025-11-05 102904" src="https://github.com/user-attachments/assets/07f4d268-a16f-4b01-84a6-fa1b8e3acc17" />

### New Sample Data Prediction
<img width="1253" height="293" alt="Screenshot 2025-11-05 102921" src="https://github.com/user-attachments/assets/bfa107da-c25a-4443-b91d-6ec1ecdead53" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.

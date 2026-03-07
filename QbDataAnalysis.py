import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- STEP 1: DATA CURATION & CLEANING ---
# Loading the dataset from the specific filename provided
filename = 'StandardQB - QB Scouting Data Sheet Creation.csv'
# --- STEP 1: DATA CURATION & CLEANING ---
qb_dataframe = pd.read_csv(filename)

# 1. Clean numeric columns
cols_to_fix = ['HS Stars (247 comp)', 'Draft position', 'Rating', 'Wonderlic/S2 equivalent']
for col_name in cols_to_fix:
    qb_dataframe[col_name] = pd.to_numeric(qb_dataframe[col_name], errors='coerce')

# 2. FEATURE CLIPPING (The "Brady Fix"): Rushing yards below 0 become 0
qb_dataframe['Rush Yards per game'] = qb_dataframe['Rush Yards per game'].clip(lower=0)
qb_dataframe['Rush TDs per game'] = qb_dataframe['Rush TDs per game'].clip(lower=0)

# 3. MISSING DATA FLAGS: Handle players like Drake Maye with no Wonderlic
# Create a 'Mute Button' flag: 1 if missing, 0 if present
qb_dataframe['Wonderlic_Is_Missing'] = qb_dataframe['Wonderlic/S2 equivalent'].isna().astype(int)

# Fill the missing gaps with the Mean so the math stays stable
E_Mean_Wonderlic = qb_dataframe['Wonderlic/S2 equivalent'].mean()
qb_dataframe['Wonderlic/S2 equivalent'] = qb_dataframe['Wonderlic/S2 equivalent'].fillna(E_Mean_Wonderlic)

# 4. Fill all other missing values (like HS Stars or Draft Position) with 0
qb_dataframe = qb_dataframe.fillna(0)

# 5. Feature/Target Separation
# Drop 'Name' and the non-numeric 'Test taken or confidential' column
features_matrix = qb_dataframe.drop(columns=['Name', 'Rating', 'Wonderlic/S2 Test taken or confidential'])
target_vector = qb_dataframe['Rating']

# 6. Use StandardScaler (Z-Scores) instead of MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
E_scaled_features = scaler.fit_transform(features_matrix)
E_target_array = target_vector.values.reshape(-1, 1)

# --- REPLACED BLOCK ---
# Create an array of row numbers so we can track names through the shuffle
indices = np.arange(len(E_scaled_features))

# Pass 'indices' into the split to generate idx_train and idx_test
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    E_scaled_features, E_target_array, indices, test_size=0.2, random_state=42
)

# Pull the names for the test set using the indices we just generated
test_names = qb_dataframe.iloc[idx_test]['Name'].values
# --- END REPLACED BLOCK ---

# Convert the data into PyTorch Tensors for processing on the Neural Network
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# --- STEP 2: DEEP LEARNING MODEL DEFINITION ---
class QB_Success_Predictor(nn.Module):
    def __init__(self, input_dim):
        super(QB_Success_Predictor, self).__init__()
        # Wider first layer (32 neurons) to catch various QB prototypes
        self.E_Layer_1 = nn.Linear(input_dim, 32)
        self.E_Dropout_1 = nn.Dropout(0.4)
        # Refinement layer (16 neurons)
        self.E_Layer_2 = nn.Linear(32, 16)
        self.E_Dropout_2 = nn.Dropout(0.2)
        # Final output for the 0-1 Rating
        self.E_Output_Layer = nn.Linear(16, 1)
        self.E_ReLU = nn.ReLU()
        self.E_Sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.training:
            # Add Gaussian Noise to inputs to stop the model from memorizing specific players
            x = x + torch.randn_like(x) * 0.02

        x = self.E_ReLU(self.E_Layer_1(x))
        x = self.E_Dropout_1(x)
        x = self.E_ReLU(self.E_Layer_2(x))
        x = self.E_Dropout_2(x)
        return self.E_Sigmoid(self.E_Output_Layer(x))


# Initialize the model using the number of columns in the feature set
model = QB_Success_Predictor(X_train_tensor.shape[1])

# --- STEP 3: ADVANCED TRAINING CONFIGURATION ---
# Mean Squared Error evaluates how far the Prediction is from the Actual Rating
E_Loss_Criterion = nn.MSELoss()

# The Optimizer now includes Weight Decay (Regularization) to prevent Overfitting
E_Initial_Learning_Rate = 0.01
E_Weight_Decay_Penalty = 1e-3

optimizer = optim.Adam(
    model.parameters(),
    lr=E_Initial_Learning_Rate,
    weight_decay=E_Weight_Decay_Penalty
)

# The Scheduler reduces the Learning Rate when the Loss improvement stalls
# Patience=15 means if the Loss doesn't drop for 15 games, we cut the speed by half (0.5)
E_Learning_Rate_Scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=15,
    factor=0.5
)

# --- STEP 4: THE OPTIMIZED TRAINING LOOP ---
epochs = 200  # Increased to allow the Scheduler time to work
loss_history = []

print("Starting Training...")
for epoch in range(epochs):
    # model.train() ensures Dropout and Gaussian Noise are ACTIVE during practice
    model.train()

    # Reset gradients to zero to prevent data from the previous Epoch from leaking
    optimizer.zero_grad()

    # Forward Pass: Generate predictions using the Neural Network
    E_Predictions = model(X_train_tensor)

    # Calculate Loss: E[Current Loss] = MSE(E_Predictions, y_train_tensor)
    loss = E_Loss_Criterion(E_Predictions, y_train_tensor)

    # Backward Pass: Calculate the Gradient (direction of error)
    loss.backward()

    # Update Weights: Adjust neurons using the Adam Optimization Rule
    optimizer.step()

    # Update the Scheduler: Inform the Scheduler of the current Loss value
    E_Learning_Rate_Scheduler.step(loss.item())

    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# --- STEP 5: EVALUATION & RESULTS ---
model.eval()
with torch.no_grad():
    final_predictions = model(X_test_tensor)
    test_loss = E_Loss_Criterion(final_predictions, y_test_tensor)
    print(f"\nFinal Test MSE Loss: {test_loss.item():.4f}")

# Visualize the training progress for the PDF report
plt.plot(loss_history)
plt.title('Neural Network Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.show()

# --- REPLACED BLOCK ---
print("\n" + "=" * 70)
print(f"{'PLAYER NAME':<35} | {'PRED':<8} | {'ACTUAL':<8} | {'DIFF'}")
print("-" * 70)

# Loop through the entire test set
for i in range(len(final_predictions)):
    name_label = test_names[i]
    prediction_value = final_predictions[i].item()
    actual_value = y_test_tensor[i].item()
    diff = abs(prediction_value - actual_value)

    # Flag differences larger than 0.25
    flag = " [!] MISSED" if diff > 0.25 else ""

    print(f"{name_label:<35} | {prediction_value:<8.2f} | {actual_value:<8.2f} | {diff:.2f}{flag}")
print("=" * 70)


# --- END REPLACED BLOCK ---


# --- STEP 6: CUSTOM SCOUTING TOOL (INFERENCE) ---
def predict_new_qb(custom_stats_dict):
    """
    Takes a dictionary of raw stats, normalizes them,
    and returns the model's predicted Rating.
    """
    # 1. Convert the dictionary to a DataFrame to match the model's feature structure
    custom_df = pd.DataFrame([custom_stats_dict])

    # 2. Normalize the input using the SAME scaler from the training data
    # Formula Rule: E[Normalized Input] = (E[Raw Input] - E[Min]) / E[Range]
    custom_scaled = scaler.transform(custom_df)

    # 3. Convert to PyTorch Tensor
    custom_tensor = torch.tensor(custom_scaled, dtype=torch.float32)

    # 4. Run the forward pass (Inference mode)
    model.eval()
    with torch.no_grad():
        prediction = model(custom_tensor)

    return prediction.item()


# Example: Inputting a "Top Prospect" with elite stats
# Note: Ensure these keys match your CSV column names exactly!
new_prospect_stats = {
    'Height (in)': 77,
    'Weight (lbs)': 235.0,
    'Years Starter (college)': 3,
    'Draft position': 1,
    'HS Stars (247 comp)': .9999,
    'School Prestige at the time': 10,
    'Support Cast (College)': 0.5,
    'Pass Yards as starter per game': 357.3,
    'Pass TDs per game': 3.6,
    'Attempts per game': 27,
    'Cmp%': 81,
    'INTs per game': .06,
    'Rush Yards per game': 95.9,
    'Rush TDs per game': 2.1,
    '40-Yard': 4.31,
    'Vert (in)': 39,
    'Hand Size': 10.5,
    'Wonderlic/S2 equivalent': 35, # Example score
    'Wonderlic_Is_Missing': 0       # Set to 1 if you don't have a score
}

rating = predict_new_qb(new_prospect_stats)
print(f"\n--- NEW PROSPECT REPORT ---")
print(f"Predicted NFL Success Rating: {rating:.4f}")
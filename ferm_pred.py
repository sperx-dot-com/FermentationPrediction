# Basis Imports
import numpy as np
import pandas as pd

# Visualisierung
import matplotlib.pyplot as plt

# Scikit-learn Grundlagen
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Modelle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# XGBoost (muss installiert sein: pip install xgboost)
from xgboost import XGBRegressor

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


np.random.seed(42)  # Reproduzierbarkeit

n_samples = 800

# Numerische Features simulieren
temp_induction = np.random.normal(loc=30.0, scale=2.0, size=n_samples)      # Grad Celsius
ph = np.random.normal(loc=7.0, scale=0.2, size=n_samples)
do = np.random.normal(loc=40.0, scale=10.0, size=n_samples)                 # Prozent
feed_rate = np.random.normal(loc=5.0, scale=1.5, size=n_samples)            # g/L/h
induction_time = np.random.normal(loc=10.0, scale=3.0, size=n_samples)      # Stunden
od_induction = np.random.normal(loc=20.0, scale=3.0, size=n_samples)        # OD600

# Kategorische Features
strain_types = np.random.choice(["strain_A", "strain_B", "strain_C"], size=n_samples)
tag_types = np.random.choice(["CASPON", "His", "None"], size=n_samples)

# Jetzt definieren wir eine "wahre" Beziehung zum Titer
# einfaches Modell mit etwas Biologie Logik und Rauschen
noise = np.random.normal(loc=0.0, scale=0.5, size=n_samples)

# Wir nehmen an:
# - CASPON erhöht die Löslichkeit => mehr Titer
# - zu hohe Temperatur schadet
# - pH zu weit weg von 7 schadet
# - zu wenig DO schadet
# - vernünftige Feed Rate und OD helfen

tag_effect = np.where(tag_types == "CASPON", 1.0,
                      np.where(tag_types == "His", 0.3, 0.0))

strain_effect = np.where(strain_types == "strain_A", 0.5,
                         np.where(strain_types == "strain_B", 0.2, -0.2))

# Basis Titer
base_titer = 2.0

# Ein etwas komplexerer, aber kontrollierter Zusammenhang
titer = (
    base_titer
    + tag_effect
    + strain_effect
    + 0.1 * (od_induction - 18)      # zu niedrige OD schadet
    - 0.05 * np.abs(temp_induction - 30)  # weg von 30 Grad ist schlecht
    - 0.08 * np.abs(ph - 7.0)
    + 0.05 * (feed_rate - 5.0)
    + 0.02 * (do - 40) / 10
    + noise
)

data = pd.DataFrame({
    "temp_induction": temp_induction,
    "ph": ph,
    "do": do,
    "feed_rate": feed_rate,
    "induction_time": induction_time,
    "od_induction": od_induction,
    "strain": strain_types,
    "tag": tag_types,
    "titer": titer
})

data.head()

# Schneller Überblick
print(data.describe())

# Verteilung der Zielvariable
plt.hist(data["titer"], bins=30)
plt.xlabel("Titer [g/L]")
plt.ylabel("Häufigkeit")
plt.title("Verteilung des simulierten Titers")
plt.show()

# Mittelwerte nach Tag Typ
print("Mean titer per tag")
print(data.groupby("tag")["titer"].mean())

X = data.drop(columns=["titer"])
y = data["titer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.shape, X_test.shape

numeric_features = ["temp_induction", "ph", "do", "feed_rate",
                    "induction_time", "od_induction"]
categorical_features = ["strain", "tag"]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

baseline_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

baseline_model.fit(X_train, y_train)

y_pred_base = baseline_model.predict(X_test)

mse_base = mean_squared_error(y_test, y_pred_base)
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

print("Baseline Linear Regression:")
print("MSE:", mse_base)
print("MAE:", mae_base)
print("R2:", r2_base)

rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest:")
print("MSE:", mse_rf)
print("MAE:", mae_rf)
print("R2:", r2_rf)


xgb_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ))
])

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost:")
print("MSE:", mse_xgb)
print("MAE:", mae_xgb)
print("R2:", r2_xgb)


# Preprocessing Transform getrennt anwenden
# Preprocessing: fit auf Trainingsdaten, transform auf Train/Test
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# In numpy/csr -> dense array umwandeln (falls sparse, z. B. durch OneHot)
X_train_array = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
X_test_array = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

X_train_array.shape, X_test_array.shape

# In PyTorch Tensors umwandeln
X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Dataset und DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# In echten Bioprocess-Projekten sind die Datensätze oft relativ klein.
# Ein einfacher Feed-Forward-Netzwerk-Ansatz mit Minibatches ist hier mehr "Demonstration"
# als zwingende Notwendigkeit. In der Praxis würden RF/XGBoost oft ausreichen.

input_dim = X_train_tensor.shape[1]

class BioprocessNet(nn.Module):
    def __init__(self, input_dim):
        super(BioprocessNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Regression auf Titer
        )
        
    def forward(self, x):
        return self.net(x)

model = BioprocessNet(input_dim)

criterion = nn.MSELoss()  # MSE als Standard für Regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


n_epochs = 50

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * X_batch.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Train MSE: {epoch_loss:.4f}")

# Training Loop:
# Für kleine Datensätze reichen oft wenige Epochen aus.
# In realen Projekten würde man hier Early Stopping und eine Validierungstrennung verwenden.

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    
y_pred_np = y_pred_tensor.numpy().flatten()

mse_torch = mean_squared_error(y_test, y_pred_np)
mae_torch = mean_absolute_error(y_test, y_pred_np)
r2_torch = r2_score(y_test, y_pred_np)

print("PyTorch Feed-Forward Network:")
print("MSE:", mse_torch)
print("MAE:", mae_torch)
print("R2:", r2_torch)


# Random Forest Modell ohne Pipeline neu fitten, um an die Feature Namen zu kommen
# Wir nutzen direkt die transformierten Features

# Fit Preprocessor separat
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_proc, y_train)

# Feature Namen aus ColumnTransformer holen
# numerische
num_features = numeric_features
# kategorische, One Hot Encoder
cat_encoder = preprocessor.named_transformers_["cat"]["onehot"]
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)

all_feature_names = list(num_features) + list(cat_feature_names)

importances = rf.feature_importances_

feat_imp = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print(feat_imp.head(15))

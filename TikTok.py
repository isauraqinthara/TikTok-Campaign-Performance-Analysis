# ----------------------------------------
# STEP 0: IMPORT LIBRARIES
# ----------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create folder for plots
os.makedirs('visualizations', exist_ok=True)

# ----------------------------------------
# STEP 1: LOAD DATA
# ----------------------------------------
file_path = r"C:\Qin\TikTok\Take-home Test - Data Intern.xlsx"
df_raw = pd.read_excel(file_path, sheet_name='Raw')

# ----------------------------------------
# STEP 2: DATA EXPLORATION & AUDIT
# ----------------------------------------
print("Preview Data:")
print(df_raw.head(), end='\n\n')

print("Data Info:")
print(df_raw.info(), end='\n\n')

print("Missing Values:")
print(df_raw.isnull().sum(), end='\n\n')

# ----------------------------------------
# STEP 3: DATA CLEANING & PREPARATION
# ----------------------------------------
df = df_raw.copy()

# Convert date column
df['p_date'] = pd.to_datetime(df['p_date'], errors='coerce')
if df['p_date'].isnull().any():
    print("Warning: Beberapa nilai tanggal tidak valid dan telah dikonversi menjadi NaT.")

# Clean monetary columns
money_cols = ['Cost (USD)', 'GMV']
for col in money_cols:
    df[col] = df[col].replace({'\$': '', ',': '', '\.': ''}, regex=True).astype(float)

# Clean numeric columns
num_cols = ['Clicks', 'Impressions', 'Sales']
for col in num_cols:
    df[col] = df[col].replace({',': '', '\.': ''}, regex=True).astype(int)

# ----------------------------------------
# STEP 4: FEATURE ENGINEERING
# ----------------------------------------
df['CTR'] = df['Clicks'] / df['Impressions']
df['CPS'] = df['Cost (USD)'] / df['Sales']
df['ROAS'] = df['GMV'] / df['Cost (USD)']
df['Conversion Rate'] = df['Sales'] / df['Clicks']

# ----------------------------------------
# STEP 5: EDA & VISUALIZATION
# ----------------------------------------

def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(f'visualizations/{filename}')
    plt.close(fig)

# CTR per client
fig = plt.figure()
sns.barplot(data=df, x='Client Name', y='CTR')
plt.title("CTR per Client")
save_plot(fig, 'CTR_per_Client.png')

# ROAS per client
fig = plt.figure()
sns.barplot(data=df, x='Client Name', y='ROAS')
plt.title("ROAS per Client")
save_plot(fig, 'ROAS_per_Client.png')

# CPS per client
fig = plt.figure()
sns.barplot(data=df, x='Client Name', y='CPS')
plt.title("Cost per Sale (CPS) per Client")
save_plot(fig, 'CPS_per_Client.png')

# Distribusi Conversion Rate
fig = plt.figure()
sns.histplot(df['Conversion Rate'], bins=10, kde=True)
plt.title("Distribusi Conversion Rate")
save_plot(fig, 'ConversionRate_Distribution.png')

print("Semua visualisasi telah disimpan di folder 'visualizations'")

# ----------------------------------------
# STEP 6: INSIGHT & BUSINESS RECOMMENDATIONS
# ----------------------------------------

# Insight

print("\nInsight Summary:\n")

# Insight 1: CTR tertinggi
ctr_by_client = df.groupby("Client Name")["CTR"].mean()
highest_ctr = ctr_by_client.idxmax()
print(f"1. Client dengan CTR tertinggi: {highest_ctr} ({ctr_by_client.max():.2%})")

# Insight 2: ROAS tertinggi
roas_by_client = df.groupby("Client Name")["ROAS"].mean()
highest_roas = roas_by_client.idxmax()
print(f"2. Client dengan ROAS tertinggi: {highest_roas} ({roas_by_client.max():.2f})")

# Insight 3: CPS terendah
cps_by_client = df.groupby("Client Name")["CPS"].mean()
lowest_cps = cps_by_client.idxmin()
print(f"3. Client dengan CPS terendah: {lowest_cps} ({cps_by_client.min():.2f} USD)")

# Insight 4: Conversion Rate
cr_by_client = df.groupby("Client Name")["Conversion Rate"].mean()
print("4. Conversion Rate per Client:")
print((cr_by_client * 100).round(2).astype(str) + '%')

# Insight 5: Korelasi antar metrik
print("\n5. Korelasi antar metrik:")
metrics = ['CTR', 'CPS', 'ROAS', 'Conversion Rate']
correlation_matrix = df[metrics].corr()
print(correlation_matrix.round(2))

# Heatmap korelasi
fig = plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasi antar KPI")
save_plot(fig, 'Correlation_Heatmap.png')

# Business Recommendations

print("\nBusiness Recommendations:\n")

# Rekomendasi 1: Klien dengan ROAS tertinggi
roas_threshold = 1.0
print(f"1. Klien dengan ROAS tertinggi layak diprioritaskan: {highest_roas} ({roas_by_client[highest_roas]:.2f})")

ctr_median = df["CTR"].median()
roas_median = df["ROAS"].median()

# Rekomendasi 2: Klien dengan CTR tinggi namun ROAS rendah
clients_low_roas = df[
    (df["CTR"] > ctr_median) & (df["ROAS"] < roas_median)
]["Client Name"].unique()

if len(clients_low_roas) > 0:
    print("2. Klien dengan CTR tinggi tapi ROAS rendah (butuh investigasi lebih lanjut):")
    for client in clients_low_roas:
        print(f" - {client}")
else:
    # Else ini sekarang terkait dengan IF di atasnya
    print("2. Tidak ada klien dengan CTR tinggi namun ROAS rendah.")

# Rekomendasi 3: Klien dengan CPS rendah dan Conversion Rate tinggi (efisiensi)
summary = df.groupby("Client Name")[["CPS", "Conversion Rate"]].mean()
efficient_clients = summary[
    (summary["CPS"] < summary["CPS"].median()) &
    (summary["Conversion Rate"] > summary["Conversion Rate"].median())
].index.tolist()

if efficient_clients:
    print("3. Klien dengan CPS rendah & Conversion Rate tinggi (berpotensi efisien):")
    for client in efficient_clients:
        print(f" - {client}")
else:
    print("3. Tidak ada klien yang sangat efisien dalam kombinasi CPS & Conversion Rate.")

# Rekomendasi 4: Klien dengan ROAS < 1.0 (butuh perhatian khusus)
roas_below_one = roas_by_client[roas_by_client < roas_threshold].index.tolist()

if roas_below_one:
    print("4. Klien dengan ROAS < 1.0 (perlu optimasi kampanye):")
    for client in roas_below_one:
        print(f" - {client}")
    print(" → Rekomendasi: lakukan A/B testing pada materi iklan & targeting.")
else:
    print("4. Tidak ada klien dengan ROAS di bawah 1.0")

# ----------------------------------------
# STEP 7: PREDICTIVE MODELING - SALES PREDICTION
# ----------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Fitur dan target
X = df[['Cost (USD)', 'Impressions']]
y = df['Sales']

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nPredictive Modeling - Sales Prediction\n")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualisasi hasil prediksi vs aktual
fig = plt.figure()
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Predicted vs Actual Sales (Linear Regression)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
save_plot(fig, 'Predicted_vs_Actual_Sales.png')
print("Visualisasi 'Predicted_vs_Actual_Sales.png' telah disimpan.")

# Koefisien model
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nModel Coefficients:")
print(coef_df)

# ----------------------------------------
# STEP 8: CLIENT SEGMENTATION - CLUSTERING BASED ON PERFORMANCE METRICS
# ----------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select features for clustering
cluster_features = df.groupby("Client Name")[["CTR", "CPS", "ROAS", "Conversion Rate"]].mean()

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(scaled_features)

# Add cluster labels to dataframe
cluster_features['Cluster'] = cluster_labels

# Reset index for plotting
cluster_plot_df = cluster_features.reset_index()

# Visualize clustering result
fig = plt.figure()
sns.scatterplot(
    data=cluster_plot_df,
    x="ROAS",
    y="Conversion Rate",
    hue="Cluster",
    style="Cluster",
    palette="Set2",
    s=100
)
plt.title("Client Segmentation based on ROAS & Conversion Rate")
plt.xlabel("ROAS")
plt.ylabel("Conversion Rate")
save_plot(fig, 'Client_Segmentation_ROAS_vs_CR.png')

# Output cluster assignments
print("\nClient Segmentation Result\n")
print(cluster_features)

# Save cluster assignments
cluster_features.to_csv('visualizations/clustered_clients.csv')
print("Visualisasi 'Client_Segmentation_ROAS_vs_CR.png' dan data 'clustered_clients.csv' telah disimpan.")
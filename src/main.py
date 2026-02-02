import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- Load dataset ----------
df = pd.read_csv("data/spotify_tracks.csv")

print("Dataset loaded")
print(df.head())

# ---------- Basic cleaning ----------
if "popularity" not in df.columns:
    raise Exception("Dataset needs a 'popularity' column")

df = df.dropna(subset=["popularity"])

# ---------- Popularity distribution ----------
plt.hist(df["popularity"], bins=30)
plt.title("Popularity Distribution")
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.savefig("outputs/popularity_distribution.png")
plt.clf()

# ---------- Popularity vs Danceability ----------
if "danceability" in df.columns:
    plt.scatter(df["danceability"], df["popularity"])
    plt.title("Popularity vs Danceability")
    plt.xlabel("Danceability")
    plt.ylabel("Popularity")
    plt.savefig("outputs/popularity_vs_danceability.png")
    plt.clf()

# ---------- Popularity vs Energy ----------
if "energy" in df.columns:
    plt.scatter(df["energy"], df["popularity"])
    plt.title("Popularity vs Energy")
    plt.xlabel("Energy")
    plt.ylabel("Popularity")
    plt.savefig("outputs/popularity_vs_energy.png")
    plt.clf()

# ---------- Correlation heatmap ----------
features = ["popularity", "danceability", "energy", "valence", "tempo"]
features = [f for f in features if f in df.columns]

corr = df[features].corr()

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=45)
plt.yticks(range(len(features)), features)
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.clf()

# ---------- Summary text ----------
summary = f"""
Average popularity: {df['popularity'].mean():.2f}
Max popularity: {df['popularity'].max()}
Min popularity: {df['popularity'].min()}
"""

with open("outputs/summary.txt", "w") as f:
    f.write(summary)

print("Done! Check the outputs folder ✅")

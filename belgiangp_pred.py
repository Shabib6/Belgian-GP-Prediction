import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

fastf1.Cache.enable_cache("f1_cache")


session_2024 = fastf1.get_session(2024, 8, "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[col] = laps_2024[col].dt.total_seconds()

sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time": "mean",
    "Sector2Time": "mean",
    "Sector3Time": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime(s)"] = (
    sector_times_2024["Sector1Time"] +
    sector_times_2024["Sector2Time"] +
    sector_times_2024["Sector3Time"]
)

clean_air_race_pace = {
    "OCO": 101.886118,
    "LEC": 101.163576,
    "PIA": 99.408667,
    "RUS": 100.847000,
    "VER": 100.950061, 
    "STR": 100.845844,
    "SAI": 101.482406,
    "HAM": 100.236000,
    "ALO": 100.790656,
    "NOR": 99.588424, 
    "HUL": 100.309938,
    "ALB": 101.85,
    "GAS": 102.28
}

qualifying_2025 = pd.DataFrame({
    "Driver": ["VER","NOR","PIA","RUS","SAI","ALB","LEC","OCO","HAM","STR","GAS","ALO","HUL"],
    "QualifyingTime (s)": [
        100.903, 100.562, 100.647, 101.260, 101.758, 101.201,
        100.900, 101.525, 101.939, 102.502, 101.633, 102.385, 101.707
    ]
})

driver_wet_performance = {
    "VER": 0.975196,  "NOR": 0.978179,  "PIA": 0.975500,
    "RUS": 0.968678,  "SAI": 0.974200,  "ALB": 0.972500,
    "LEC": 0.975862,  "OCO": 0.973000,  "HAM": 0.976464,
    "STR": 0.971800,  "GAS": 0.979000,  "ALO": 0.972655,
    "HUL": 0.980000
}

df = qualifying_2025.copy()
df["CleanAirRacePace (s)"] = df["Driver"].map(clean_air_race_pace)
df["WetPerformanceFactor"] = df["Driver"].map(driver_wet_performance)
df["AdjustedRacePace"] = df["CleanAirRacePace (s)"] * df["WetPerformanceFactor"]
df["QualiVsRaceRatio"] = df["QualifyingTime (s)"] / df["AdjustedRacePace"]


df["Winner"] = df["Driver"].apply(lambda x: 1 if x == "VER" else 0)


X = df[["CleanAirRacePace (s)", "WetPerformanceFactor", "AdjustedRacePace", "QualiVsRaceRatio"]]
y = df["Winner"]

# Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# For simplicity, I will use the entire dataset for training in this example

model = GradientBoostingClassifier()
model.fit(X, y)
platt_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
platt_model.fit(X, y)


calibrated_probs = platt_model.predict_proba(X)[:, 1]

# I added uncertainty due to the likelihood of rain and safety car deployment
# This is a simplified model and should be adjusted based on real-world data and conditions
def add_uncertainty(probs, is_rain=True, safety_car_chance=0.05):
    if is_rain:
        weather_factor = 1.0
        tire_variance = 0.05
    else:
        weather_factor = 0.2
        tire_variance = 0.01

    noisy_probs = []
    for prob in probs:
        noise = np.random.normal(loc=0, scale=tire_variance + safety_car_chance + weather_factor * 0.02)
        adjusted = prob + noise
        noisy_probs.append(adjusted)

    noisy_probs = np.clip(noisy_probs, 0, 1)
    total = sum(noisy_probs)
    return [p / total for p in noisy_probs]

adjusted_probs = add_uncertainty(calibrated_probs, is_rain=True)
df["WinProbability"] = adjusted_probs

df_sorted = df.sort_values("WinProbability", ascending=False)
print(df_sorted[["Driver", "WinProbability"]])
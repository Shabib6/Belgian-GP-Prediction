import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache("f1_cache")

#loading data - 2024 Belgian Grand Prix
session_2024 = fastf1.get_session(2024, 8, "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

#convert laptime and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[col] = laps_2024[col].dt.total_seconds()

#aggregate sector times by driver
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
    "HUL": 100.309938
}

#Quali data from 2025 BlegianGP
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER","NOR","PIA","RUS","SAI","ALB","LEC","OCO","HAM","STR","GAS","ALO","HUL"],
    "QualifyingTime (s)": [
        100.903, #VER
        100.562, #NOR
        100.647, #PIA
        101.260, # RUS
        101.758, # SAI
        101.201, # ALB
        100.900, # LEC
        101.525, # OCO
        101.939, # HAM
        102.502, # STR
        101.633, # GAS
        102.385, # ALO
        101.707  # HUL
    ]
})
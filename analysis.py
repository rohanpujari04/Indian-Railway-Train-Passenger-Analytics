
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# STEP 1: LOAD AND CLEAN DATA
# ==================================================

input_file = "data/Train_details_22122017.csv"
output_file = "Indian_Railway_PowerBI_Data.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Dataset not found: {input_file}")

df = pd.read_csv(input_file, low_memory=False)

print("===== DATASET OVERVIEW =====")
print("Original Shape:", df.shape)
print("Original Missing Values:", df.isna().sum().sum())

# Clean text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Convert numeric columns
for col in ["Train No", "SEQ", "Distance"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows missing essential information
required = [
    "Train No", "SEQ", "Arrival time", "Departure Time",
    "Distance", "Source Station", "Source Station Name",
    "Destination Station", "Destination Station Name"
]
df = df.dropna(subset=required).copy()

# Convert time values
for col in ["Arrival time", "Departure Time"]:
    df[col] = pd.to_datetime(
        df[col], format="%H:%M:%S", errors="coerce"
    )

df = df.dropna(
    subset=["Arrival time", "Departure Time"]
).copy()

df["Train No"] = df["Train No"].astype("int64")

print("\n===== CLEANING SUMMARY =====")
print("Cleaned Shape:", df.shape)
print("Missing Values:", df.isna().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

# ==================================================
# STEP 2: HALT TIME ANALYSIS
# ==================================================

arrival = df["Arrival time"]
departure = df["Departure Time"]

df["Halt Time"] = (
    (departure - arrival).dt.total_seconds() / 60
)

# Correct overnight time differences
df.loc[df["Halt Time"] < 0, "Halt Time"] += 1440

first_seq = df.groupby("Train No")["SEQ"].transform("min")
last_seq = df.groupby("Train No")["SEQ"].transform("max")

zero_arrival = (
    df["Arrival time"].dt.strftime("%H:%M:%S") == "00:00:00"
)
zero_departure = (
    df["Departure Time"].dt.strftime("%H:%M:%S") == "00:00:00"
)

df["Halt Time Cleaned"] = df["Halt Time"]

# Exclude unknown halt times at route endpoints
df.loc[
    (df["SEQ"] == first_seq) & zero_arrival,
    "Halt Time Cleaned"
] = np.nan

df.loc[
    (df["SEQ"] == last_seq) & zero_departure,
    "Halt Time Cleaned"
] = np.nan

print("\n===== HALT TIME SUMMARY =====")
print(df["Halt Time Cleaned"].describe())
print("Negative Halt Times:",
      (df["Halt Time Cleaned"] < 0).sum())
print("Halt Times Above 60 Minutes:",
      (df["Halt Time Cleaned"] > 60).sum())
print("Unknown Halt Times:",
      df["Halt Time Cleaned"].isna().sum())

# ==================================================
# STEP 3: TRAIN AND STATION ANALYSIS
# ==================================================

train_summary = (
    df.groupby(["Train No", "Train Name"])
      .agg(
          Station_Records=("Station Code", "size"),
          Start_Distance=("Distance", "min"),
          End_Distance=("Distance", "max")
      )
      .reset_index()
)

train_summary["Route_Length"] = (
    train_summary["End_Distance"]
    - train_summary["Start_Distance"]
)

print("\n===== TOP 10 TRAINS BY STATION RECORDS =====")
print(
    train_summary.nlargest(10, "Station_Records")[
        ["Train No", "Train Name", "Station_Records"]
    ].to_string(index=False)
)

station_summary = (
    df.groupby(["Station Code", "Station Name"])
      .size()
      .reset_index(name="Train_Records")
      .sort_values("Train_Records", ascending=False)
)

print("\n===== TOP 10 STATIONS =====")
print(station_summary.head(10).to_string(index=False))

print("\n===== LONGEST TRAIN ROUTES =====")
print(
    train_summary.nlargest(10, "Route_Length")[
        ["Train No", "Train Name", "Route_Length"]
    ].to_string(index=False)
)

# ==================================================
# STEP 4: ROUTE CATEGORY ANALYSIS
# ==================================================

bins = [0, 50, 100, 250, 500, 1000, 2000, 3000, np.inf]

labels = [
    "0-50 km",
    "51-100 km",
    "101-250 km",
    "251-500 km",
    "501-1000 km",
    "1001-2000 km",
    "2001-3000 km",
    "3001+ km"
]

train_summary["Route_Length_Category"] = pd.cut(
    train_summary["Route_Length"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

route_counts = (
    train_summary["Route_Length_Category"]
    .value_counts(sort=False)
    .reindex(labels, fill_value=0)
)

print("\n===== ROUTE LENGTH DISTRIBUTION =====")
print(route_counts)

# Add route information to every station record
# Existing columns are preserved; two columns are added.
df["_row_order"] = np.arange(len(df))

df = df.merge(
    train_summary[
        ["Train No", "Train Name",
         "Route_Length", "Route_Length_Category"]
    ],
    on=["Train No", "Train Name"],
    how="left",
    validate="many_to_one"
)

df = (
    df.sort_values("_row_order")
      .drop(columns="_row_order")
      .reset_index(drop=True)
)

# ==================================================
# STEP 5: IMPORTANT VISUALIZATIONS
# ==================================================

# Chart 1: Top 10 trains by station records
top_trains = train_summary.nlargest(10, "Station_Records")

plt.figure(figsize=(12, 6))
plt.bar(top_trains["Train Name"], top_trains["Station_Records"])
plt.title("Top 10 Trains by Station Records")
plt.xlabel("Train Name")
plt.ylabel("Station Records")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Chart 2: Top 10 stations
top_stations = station_summary.head(10)

plt.figure(figsize=(12, 6))
plt.bar(top_stations["Station Name"], top_stations["Train_Records"])
plt.title("Top 10 Stations by Train Records")
plt.xlabel("Station")
plt.ylabel("Train Records")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Chart 3: Longest train routes
top_routes = train_summary.nlargest(10, "Route_Length")

plt.figure(figsize=(12, 6))
plt.barh(
    top_routes["Train Name"],
    top_routes["Route_Length"]
)
plt.title("Top 10 Longest Train Routes")
plt.xlabel("Route Length (km)")
plt.ylabel("Train")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Chart 4: Route length distribution
plt.figure(figsize=(11, 6))
plt.bar(route_counts.index.astype(str), route_counts.values)
plt.title("Train Count by Route Length")
plt.xlabel("Route Length Category")
plt.ylabel("Number of Trains")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Chart 5: Route length vs station records
plt.figure(figsize=(10, 6))
plt.scatter(
    train_summary["Route_Length"],
    train_summary["Station_Records"],
    alpha=0.4
)
plt.title("Route Length vs Station Records")
plt.xlabel("Route Length (km)")
plt.ylabel("Station Records")
plt.tight_layout()
plt.show()

correlation = train_summary[
    ["Route_Length", "Station_Records"]
].corr().iloc[0, 1]

print("\nRoute Length / Station Records Correlation:",
      round(correlation, 4))

# ==================================================
# STEP 6: EXPORT POWER BI DATASET
# ==================================================

# Preserve unknown halt times in the analysis.
# Use 0 in the export, matching the previous Power BI process.
powerbi_df = df.copy()

powerbi_df["Halt Time Cleaned"] = (
    powerbi_df["Halt Time Cleaned"].fillna(0)
)

output_file = "Indian_Railway_PowerBI_Data_updated.csv"
powerbi_df.to_csv(output_file, index=False)

print("Power BI CSV successfully saved:", output_file)

# ==================================================
# STEP 7: FINAL VALIDATION
# ==================================================

final_df = pd.read_csv(output_file, low_memory=False)

print("\n===== FINAL POWER BI VALIDATION =====")
print("File:", os.path.abspath(output_file))
print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))
print("Column Names:", final_df.columns.tolist())
print("Missing Values:", final_df.isna().sum().sum())
print("Duplicate Rows:", final_df.duplicated().sum())
print("File Size (MB):",
      round(os.path.getsize(output_file) / (1024 * 1024), 2))

print("\nTotal Trains:", df["Train No"].nunique())
print("Total Train Names:", df["Train Name"].nunique())
print("Total Stations:", df["Station Code"].nunique())
print("Maximum Route Length:",
      round(train_summary["Route_Length"].max(), 2))

if (
    len(final_df) == len(df)
    and final_df.isna().sum().sum() == 0
    and final_df.duplicated().sum() == 0
):
    print("\nPOWER BI DATA VALIDATION: PASSED")
else:
    print("\nPOWER BI DATA VALIDATION: CHECK REQUIRED")

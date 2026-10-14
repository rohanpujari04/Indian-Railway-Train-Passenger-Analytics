import pandas as pd
import numpy as np

df = pd.read_csv("data/Train_details_22122017.csv", low_memory=False)

print("Shape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nSEQ Sample Values:")
print(df["SEQ"].head(10))

print("\nDistance Sample Values:")
print(df["Distance"].head(10))

# Step 3: Convert SEQ and Distance to numeric

df["SEQ"] = pd.to_numeric(df["SEQ"], errors="coerce")
df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")

print("\nData Types after Conversion:")
print(df[["SEQ", "Distance"]].dtypes)

# Step 4: Check missing values after conversion

print("\nMissing Values after Conversion:")
print(df.isnull().sum())

# Step 5: Inspect rows with missing values

missing_rows = df[df.isnull().any(axis=1)]

print("\nRows with Missing Values:")
print(missing_rows)

# Step 6: Inspect problematic rows with surrounding data

print(df.loc[2305:2323])

# Step 7: Remove rows with missing important information

df = df.dropna(subset=[
    "SEQ",
    "Arrival time",
    "Departure Time",
    "Distance",
    "Source Station",
    "Source Station Name",
    "Destination Station",
    "Destination Station Name"
])

print("\nShape after removing invalid rows:")
print(df.shape)

# Step 8: Final missing value check

print("\nMissing Values after Cleaning:")
print(df.isnull().sum())

# Step 9: Check extra spaces in text columns

text_columns = df.select_dtypes(include="object").columns

for col in text_columns:
    df[col] = df[col].str.strip()

print("\nText columns cleaned successfully.")

# Step 10: Check Train No values

print("\nTrain No Data Type:")
print(df["Train No"].dtype)

print("\nNon-Numeric Train No Values:")
print(df[~df["Train No"].str.isnumeric()]["Train No"].unique())

# Step 11: Convert Train No to numeric

df["Train No"] = pd.to_numeric(df["Train No"])

print("\nTrain No Data Type after Conversion:")
print(df["Train No"].dtype)

# Step 12: Check time values

print("\nArrival Time Sample:")
print(df["Arrival time"].head(10))

print("\nDeparture Time Sample:")
print(df["Departure Time"].head(10))

# Step 13: Convert time columns to datetime

df["Arrival time"] = pd.to_datetime(df["Arrival time"], format="%H:%M:%S")
df["Departure Time"] = pd.to_datetime(df["Departure Time"], format="%H:%M:%S")

print("\nTime Data Types:")
print(df[["Arrival time", "Departure Time"]].dtypes)

print("\nFinal Data Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nUnique Values:")
print("Train Names:", df["Train Name"].nunique())
print("Station Codes:", df["Station Code"].nunique())
print("Station Names:", df["Station Name"].nunique())
print("Source Stations:", df["Source Station"].nunique())
print("Destination Stations:", df["Destination Station"].nunique())

print("\nDistance Summary:")
print("Minimum Distance:", df["Distance"].min())
print("Maximum Distance:", df["Distance"].max())
print("Negative Distance:", (df["Distance"] < 0).sum())

print("\nSEQ Summary:")
print("Minimum SEQ:", df["SEQ"].min())
print("Maximum SEQ:", df["SEQ"].max())
print("Missing SEQ:", df["SEQ"].isnull().sum())

print("\nTime Range:")

print("Earliest Arrival Time:", df["Arrival time"].min().time())
print("Latest Arrival Time:", df["Arrival time"].max().time())

print("Earliest Departure Time:", df["Departure Time"].min().time())
print("Latest Departure Time:", df["Departure Time"].max().time())

df["Halt Time"] = (
    df["Departure Time"] - df["Arrival time"]
).dt.total_seconds() / 60

print("\nHalt Time Sample:")
print(df[["Arrival time", "Departure Time", "Halt Time"]].head(10))

print("\nZero Time Count:")

print("Arrival Time 00:00:", (df["Arrival time"].dt.time == pd.Timestamp("00:00:00").time()).sum())

print("Departure Time 00:00:", (df["Departure Time"].dt.time == pd.Timestamp("00:00:00").time()).sum())

print("\nNegative Halt Time:")
print((df["Halt Time"] < 0).sum())

df["Halt Time"] = (
    (df["Departure Time"] - df["Arrival time"])
    .dt.total_seconds() / 60
)

df.loc[df["Halt Time"] < 0, "Halt Time"] += 24 * 60

print("\nCorrected Halt Time:")
print(df[["Arrival time", "Departure Time", "Halt Time"]].head(10))

print("\nHalt Time Statistics:")
print(df["Halt Time"].describe())

print("\nHalt Time > 10 Minutes:")
print((df["Halt Time"] > 10).sum())

print("\nHalt Time > 60 Minutes:")
print((df["Halt Time"] > 60).sum())

zero_arrival = df["Arrival time"].dt.time == pd.Timestamp("00:00:00").time()
zero_departure = df["Departure Time"].dt.time == pd.Timestamp("00:00:00").time()

print("\nHalt Time > 60 with 00:00:")

print("Zero Arrival & Halt > 60:",
      ((zero_arrival) & (df["Halt Time"] > 60)).sum())

print("Zero Departure & Halt > 60:",
      ((zero_departure) & (df["Halt Time"] > 60)).sum())

zero_arrival = df["Arrival time"].dt.time == pd.Timestamp("00:00:00").time()
zero_departure = df["Departure Time"].dt.time == pd.Timestamp("00:00:00").time()

normal_halt = df.loc[
    (~zero_arrival) & (~zero_departure),
    "Halt Time"
]

print("\nNormal Halt Time Statistics:")
print(normal_halt.describe())

print("\nNegative Halt Time After Correction:")
print((df["Halt Time"] < 0).sum())

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Data Types:")
print(df.dtypes)

print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nHalt Time Value Counts:")
print(df["Halt Time"].value_counts().sort_index().head(20))

print("\nHalt Time Greater Than 60 Minutes:")
print(df[df["Halt Time"] > 60][
    ["Train No", "Train Name", "Station Name",
     "Arrival time", "Departure Time", "Halt Time"]
].head(20))

print("\nZero Arrival Time:")
print((df["Arrival time"].dt.time == pd.Timestamp("00:00:00").time()).sum())

print("\nZero Departure Time:")
print((df["Departure Time"].dt.time == pd.Timestamp("00:00:00").time()).sum())

print("\nRecords with Zero Arrival Time:")
print(df[df["Arrival time"].dt.time == pd.Timestamp("00:00:00").time()][
    ["Train No", "Train Name", "Station Name",
     "Arrival time", "Departure Time", "Halt Time"]
].head(20))

suspicious = df[
    (
        (df["Arrival time"].dt.time == pd.Timestamp("00:00:00").time()) |
        (df["Departure Time"].dt.time == pd.Timestamp("00:00:00").time())
    )
    & (df["Halt Time"] > 60)
]

print("\nSuspicious Halt Records:")
print(suspicious.shape[0])

print("\nTop 20 Highest Halt Times:")
print(df["Halt Time"].nlargest(20).to_list())

print("\nNormal Halt Time Statistics:")
print(df[df["Halt Time"] <= 60]["Halt Time"].describe())

print("\nHalt Time Greater Than 60:")
print((df["Halt Time"] > 60).sum())

print("Top 20 Highest Halt Time Records:")
print(df.nlargest(20, "Halt Time")[
    ["Train No", "Train Name", "Station Name",
     "Arrival time", "Departure Time", "Halt Time"]
])

print("Halt Time Greater Than 60:")
print((df["Halt Time"] > 60).sum())

print("\nHalt Time Greater Than 60 Sample:")
print(df[df["Halt Time"] > 60][
    ["Train No", "Train Name", "Station Name",
     "Arrival time", "Departure Time", "Halt Time"]
].head(20))

print("Arrival Time = 00:00:")
print((df["Arrival time"].dt.time == pd.Timestamp("00:00:00").time()).sum())

print("\nDeparture Time = 00:00:")
print((df["Departure Time"].dt.time == pd.Timestamp("00:00:00").time()).sum())

print("Arrival Time = 00:00:")
print((df["Arrival time"].dt.time == pd.Timestamp("00:00:00").time()).sum())

midnight_halt = df[
    (df["Halt Time"] > 60) &
    (
        (df["Arrival time"].dt.time == pd.Timestamp("00:00:00").time()) |
        (df["Departure Time"].dt.time == pd.Timestamp("00:00:00").time())
    )
]

print("Suspicious Midnight Halt Records:")
print("Count:", len(midnight_halt))

print("\nSample Records:")
print(midnight_halt[
    ["Train No", "Train Name", "Station Name",
     "Arrival time", "Departure Time", "Halt Time"]
].head(20))

print("Arrival Time Value Counts:")
print(df["Arrival time"].dt.strftime("%H:%M:%S").value_counts().head(10))

print("\nDeparture Time Value Counts:")
print(df["Departure Time"].dt.strftime("%H:%M:%S").value_counts().head(10))

normal_halt = df[
    (df["Arrival time"].dt.time != pd.Timestamp("00:00:00").time()) &
    (df["Departure Time"].dt.time != pd.Timestamp("00:00:00").time())
]

print("Normal Time Records:", len(normal_halt))

print("\nNormal Halt Time Statistics:")
print(normal_halt["Halt Time"].describe())

print("\nNormal Halt Time Greater Than 60:")
print((normal_halt["Halt Time"] > 60).sum())

normal_high_halt = normal_halt[
    normal_halt["Halt Time"] > 60
]

print("Normal High Halt Records:")
print("Count:", len(normal_high_halt))

print("\nRecords:")
print(normal_high_halt[
    ["Train No", "Train Name", "Station Name",
     "Arrival time", "Departure Time", "Halt Time"]
].head(30))

extreme_halt = df[df["Halt Time"] > 300]

print("Extreme Halt Time Records:")
print("Count:", len(extreme_halt))

print("\nExtreme Halt Time Sample:")
print(extreme_halt[
    ["Train No", "Train Name", "Station Name",
     "Arrival time", "Departure Time", "Halt Time"]
].head(20))

arrival = df["Arrival time"]
departure = df["Departure Time"]

corrected_halt = (
    (departure - arrival).dt.total_seconds() / 60
)

corrected_halt = corrected_halt.where(
    corrected_halt >= 0,
    corrected_halt + 1440
)

print("Corrected Halt Time Sample:")
print(corrected_halt.head(20))

# Check selected trains with SEQ
check_trains = df[df["Train No"].isin([107, 108])]

print(check_trains[
    ["Train No", "SEQ", "Station Name",
     "Arrival time", "Departure Time", "Halt Time"]
].head(30))

# Identify first and last station records
first_station = df.groupby("Train No")["SEQ"].transform("min")
last_station = df.groupby("Train No")["SEQ"].transform("max")

# First station with Arrival Time = 00:00
first_station_zero = df[
    (df["SEQ"] == first_station) &
    (df["Arrival time"].dt.hour == 0) &
    (df["Arrival time"].dt.minute == 0)
]

# Last station with Departure Time = 00:00
last_station_zero = df[
    (df["SEQ"] == last_station) &
    (df["Departure Time"].dt.hour == 0) &
    (df["Departure Time"].dt.minute == 0)
]

print("First Station with 00:00 Arrival:")
print("Count:", len(first_station_zero))

print("\nLast Station with 00:00 Departure:")
print("Count:", len(last_station_zero))

# Create a copy of current Halt Time
df["Halt Time Cleaned"] = df["Halt Time"]

# First station: Arrival time is 00:00
df.loc[
    (df["SEQ"] == first_station) &
    (df["Arrival time"].dt.hour == 0) &
    (df["Arrival time"].dt.minute == 0),
    "Halt Time Cleaned"
] = np.nan

# Last station: Departure time is 00:00
df.loc[
    (df["SEQ"] == last_station) &
    (df["Departure Time"].dt.hour == 0) &
    (df["Departure Time"].dt.minute == 0),
    "Halt Time Cleaned"
] = np.nan

print("Cleaned Halt Time Sample:")
print(df[
    ["Train No", "SEQ", "Station Name",
     "Arrival time", "Departure Time",
     "Halt Time", "Halt Time Cleaned"]
].head(10))

print("Cleaned Halt Time Statistics:")
print(df["Halt Time Cleaned"].describe())

print("\nMissing Halt Time After Cleaning:")
print(df["Halt Time Cleaned"].isna().sum())

# Check remaining high halt times after cleaning

high_halt = df[df["Halt Time Cleaned"] > 60]
extreme_halt_cleaned = df[df["Halt Time Cleaned"] > 300]

print("Halt Time > 60 Minutes:")
print("Count:", len(high_halt))

print("\nHalt Time > 300 Minutes:")
print("Count:", len(extreme_halt_cleaned))

print("Remaining Halt Time > 60 Minutes:")

print(high_halt[
    ["Train No", "Train Name", "SEQ", "Station Name",
     "Arrival time", "Departure Time", "Halt Time Cleaned"]
].sort_values("Halt Time Cleaned", ascending=False).head(30))

print("Final Halt Time Validation:")

print("\nNegative Values:")
print((df["Halt Time Cleaned"] < 0).sum())

print("\nValues Greater Than 300:")
print((df["Halt Time Cleaned"] > 300).sum())

print("\nMissing Values:")
print(df["Halt Time Cleaned"].isna().sum())

print("\nFinal Dataset Shape:")
print(df.shape)

print("Rows with Missing Halt Time Cleaned:")

missing_halt_rows = df[df["Halt Time Cleaned"].isna()]

print("Count:", len(missing_halt_rows))

print("\nSample:")
print(
    missing_halt_rows[
        [
            "Train No",
            "Train Name",
            "SEQ",
            "Station Code",
            "Station Name",
            "Arrival time",
            "Departure Time",
            "Halt Time",
            "Halt Time Cleaned"
        ]
    ].head(20)
)

print("Current Dataset Shape:")
print(df.shape)

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nRows with completely missing values:")
print(df.isna().all(axis=1).sum())

print("\nRows with Missing Train No:")
print(df["Train No"].isna().sum())

print("\nRows with Missing Train Name:")
print(df["Train Name"].isna().sum())

print("Train-wise Row Count Check:")

train_counts = df.groupby("Train No").size()

print("\nMinimum rows for a train:")
print(train_counts.min())

print("\nMaximum rows for a train:")
print(train_counts.max())

print("\nNumber of unique trains:")
print(df["Train No"].nunique())

print("Index Validation:")

print("\nMinimum Index:")
print(df.index.min())

print("\nMaximum Index:")
print(df.index.max())

print("\nNumber of Index Values:")
print(len(df.index))

print("\nUnique Index Values:")
print(df.index.nunique())

print("\nMissing Index Values:")
expected_index = set(range(186124))
missing_indices = expected_index - set(df.index)

print(len(missing_indices))
print(sorted(missing_indices)[:30])

print("Checking rows around missing indexes:")

missing_indices = [2308, 2309, 2320, 2321,
                   135826, 135827,
                   135884, 135885,
                   140468, 140469]

for idx in missing_indices:
    print("\n" + "=" * 80)
    print("Missing Index:", idx)

    print("\nPrevious Row:")
    print(df.loc[idx - 1] if idx - 1 in df.index else "Not Available")

    print("\nNext Row:")
    print(df.loc[idx + 1] if idx + 1 in df.index else "Not Available")

    print("Exact Context Around Missing Indexes:")

missing_indices = [2308, 2309, 2320, 2321,
                   135826, 135827,
                   135884, 135885,
                   140468, 140469]

for idx in missing_indices:
    print("\n" + "=" * 90)
    print("Missing Index:", idx)

    start_idx = max(0, idx - 2)
    end_idx = min(df.index.max(), idx + 2)

    context = df.loc[
        (df.index >= start_idx) &
        (df.index <= end_idx),
        ["Train No", "Train Name", "SEQ", "Station Code",
         "Station Name", "Arrival time", "Departure Time",
         "Halt Time", "Halt Time Cleaned"]
    ]

    print(context)

    print("Checking Original 10 Removed Rows:")

original_df = pd.read_csv(
    "data/Train_details_22122017.csv",
    low_memory=False
)

removed_indices = [
    2308, 2309, 2320, 2321,
    135826, 135827,
    135884, 135885,
    140468, 140469
]

check_columns = [
    "SEQ",
    "Arrival time",
    "Departure Time",
    "Distance",
    "Source Station",
    "Source Station Name",
    "Destination Station",
    "Destination Station Name"
]

removed_rows = original_df.loc[
    removed_indices,
    ["Train No", "Train Name", "Station Code", "Station Name"] + check_columns
]

print(removed_rows)

print("\nMissing Values in Removed Rows:")
print(removed_rows[check_columns].isna().sum())

print("===== FINAL HALT TIME VALIDATION =====")

print("\n1. Negative Values:")
print((df["Halt Time Cleaned"] < 0).sum())

print("\n2. Values Greater Than 300 Minutes:")
print((df["Halt Time Cleaned"] > 300).sum())

print("\n3. Values Greater Than 60 Minutes:")
print((df["Halt Time Cleaned"] > 60).sum())

print("\n4. Missing Values:")
print(df["Halt Time Cleaned"].isna().sum())

print("\n5. Halt Time Statistics:")
print(df["Halt Time Cleaned"].describe())

print("\n6. Final Dataset Shape:")
print(df.shape)

print("===== FINAL HALT TIME VALIDATION =====")

print("\n1. Negative Values:")
print((df["Halt Time Cleaned"] < 0).sum())

print("\n2. Values Greater Than 300 Minutes:")
print((df["Halt Time Cleaned"] > 300).sum())

print("\n3. Values Greater Than 60 Minutes:")
print((df["Halt Time Cleaned"] > 60).sum())

print("\n4. Missing Values:")
print(df["Halt Time Cleaned"].isna().sum())

print("\n5. Halt Time Statistics:")
print(df["Halt Time Cleaned"].describe())

print("\n6. Final Dataset Shape:")
print(df.shape)

print("===== STEP 33: FINAL DATA QUALITY CHECK =====")

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Duplicate Rows:")
print(df.duplicated().sum())

print("\n3. Missing Values:")
print(df.isna().sum())

print("\n4. Data Types:")
print(df.dtypes)

print("\n5. Unique Train Numbers:")
print(df["Train No"].nunique())

print("\n6. Unique Train Names:")
print(df["Train Name"].nunique())

print("\n7. Unique Station Codes:")
print(df["Station Code"].nunique())

print("\n8. Unique Station Names:")
print(df["Station Name"].nunique())

##EDA##

print("===== STEP 34: BASIC EDA OVERVIEW =====")

print("\n1. Total Train Numbers:")
print(df["Train No"].nunique())

print("\n2. Total Train Names:")
print(df["Train Name"].nunique())

print("\n3. Total Stations:")
print(df["Station Code"].nunique())

print("\n4. Total Route Records:")
print(len(df))

print("\n5. Total Distance:")
print(df["Distance"].sum())

print("\n6. Average Distance:")
print(df["Distance"].mean())

print("\n7. Maximum Distance:")
print(df["Distance"].max())

print("\n8. Average Halt Time:")
print(df["Halt Time Cleaned"].mean())

print("\n9. Maximum Halt Time:")
print(df["Halt Time Cleaned"].max())

print("\n10. Minimum Halt Time:")
print(df["Halt Time Cleaned"].min())

print("===== STEP 35: TRAIN-WISE ANALYSIS =====")

train_record_count = (
    df.groupby(["Train No", "Train Name"])
      .size()
      .reset_index(name="Station Records")
      .sort_values("Station Records", ascending=False)
)

print("\nTop 10 Trains by Station Records:")
print(train_record_count.head(10))

import matplotlib.pyplot as plt

top_10_trains = train_record_count.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_10_trains["Train Name"],
    top_10_trains["Station Records"]
)

plt.xlabel("Train Name")
plt.ylabel("Number of Station Records")
plt.title("Top 10 Trains by Number of Station Records")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("===== STEP 37: STATION-WISE ANALYSIS =====")

station_record_count = (
    df.groupby(["Station Code", "Station Name"])
      .size()
      .reset_index(name="Train Records")
      .sort_values("Train Records", ascending=False)
)

print("\nTop 10 Stations by Train Records:")
print(station_record_count.head(10))

import matplotlib.pyplot as plt

top_10_stations = station_record_count.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_10_stations["Station Name"],
    top_10_stations["Train Records"]
)

plt.xlabel("Station Name")
plt.ylabel("Number of Train Records")
plt.title("Top 10 Stations by Train Records")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

print("===== STEP 39: DISTANCE ANALYSIS =====")

print("\nDistance Statistics:")
print(df["Distance"].describe())

print("\nTop 10 Records by Distance:")

top_10_distance = (
    df[["Train No", "Train Name", "Station Name", "Distance"]]
    .sort_values("Distance", ascending=False)
    .head(10)
)

print(top_10_distance)

import matplotlib.pyplot as plt

top_10_distance = (
    df[["Train No", "Train Name", "Station Name", "Distance"]]
    .sort_values("Distance", ascending=False)
    .head(10)
)

labels = (
    top_10_distance["Train No"].astype(str)
    + " - "
    + top_10_distance["Station Name"]
)

plt.figure(figsize=(12, 6))

plt.bar(labels, top_10_distance["Distance"])

plt.xlabel("Train - Station")
plt.ylabel("Distance (km)")
plt.title("Top 10 Records by Distance")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

top_10_distance = (
    df[["Train No", "Train Name", "Station Name", "Distance"]]
    .sort_values("Distance", ascending=False)
    .head(10)
)

labels = (
    top_10_distance["Train No"].astype(str)
    + " - "
    + top_10_distance["Station Name"]
)

plt.figure(figsize=(12, 6))

plt.bar(labels, top_10_distance["Distance"])

plt.xlabel("Train - Station")
plt.ylabel("Distance (km)")
plt.title("Top 10 Records by Distance")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

top_10_stations = station_record_count.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_10_stations["Station Name"],
    top_10_stations["Train Records"]
)

plt.xlabel("Station Name")
plt.ylabel("Number of Train Records")
plt.title("Top 10 Stations by Train Records")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

top_10_distance = (
    df[["Train No", "Train Name", "Station Name", "Distance"]]
    .sort_values("Distance", ascending=False)
    .head(10)
)

labels = (
    top_10_distance["Train No"].astype(str)
    + " - "
    + top_10_distance["Station Name"]
)

plt.figure(figsize=(12, 6))

plt.bar(labels, top_10_distance["Distance"])

plt.xlabel("Train - Station")
plt.ylabel("Distance (km)")
plt.title("Top 10 Records by Distance")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

top_10_distance = (
    df[["Train No", "Train Name", "Station Name", "Distance"]]
    .sort_values("Distance", ascending=False)
    .head(10)
)

labels = (
    top_10_distance["Train No"].astype(str)
    + " - "
    + top_10_distance["Station Name"]
)

plt.figure(figsize=(12, 6))

plt.bar(labels, top_10_distance["Distance"])

plt.xlabel("Train - Station")
plt.ylabel("Distance (km)")
plt.title("Top 10 Records by Distance")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

print("===== STEP 40: DISTANCE VISUALIZATION =====")

import matplotlib.pyplot as plt

top_10_distance_chart = (
    df[["Train No", "Train Name", "Station Name", "Distance"]]
    .sort_values("Distance", ascending=False)
    .head(10)
)

labels = (
    top_10_distance_chart["Train No"].astype(str)
    + " - "
    + top_10_distance_chart["Station Name"]
)

plt.figure(figsize=(12, 6))

plt.bar(
    labels,
    top_10_distance_chart["Distance"]
)

plt.xlabel("Train - Station")
plt.ylabel("Distance (km)")
plt.title("Top 10 Records by Distance")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

print("\nStep 40 completed successfully.")

print("===== STEP 41: TRAIN-WISE MAXIMUM DISTANCE ANALYSIS =====")

train_max_distance = (
    df.groupby(["Train No", "Train Name"])["Distance"]
      .max()
      .reset_index(name="Maximum Distance")
      .sort_values("Maximum Distance", ascending=False)
)

print("\nTop 10 Trains by Maximum Distance:")

print(train_max_distance.head(10))

print("===== STEP 42: TRAIN-WISE DISTANCE VISUALIZATION =====")

import matplotlib.pyplot as plt

top_10_train_distance = train_max_distance.head(10)

labels = (
    top_10_train_distance["Train No"].astype(str)
    + " - "
    + top_10_train_distance["Train Name"]
)

plt.figure(figsize=(12, 6))

plt.bar(
    labels,
    top_10_train_distance["Maximum Distance"]
)

plt.xlabel("Train")
plt.ylabel("Maximum Distance (km)")
plt.title("Top 10 Trains by Maximum Distance")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

print("\nStep 42 completed successfully.")

print("===== STEP 43: TRAIN-WISE DISTANCE SUMMARY =====")

train_distance_summary = (
    df.groupby(["Train No", "Train Name"])["Distance"]
      .agg(
          Minimum_Distance="min",
          Average_Distance="mean",
          Maximum_Distance="max"
      )
      .reset_index()
      .sort_values("Maximum_Distance", ascending=False)
)

print("\nTop 10 Trains by Maximum Distance:")
print(train_distance_summary.head(10))

print("===== STEP 44: AVERAGE VS MAXIMUM DISTANCE =====")

top_10_train_summary = train_distance_summary.head(10)

labels = (
    top_10_train_summary["Train No"].astype(str)
    + " - "
    + top_10_train_summary["Train Name"]
)

x = range(len(labels))
width = 0.35

plt.figure(figsize=(14, 6))

plt.bar(
    [i - width / 2 for i in x],
    top_10_train_summary["Average_Distance"],
    width=width,
    label="Average Distance"
)

plt.bar(
    [i + width / 2 for i in x],
    top_10_train_summary["Maximum_Distance"],
    width=width,
    label="Maximum Distance"
)

plt.xlabel("Train")
plt.ylabel("Distance (km)")
plt.title("Average vs Maximum Distance of Top 10 Trains")

plt.xticks(x, labels, rotation=45, ha="right")
plt.legend()

plt.tight_layout()
plt.show()

print("\nStep 44 completed successfully.")

print("===== STEP 45: TRAIN ROUTE LENGTH ANALYSIS =====")

train_route_length = (
    df.groupby(["Train No", "Train Name"])["Distance"]
      .agg(
          Start_Distance="min",
          End_Distance="max"
      )
      .reset_index()
)

train_route_length["Route_Length"] = (
    train_route_length["End_Distance"]
    - train_route_length["Start_Distance"]
)

train_route_length = train_route_length.sort_values(
    "Route_Length",
    ascending=False
)

print("\nTop 10 Longest Train Routes:")
print(train_route_length.head(10))

print("\nRoute Length Statistics:")
print(train_route_length["Route_Length"].describe())

print("\nStep 45 completed successfully.")

print("===== STEP 46: ROUTE LENGTH DISTRIBUTION =====")

route_bins = [0, 50, 100, 250, 500, 1000, 2000, 3000, 5000]

route_labels = [
    "0-50 km",
    "51-100 km",
    "101-250 km",
    "251-500 km",
    "501-1000 km",
    "1001-2000 km",
    "2001-3000 km",
    "3001+ km"
]

train_route_length["Route_Category"] = pd.cut(
    train_route_length["Route_Length"],
    bins=route_bins,
    labels=route_labels,
    include_lowest=True
)

route_distribution = (
    train_route_length["Route_Category"]
    .value_counts()
    .reindex(route_labels)
    .fillna(0)
    .astype(int)
)

print("\nTrain Count by Route Length:")
print(route_distribution)

plt.figure(figsize=(12, 6))

plt.bar(
    route_distribution.index,
    route_distribution.values
)

plt.xlabel("Route Length")
plt.ylabel("Number of Trains")
plt.title("Distribution of Train Routes by Route Length")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nStep 46 completed successfully.")

print("===== STEP 47: ROUTE LENGTH PERCENTAGE ANALYSIS =====")

route_percentage = (
    train_route_length["Route_Category"]
    .value_counts()
    .reindex(route_labels)
    .fillna(0)
)

route_percentage = (
    route_percentage / route_percentage.sum() * 100
).round(2)

route_percentage_df = pd.DataFrame({
    "Train_Count": route_distribution,
    "Percentage": route_percentage
})

print("\nRoute Length Distribution with Percentage:")
print(route_percentage_df)

print("\nStep 47 completed successfully.")

print("===== STEP 48: TOP 10 LONGEST TRAIN ROUTES =====")

top_10_routes = train_route_length.head(10).copy()

top_10_routes["Train_Label"] = (
    top_10_routes["Train No"].astype(str)
    + " - "
    + top_10_routes["Train Name"]
)

plt.figure(figsize=(12, 7))

plt.barh(
    top_10_routes["Train_Label"],
    top_10_routes["Route_Length"]
)

plt.xlabel("Route Length (km)")
plt.ylabel("Train")
plt.title("Top 10 Longest Train Routes")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

print("\nTop 10 Longest Routes:")
print(
    top_10_routes[
        ["Train No", "Train Name", "Route_Length"]
    ]
)

print("\nStep 48 completed successfully.")

print("===== STEP 49: ROUTE LENGTH VS STATION RECORDS =====")

station_records = (
    df.groupby(["Train No", "Train Name"])
      .size()
      .reset_index(name="Station_Records")
)

route_station_analysis = train_route_length.merge(
    station_records,
    on=["Train No", "Train Name"],
    how="left"
)

route_station_analysis = route_station_analysis.sort_values(
    "Route_Length",
    ascending=False
)

print("\nTop 10 Longest Routes with Station Records:")

print(
    route_station_analysis[
        [
            "Train No",
            "Train Name",
            "Route_Length",
            "Station_Records"
        ]
    ].head(10)
)

print("\nStep 49 completed successfully.")

print("===== STEP 50: AVERAGE DISTANCE PER STATION =====")

route_station_analysis["Average_Distance_Per_Station"] = (
    route_station_analysis["Route_Length"]
    / route_station_analysis["Station_Records"]
)

route_station_analysis = route_station_analysis.sort_values(
    "Average_Distance_Per_Station",
    ascending=False
)

print("\nTop 10 Trains by Average Distance per Station:")

print(
    route_station_analysis[
        [
            "Train No",
            "Train Name",
            "Route_Length",
            "Station_Records",
            "Average_Distance_Per_Station"
        ]
    ].head(10)
)

print("\nStep 50 completed successfully.")

print("===== STEP 51: FILTERED DISTANCE PER STATION ANALYSIS =====")

filtered_route_analysis = route_station_analysis[
    route_station_analysis["Station_Records"] >= 10
].copy()

filtered_route_analysis = filtered_route_analysis.sort_values(
    "Average_Distance_Per_Station",
    ascending=False
)

print("\nTop 10 Trains with Minimum 10 Station Records:")

print(
    filtered_route_analysis[
        [
            "Train No",
            "Train Name",
            "Route_Length",
            "Station_Records",
            "Average_Distance_Per_Station"
        ]
    ].head(10)
)

print("\nFiltered Train Count:")
print(len(filtered_route_analysis))

print("\nStep 51 completed successfully.")

print("===== STEP 52: ROUTE LENGTH VS STATION RECORDS =====")

correlation = route_station_analysis[
    ["Route_Length", "Station_Records"]
].corr()

print("\nCorrelation Matrix:")
print(correlation)

print("\nRoute Length vs Station Records Correlation:")
print(
    round(
        correlation.loc["Route_Length", "Station_Records"],
        4
    )
)

print("\nStep 52 completed successfully.")

print("===== STEP 53: ROUTE LENGTH VS STATION RECORDS VISUALIZATION =====")

plt.figure(figsize=(12, 7))

plt.scatter(
    route_station_analysis["Route_Length"],
    route_station_analysis["Station_Records"],
    alpha=0.4
)

plt.xlabel("Route Length (km)")
plt.ylabel("Station Records")
plt.title("Route Length vs Station Records")

plt.tight_layout()
plt.show()

print("\nCorrelation:", round(
    route_station_analysis["Route_Length"]
    .corr(route_station_analysis["Station_Records"]),
    4
))

print("\nStep 53 completed successfully.")

print("===== STEP 54: ROUTE LENGTH CATEGORY VS AVERAGE STATION RECORDS =====")

route_station_analysis["Route_Length_Category"] = pd.cut(
    route_station_analysis["Route_Length"],
    bins=[0, 50, 100, 250, 500, 1000, 2000, 3000, float("inf")],
    labels=[
        "0-50 km",
        "51-100 km",
        "101-250 km",
        "251-500 km",
        "501-1000 km",
        "1001-2000 km",
        "2001-3000 km",
        "3001+ km"
    ]
)

category_station_analysis = (
    route_station_analysis
    .groupby("Route_Length_Category", observed=False)["Station_Records"]
    .agg(
        Average_Station_Records="mean",
        Minimum_Station_Records="min",
        Maximum_Station_Records="max"
    )
    .reset_index()
)

print("\nRoute Length Category vs Station Records:")
print(category_station_analysis)

print("\nStep 54 completed successfully.")

print("===== STEP 55: ROUTE LENGTH CATEGORY VS AVERAGE STATION RECORDS VISUALIZATION =====")

plt.figure(figsize=(12, 7))

plt.bar(
    category_station_analysis["Route_Length_Category"].astype(str),
    category_station_analysis["Average_Station_Records"]
)

plt.xlabel("Route Length Category")
plt.ylabel("Average Station Records")
plt.title("Route Length Category vs Average Station Records")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nStep 55 completed successfully.")

print("===== STEP 56: ROUTE LENGTH CATEGORY VS MAXIMUM STATION RECORDS =====")

plt.figure(figsize=(12, 7))

plt.bar(
    category_station_analysis["Route_Length_Category"].astype(str),
    category_station_analysis["Maximum_Station_Records"]
)

plt.xlabel("Route Length Category")
plt.ylabel("Maximum Station Records")
plt.title("Route Length Category vs Maximum Station Records")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nMaximum Station Records by Route Length Category:")
print(
    category_station_analysis[
        ["Route_Length_Category", "Maximum_Station_Records"]
    ]
)

print("\nStep 56 completed successfully.")

print("===== STEP 57: ROUTE LENGTH CATEGORY VS MINIMUM STATION RECORDS =====")

plt.figure(figsize=(12, 7))

plt.bar(
    category_station_analysis["Route_Length_Category"].astype(str),
    category_station_analysis["Minimum_Station_Records"]
)

plt.xlabel("Route Length Category")
plt.ylabel("Minimum Station Records")
plt.title("Route Length Category vs Minimum Station Records")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nMinimum Station Records by Route Length Category:")
print(
    category_station_analysis[
        ["Route_Length_Category", "Minimum_Station_Records"]
    ]
)

print("\nStep 57 completed successfully.")

print("===== STEP 58: ROUTE LENGTH CATEGORY STATION RECORDS SUMMARY =====")

route_length_summary = category_station_analysis[
    [
        "Route_Length_Category",
        "Average_Station_Records",
        "Minimum_Station_Records",
        "Maximum_Station_Records"
    ]
].copy()

print("\nRoute Length Category Station Records Summary:")
print(route_length_summary)

print("\nStep 58 completed successfully.")

print("===== STEP 59: STATION RECORDS RANGE BY ROUTE LENGTH CATEGORY =====")

category_station_analysis["Station_Records_Range"] = (
    category_station_analysis["Maximum_Station_Records"]
    - category_station_analysis["Minimum_Station_Records"]
)

print("\nStation Records Range by Route Length Category:")

print(
    category_station_analysis[
        [
            "Route_Length_Category",
            "Minimum_Station_Records",
            "Maximum_Station_Records",
            "Station_Records_Range"
        ]
    ]
)

print("\nStep 59 completed successfully.")

print("===== STEP 60: ROUTE LENGTH CATEGORY VS STATION RECORDS RANGE =====")

plt.figure(figsize=(12, 7))

plt.bar(
    category_station_analysis["Route_Length_Category"].astype(str),
    category_station_analysis["Station_Records_Range"]
)

plt.xlabel("Route Length Category")
plt.ylabel("Station Records Range")
plt.title("Route Length Category vs Station Records Range")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nStation Records Range:")
print(
    category_station_analysis[
        ["Route_Length_Category", "Station_Records_Range"]
    ]
)

print("\nStep 60 completed successfully.")

print("===== STEP 61: STATION RECORDS RANGE VISUALIZATION =====")

plt.figure(figsize=(12, 7))

plt.bar(
    category_station_analysis["Route_Length_Category"].astype(str),
    category_station_analysis["Station_Records_Range"]
)

plt.xlabel("Route Length Category")
plt.ylabel("Station Records Range")
plt.title("Station Records Range by Route Length Category")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nStep 61 completed successfully.")

print("===== STEP 62: SORTED AVERAGE STATION RECORDS BY ROUTE LENGTH =====")

sorted_average_analysis = (
    category_station_analysis[
        ["Route_Length_Category", "Average_Station_Records"]
    ]
    .sort_values(
        by="Average_Station_Records",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\nSorted Average Station Records:")

print(sorted_average_analysis)

print("\nStep 62 completed successfully.")

print("===== STEP 63: SORTED AVERAGE STATION RECORDS VISUALIZATION =====")

plt.figure(figsize=(12, 7))

plt.bar(
    sorted_average_analysis["Route_Length_Category"].astype(str),
    sorted_average_analysis["Average_Station_Records"]
)

plt.xlabel("Route Length Category")
plt.ylabel("Average Station Records")
plt.title("Average Station Records by Route Length Category")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nStep 63 completed successfully.")

print("===== STEP 64: ROUTE LENGTH VS AVERAGE STATION RECORDS CORRELATION =====")

category_route_midpoint = {
    "0-50 km": 25,
    "51-100 km": 75,
    "101-250 km": 175,
    "251-500 km": 375,
    "501-1000 km": 750,
    "1001-2000 km": 1500,
    "2001-3000 km": 2500,
    "3001+ km": 3500
}

category_station_analysis["Route_Length_Midpoint"] = (
    category_station_analysis["Route_Length_Category"]
    .astype(str)
    .map(category_route_midpoint)
)

category_correlation = (
    category_station_analysis["Route_Length_Midpoint"]
    .corr(category_station_analysis["Average_Station_Records"])
)

print("\nCategory-Level Correlation:")
print(round(category_correlation, 4))

print("\nStep 64 completed successfully.")

print("===== STEP 65: ROUTE LENGTH VS AVERAGE STATION RECORDS VISUALIZATION =====")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    category_station_analysis["Route_Length_Midpoint"],
    category_station_analysis["Average_Station_Records"],
    s=80
)

plt.xlabel("Route Length Midpoint (km)")
plt.ylabel("Average Station Records")
plt.title("Route Length vs Average Station Records")

plt.grid(True)
plt.show()

print("\nCategory-Level Correlation:", round(category_correlation, 4))
print("\nStep 65 completed successfully.")

print("===== STEP 66: CATEGORY-WISE ROUTE LENGTH & STATION RECORD SUMMARY =====")

step66_summary = category_station_analysis[
    [
        "Route_Length_Category",
        "Route_Length_Midpoint",
        "Average_Station_Records"
    ]
].copy()

print("\nCategory-wise Summary:")
print(step66_summary.to_string(index=False))

print("\nStep 66 completed successfully.")

print("===== STEP 67: CATEGORY-WISE AVERAGE STATION RECORDS VISUALIZATION =====")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.bar(
    step66_summary["Route_Length_Category"],
    step66_summary["Average_Station_Records"]
)

plt.xlabel("Route Length Category")
plt.ylabel("Average Station Records")
plt.title("Category-wise Average Station Records")

plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\nStep 67 completed successfully.")

print("===== STEP 68: SORTED CATEGORY-WISE AVERAGE STATION RECORDS =====")

step68_sorted = step66_summary.sort_values(
    by="Average_Station_Records",
    ascending=False
)

print("\nCategories Sorted by Average Station Records:")
print(step68_sorted.to_string(index=False))

print("\nStep 68 completed successfully.")

print("===== STEP 69: CATEGORY-WISE STATION RECORD CONTRIBUTION =====")

step69_summary = step68_sorted.copy()

total_average_records = step69_summary["Average_Station_Records"].sum()

step69_summary["Record_Contribution_Percentage"] = (
    step69_summary["Average_Station_Records"]
    / total_average_records
) * 100

print("\nCategory-wise Record Contribution:")
print(
    step69_summary[
        [
            "Route_Length_Category",
            "Average_Station_Records",
            "Record_Contribution_Percentage"
        ]
    ].to_string(index=False)
)

print("\nTotal Contribution: {:.2f}%".format(
    step69_summary["Record_Contribution_Percentage"].sum()
))

print("\nStep 69 completed successfully.")

print("===== STEP 70: CATEGORY-WISE STATION RECORD CONTRIBUTION VISUALIZATION =====")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.bar(
    step69_summary["Route_Length_Category"],
    step69_summary["Record_Contribution_Percentage"]
)

plt.xlabel("Route Length Category")
plt.ylabel("Record Contribution (%)")
plt.title("Category-Wise Station Record Contribution")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nStep 70 completed successfully.")

print("===== STEP 71: CATEGORY-WISE CONTRIBUTION SUMMARY =====")

step71_summary = step69_summary[
    [
        "Route_Length_Category",
        "Average_Station_Records",
        "Record_Contribution_Percentage"
    ]
].copy()

step71_summary["Record_Contribution_Percentage"] = (
    step71_summary["Record_Contribution_Percentage"].round(2)
)

print("\nCategory-wise Contribution Summary:")
print(step71_summary.to_string(index=False))

print("\nHighest Contribution Category:")
print(
    step71_summary.loc[
        step71_summary["Record_Contribution_Percentage"].idxmax(),
        "Route_Length_Category"
    ]
)

print("\nLowest Contribution Category:")
print(
    step71_summary.loc[
        step71_summary["Record_Contribution_Percentage"].idxmin(),
        "Route_Length_Category"
    ]
)

print("\nStep 71 completed successfully.")

print("===== STEP 72: CUMULATIVE CATEGORY-WISE CONTRIBUTION =====")

step72_summary = step71_summary.copy()

step72_summary["Cumulative_Contribution_Percentage"] = (
    step72_summary["Record_Contribution_Percentage"].cumsum()
)

print("\nCumulative Category-wise Contribution:")
print(
    step72_summary[
        [
            "Route_Length_Category",
            "Record_Contribution_Percentage",
            "Cumulative_Contribution_Percentage"
        ]
    ].to_string(index=False)
)

print("\nFinal Cumulative Contribution: {:.2f}%".format(
    step72_summary["Cumulative_Contribution_Percentage"].iloc[-1]
))

print("\nStep 72 completed successfully.")

print("===== STEP 73: CUMULATIVE CONTRIBUTION VISUALIZATION =====")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    step72_summary["Route_Length_Category"],
    step72_summary["Cumulative_Contribution_Percentage"],
    marker="o"
)

plt.xlabel("Route Length Category")
plt.ylabel("Cumulative Contribution (%)")
plt.title("Cumulative Category-Wise Station Record Contribution")

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nStep 73 completed successfully.")

print("===== STEP 74: CONTRIBUTION THRESHOLD ANALYSIS =====")

thresholds = [50, 75, 90]

for threshold in thresholds:
    threshold_row = step72_summary[
        step72_summary["Cumulative_Contribution_Percentage"] >= threshold
    ].iloc[0]

    print(
        f"\nCategory reaching {threshold}% cumulative contribution:"
    )
    print(
        f"{threshold_row['Route_Length_Category']} "
        f"({threshold_row['Cumulative_Contribution_Percentage']:.2f}%)"
    )

print("\nStep 74 completed successfully.")

print("===== STEP 75: TOP CATEGORY CONTRIBUTION SUMMARY =====")

top_3 = step72_summary.iloc[:3]["Record_Contribution_Percentage"].sum()
top_5 = step72_summary.iloc[:5]["Record_Contribution_Percentage"].sum()
top_6 = step72_summary.iloc[:6]["Record_Contribution_Percentage"].sum()

print("\nTop 3 Categories Contribution: {:.2f}%".format(top_3))
print("Top 5 Categories Contribution: {:.2f}%".format(top_5))
print("Top 6 Categories Contribution: {:.2f}%".format(top_6))

print("\nStep 75 completed successfully.")

print("===== STEP 76: TOP CATEGORY CONTRIBUTION VISUALIZATION =====")

import matplotlib.pyplot as plt

top_categories = step72_summary.iloc[:6]

plt.figure(figsize=(10, 6))

plt.bar(
    top_categories["Route_Length_Category"],
    top_categories["Record_Contribution_Percentage"]
)

plt.xlabel("Route Length Category")
plt.ylabel("Record Contribution (%)")
plt.title("Top 6 Route Length Categories - Record Contribution")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nStep 76 completed successfully.")

print("===== STEP 77: TOP 6 VS REMAINING CATEGORIES =====")

top_6_contribution = step72_summary.iloc[:6]["Record_Contribution_Percentage"].sum()
remaining_contribution = step72_summary.iloc[6:]["Record_Contribution_Percentage"].sum()

step77_summary = pd.DataFrame({
    "Group": ["Top 6 Categories", "Remaining Categories"],
    "Contribution_Percentage": [
        top_6_contribution,
        remaining_contribution
    ]
})

print("\nContribution Comparison:")
print(step77_summary.to_string(index=False))

print("\nTotal Contribution: {:.2f}%".format(
    step77_summary["Contribution_Percentage"].sum()
))

print("\nStep 77 completed successfully.")

print("===== STEP 78: TOP 6 VS REMAINING CONTRIBUTION VISUALIZATION =====")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.bar(
    step77_summary["Group"],
    step77_summary["Contribution_Percentage"]
)

plt.xlabel("Category Group")
plt.ylabel("Contribution (%)")
plt.title("Top 6 vs Remaining Categories Contribution")

plt.tight_layout()
plt.show()

print("\nStep 78 completed successfully.")

print("===== STEP 79: CATEGORY CONTRIBUTION STATISTICS =====")

contribution_stats = step69_summary["Record_Contribution_Percentage"].describe()

print("\nContribution Percentage Statistics:")
print(contribution_stats.to_string())

print("\nMaximum Contribution: {:.2f}%".format(
    step69_summary["Record_Contribution_Percentage"].max()
))

print("Minimum Contribution: {:.2f}%".format(
    step69_summary["Record_Contribution_Percentage"].min()
))

print("Average Contribution: {:.2f}%".format(
    step69_summary["Record_Contribution_Percentage"].mean()
))

print("\nStep 79 completed successfully.")

print("===== STEP 80: CATEGORY CONTRIBUTION RANGE =====")

max_contribution = step69_summary["Record_Contribution_Percentage"].max()
min_contribution = step69_summary["Record_Contribution_Percentage"].min()

contribution_range = max_contribution - min_contribution

print("\nMaximum Contribution: {:.2f}%".format(max_contribution))
print("Minimum Contribution: {:.2f}%".format(min_contribution))
print("Contribution Range: {:.2f}%".format(contribution_range))

print("\nStep 80 completed successfully.")

print("===== STEP 81: CATEGORY CONTRIBUTION VISUALIZATION =====")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.bar(
    step69_summary["Route_Length_Category"],
    step69_summary["Record_Contribution_Percentage"]
)

plt.xlabel("Route Length Category")
plt.ylabel("Record Contribution Percentage (%)")
plt.title("Category-wise Record Contribution Percentage")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nStep 81 completed successfully.")

print("===== STEP 82: SORTED CATEGORY CONTRIBUTION =====")

step82_summary = step69_summary[
    [
        "Route_Length_Category",
        "Record_Contribution_Percentage"
    ]
].copy()

step82_summary = step82_summary.sort_values(
    by="Record_Contribution_Percentage",
    ascending=False
).reset_index(drop=True)

print("\nCategories Sorted by Record Contribution:")

print(
    step82_summary.to_string(index=False)
)

print("\nStep 82 completed successfully.")

print("===== STEP 83: CUMULATIVE CATEGORY CONTRIBUTION =====")

step83_summary = step82_summary.copy()

step83_summary["Cumulative_Contribution_Percentage"] = (
    step83_summary["Record_Contribution_Percentage"].cumsum()
)

print("\nCumulative Category Contribution:")

print(
    step83_summary[
        [
            "Route_Length_Category",
            "Record_Contribution_Percentage",
            "Cumulative_Contribution_Percentage"
        ]
    ].to_string(index=False)
)

print(
    "\nFinal Cumulative Contribution: {:.2f}%".format(
        step83_summary["Cumulative_Contribution_Percentage"].iloc[-1]
    )
)

print("\nStep 83 completed successfully.")

# ===== STEP 84: CUMULATIVE CATEGORY CONTRIBUTION VISUALIZATION =====

import matplotlib.pyplot as plt

print("===== STEP 84: CUMULATIVE CATEGORY CONTRIBUTION VISUALIZATION =====")

# Recreate cumulative contribution data from Step 69 summary
step84_data = step69_summary[
    [
        "Route_Length_Category",
        "Record_Contribution_Percentage"
    ]
].copy()

# Sort by contribution from highest to lowest
step84_data = step84_data.sort_values(
    by="Record_Contribution_Percentage",
    ascending=False
).reset_index(drop=True)

# Calculate cumulative contribution
step84_data["Cumulative_Contribution_Percentage"] = (
    step84_data["Record_Contribution_Percentage"].cumsum()
)

# Visualization
plt.figure(figsize=(12, 6))

plt.plot(
    step84_data["Route_Length_Category"],
    step84_data["Cumulative_Contribution_Percentage"],
    marker="o"
)

plt.title("Cumulative Category Contribution")
plt.xlabel("Route Length Category")
plt.ylabel("Cumulative Contribution (%)")
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nFinal Cumulative Contribution:",
      f"{step84_data['Cumulative_Contribution_Percentage'].iloc[-1]:.2f}%")

print("\nStep 84 completed successfully.")

# ===== STEP 85: TOP CONTRIBUTING CATEGORY =====

print("===== STEP 85: TOP CONTRIBUTING CATEGORY =====")

top_category = step84_data.loc[
    step84_data["Record_Contribution_Percentage"].idxmax()
]

print("\nTop Contributing Route Length Category:")
print("Category:", top_category["Route_Length_Category"])
print(
    "Record Contribution:",
    f"{top_category['Record_Contribution_Percentage']:.2f}%"
)

print("\nStep 85 completed successfully.")

# ===== STEP 86: CATEGORY CONTRIBUTION COMPARISON =====

import matplotlib.pyplot as plt

print("===== STEP 86: CATEGORY CONTRIBUTION COMPARISON =====")

plt.figure(figsize=(12, 6))

plt.bar(
    step84_data["Route_Length_Category"],
    step84_data["Record_Contribution_Percentage"]
)

plt.title("Route Length Category Contribution Comparison")
plt.xlabel("Route Length Category")
plt.ylabel("Record Contribution (%)")

plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

print("\nStep 86 completed successfully.")

# ===== STEP 87: POWER BI DATA VALIDATION =====

print("===== STEP 87: POWER BI DATA VALIDATION =====")

# Power BI CSV export
powerbi_file = "Indian_Railway_PowerBI_Data.csv"
df.to_csv(powerbi_file, index=False)

print(f"\nPower BI dataset exported successfully: {powerbi_file}")

# Read exported Power BI file
powerbi_df = pd.read_csv(powerbi_file)

print("\nPower BI Dataset Validation:")
print(f"Total Rows: {len(powerbi_df)}")
print(f"Total Columns: {len(powerbi_df.columns)}")

print("\nColumn Names:")
print(powerbi_df.columns.tolist())

print("\nMissing Values:")
print(powerbi_df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(powerbi_df.duplicated().sum())

print("\nStep 87 completed successfully.")

# ===== STEP 88: POWER BI MISSING VALUE ANALYSIS =====

print("===== STEP 88: POWER BI MISSING VALUE ANALYSIS =====")

missing_values = powerbi_df.isnull().sum()

missing_values = missing_values[missing_values > 0].sort_values(
    ascending=False
)

print("\nColumns With Missing Values:")
print(missing_values)

print("\nTotal Columns With Missing Values:")
print(len(missing_values))

print("\nStep 88 completed successfully.")

# ===== STEP 89: HANDLE MISSING HALT TIME =====

print("===== STEP 89: HANDLE MISSING HALT TIME =====")

missing_before = powerbi_df["Halt Time Cleaned"].isnull().sum()

print(f"\nMissing Halt Time Before Cleaning: {missing_before}")

# Fill missing Halt Time Cleaned with 0
powerbi_df["Halt Time Cleaned"] = powerbi_df["Halt Time Cleaned"].fillna(0)

missing_after = powerbi_df["Halt Time Cleaned"].isnull().sum()

print(f"Missing Halt Time After Cleaning: {missing_after}")

# Save cleaned Power BI dataset
powerbi_df.to_csv(powerbi_file, index=False)

print(f"\nCleaned Power BI dataset saved: {powerbi_file}")

print("\nStep 89 completed successfully.")

# ===== STEP 90: FINAL POWER BI DATA VALIDATION =====

print("===== STEP 90: FINAL POWER BI DATA VALIDATION =====")

# Reload final Power BI dataset
final_powerbi_df = pd.read_csv(powerbi_file)

print("\nFinal Dataset:")
print(f"Total Rows: {len(final_powerbi_df)}")
print(f"Total Columns: {len(final_powerbi_df.columns)}")

# Missing values
total_missing = final_powerbi_df.isnull().sum().sum()

print(f"\nTotal Missing Values: {total_missing}")

# Duplicate rows
total_duplicates = final_powerbi_df.duplicated().sum()

print(f"Total Duplicate Rows: {total_duplicates}")

# Original vs Power BI row count
print(f"\nOriginal Dataset Rows: {len(df)}")
print(f"Power BI Dataset Rows: {len(final_powerbi_df)}")

# Validation status
if (
    total_missing == 0
    and total_duplicates == 0
    and len(df) == len(final_powerbi_df)
):
    print("\nPOWER BI DATA VALIDATION: PASSED")
else:
    print("\nPOWER BI DATA VALIDATION: CHECK REQUIRED")

print("\nStep 90 completed successfully.")

# ===== STEP 91: POWER BI READY DATASET SUMMARY =====

print("===== STEP 91: POWER BI READY DATASET SUMMARY =====")

print("\nPower BI Dataset Information:")
print(f"Rows: {len(final_powerbi_df)}")
print(f"Columns: {len(final_powerbi_df.columns)}")

print("\nPower BI Columns:")
for i, column in enumerate(final_powerbi_df.columns, start=1):
    print(f"{i}. {column}")

print("\nData Types:")
print(final_powerbi_df.dtypes)

print("\nFinal Power BI File:")
print(powerbi_file)

print("\nPower BI dataset is ready for dashboard development.")

print("\nStep 91 completed successfully.")

# ===== STEP 92: POWER BI FILE VERIFICATION =====

import os

print("===== STEP 92: POWER BI FILE VERIFICATION =====")

file_exists = os.path.exists(powerbi_file)

print(f"\nPower BI File Exists: {file_exists}")

if file_exists:
    file_size_mb = os.path.getsize(powerbi_file) / (1024 * 1024)

    print(f"File Name: {powerbi_file}")
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"File Location: {os.path.abspath(powerbi_file)}")

print("\nPower BI import file verification completed.")

print("\nStep 92 completed successfully.")
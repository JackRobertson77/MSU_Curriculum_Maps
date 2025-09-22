import pandas as pd

#basic cleaning for now
def clean_cns_data(path: str) -> pd.DataFrame:

    df = pd.read_excel(path)

    #drop redundant/unused columns
    drop_cols = ["Tracking Major Title", "Description", "Eff End Term"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    #remove extra spaces
    if "Course" in df.columns:
        df["Course"] = df["Course"].astype(str).str.replace(" ", "").str.upper()

    #categorical data 
    if "Year" in df.columns:
        year_map = {
            "Freshman": 1,
            "Sophomore": 2,
            "Junior": 3,
            "Senior": 4,
            "Fifth Year": 5
        }
        df["Year"] = df["Year"].replace(year_map)

    #make sure credit columns are numeric
    for col in ["Credits", "Credits Min", "Credits Max"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    #removing prerequisites in parentheses
    if "Plan Title" in df.columns:
        df["Plan Title"] = df["Plan Title"].str.replace(r"\s*\(.*\)", "", regex=True)

    return df

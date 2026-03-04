"""
Weather Data Crawler

Downloads weather data from Oikolab API and saves to CSV format.
[Oikolab API](https://docs.oikolab.com/)

Usage:
    python oikolab.py --location "Shanghai" --start 2024-01-01 --end 2024-12-31
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import io
import time

# ===============================================================
# API Configuration
API_URL = 'https://api.oikolab.com/weather'
API_KEY = ''  # Your API key
# ===============================================================

# Required parameters for EnergyPlus simulation
REQUIRED_PARAMETERS = [
    # Temperature & Humidity (Core)
    'temperature',                      # Dry Bulb Temperature [°C]
    'dewpoint_temperature',             # Dew Point Temperature [°C]
    'relative_humidity',                # Relative Humidity [%]
    
    # Pressure
    'surface_pressure',                 # Atmospheric Pressure [Pa]
    
    # Solar Radiation (Critical for PV simulation)
    'direct_normal_solar_radiation',    # DNI [W/m²]
    'surface_diffuse_solar_radiation',  # DHI [W/m²]
    'surface_solar_radiation',          # GHI [W/m²] - for validation
    'surface_thermal_radiation',        # Infrared [W/m²] - for sky temperature
    'surface_direct_solar_radiation',   # Direct Horizontal Irradiance [W/m²]
    
    # Wind (10m height, EnergyPlus will adjust)
    'wind_speed',                       # Wind Speed [m/s]
    'wind_direction',                   # Wind Direction [degrees]
    
    # Other
    'total_cloud_cover',                # Cloud Cover [fraction 0-1]
    'total_precipitation',              # Precipitation [mm]
    'snowfall',                         # Snowfall [mm]
]


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _month_ranges(start_date: str, end_date: str):
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    cur = datetime(start_dt.year, start_dt.month, 1)
    while cur <= end_dt:
        if cur.month == 12:
            next_month = datetime(cur.year + 1, 1, 1)
        else:
            next_month = datetime(cur.year, cur.month + 1, 1)

        chunk_start = max(start_dt, cur)
        chunk_end = min(end_dt, next_month - pd.Timedelta(days=1))
        yield chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = next_month


def _location_slug(location: str) -> str:
    slug = location.strip().lower()
    for ch in [",", "/", "\\", ":", " "]:
        slug = slug.replace(ch, "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def download_weather_data(
    location: str,
    start_date: str,
    end_date: str,
    output_path: str = None,
    model: str = "era5",
    max_retries: int = 3,
    retry_sleep_s: float = 2.0,
) -> pd.DataFrame:
    """
    Download weather data from Oikolab API.
    
    Args:
        location: Location string (e.g., "Shanghai, China")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Optional path to save CSV file
        model: Weather model/dataset (default: era5)
        max_retries: Maximum number of retries (default: 3)
        retry_sleep_s: Sleep time between retries (default: 2.0)
    
    Returns:
        DataFrame with weather data
    """
    print(f"Downloading weather data for {location}...")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Parameters: {len(REQUIRED_PARAMETERS)} variables")
    
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                API_URL,
                params={
                    'param': REQUIRED_PARAMETERS,
                    'location': location,
                    'start': start_date,
                    'end': end_date,
                    'freq': 'H',  # Hourly data
                    'resample_method': 'mean',
                    'format': 'csv',
                    'model': model,
                },
                headers={'api-key': API_KEY},
                timeout=60,
            )

            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code}\n{response.text}")

            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_sleep_s * attempt)
            else:
                raise
    
    # Parse CSV response
    df = pd.read_csv(io.StringIO(response.text))
    
    # Normalize column names: "temperature (degC)" -> "temperature"
    df = normalize_column_names(df)
    
    print(f"  Downloaded {len(df)} hourly records")
    print(f"  Columns: {list(df.columns)}")
    
    # Save to file if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
    
    return df


def download_weather_data_chunked(
    location: str,
    start_date: str,
    end_date: str,
    out_dir: str,
    model: str = "era5",
) -> pd.DataFrame:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    parts = []
    for chunk_start, chunk_end in _month_ranges(start_date=start_date, end_date=end_date):
        slug = _location_slug(location)
        out_file = out_dir_path / f"{slug}_{chunk_start}_{chunk_end}_{model}.csv"
        df_part = download_weather_data(
            location=location,
            start_date=chunk_start,
            end_date=chunk_end,
            output_path=str(out_file),
            model=model,
        )
        parts.append(df_part)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, axis=0, ignore_index=True)
    time_col = _find_time_column(df)
    if time_col:
        df = df.drop_duplicates(subset=[time_col]).reset_index(drop=True)
    else:
        df = df.drop_duplicates().reset_index(drop=True)

    merged_file = out_dir_path / f"{_location_slug(location)}_{start_date}_{end_date}_{model}_merged.csv"
    df.to_csv(merged_file, index=False)
    print(f"  Saved merged to: {merged_file}")
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Oikolab column names by removing unit suffixes.
    
    Example: "temperature (degC)" -> "temperature"
    """
    rename_map = {}
    for col in df.columns:
        # Extract base name before the unit in parentheses
        if ' (' in col:
            base_name = col.split(' (')[0]
            rename_map[col] = base_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


def _find_time_column(df: pd.DataFrame) -> str:
    """Find the datetime column in DataFrame."""
    for col in ['datetime', 'time', 'timestamp', 'date']:
        if col in df.columns:
            return col
    return None


def validate_data(df: pd.DataFrame) -> dict:
    """Validate downloaded weather data."""
    errors = []
    warnings = []
    
    # Check for required columns
    missing = [p for p in REQUIRED_PARAMETERS if p not in df.columns]
    if missing:
        errors.append(f"Missing required parameters: {missing}")
    
    # Check for missing values
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            warnings.append(f"Column '{col}' has {null_count} missing values")
    
    # Check time column
    time_col = _find_time_column(df)
    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            errors.append(f"Failed to parse time column: {e}")
    
    return {'errors': errors, 'warnings': warnings}


def print_data_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of downloaded data."""
    print("\n📊 DATA SUMMARY:")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    
    time_col = _find_time_column(df)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        print(f"  Time range: {df[time_col].min()} to {df[time_col].max()}")
    
    # Print basic stats for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        print("\n  Numeric column statistics:")
        for col in numeric_cols[:5]:  # Show first 5
            print(f"    {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Download weather data from Oikolab for EnergyPlus simulation"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="Beijing, China",
        help="Location string (default: Beijing, China)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Start date YYYY-MM-DD (default: 2024-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="End date YYYY-MM-DD (default: 2025-12-31)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="era5",
        help="Weather model/dataset (default: era5)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/raw/weather/oikolab",
        help="Output directory for chunked CSVs and merged CSV",
    )
    args = parser.parse_args()
    
    df = download_weather_data_chunked(
        location=args.location,
        start_date=args.start,
        end_date=args.end,
        out_dir=args.outdir,
        model=args.model,
    )
    
    # Validate
    validation = validate_data(df)
    if validation['errors']:
        print("\n⚠️ VALIDATION ERRORS:")
        for err in validation['errors']:
            print(f"  - {err}")
    if validation['warnings']:
        print("\n⚠️ WARNINGS:")
        for warn in validation['warnings']:
            print(f"  - {warn}")
    
    # Print summary
    print_data_summary(df)
    
    print("\n✅ Download complete!")
    print(f"Next step: Convert to EPW format using epw_converter.py")


if __name__ == "__main__":

    main()
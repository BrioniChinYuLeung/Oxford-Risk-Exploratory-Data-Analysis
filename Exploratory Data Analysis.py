# Imports
import requests
import datetime
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from matplotlib.ticker import FuncFormatter, LogLocator
import matplotlib.dates as mdates
import time

# --- 1. Financial Assets Data Analysis ---
assets_df = pd.read_csv('/Users/leungchinyu/Desktop/Financial Assets data.csv')

# 1.1 Initial Data Inspection
print("Financial Assets Data")
print("\nFirst 5 rows:")
print(assets_df.head())
print("-" * 50)

print("\nLast 5 rows:")
print(assets_df.tail())
print("-" * 50)

print("\nDataframe Information:")
print(assets_df.info())  # No missing values
print("-" * 50)

print(f'Number of duplicated rows: {assets_df.duplicated().sum()}')  # No duplicated rows
print("-" * 50)

# 1.2 Currency Conversion
# Using Freecurrencyapi (Historical Rates)
API_KEY = 'fca_live_KuFS9J5eRsNoSTmaI5FCsHhCbDZLQ4tJ6OvvYUbk'
url = "https://api.freecurrencyapi.com/v1/historical"

# Setting GBP as base currency
TARGET_BASE_CURRENCY = 'GBP'

# Caching Mechanism
exchange_rate_cache = {}
CACHE_FILE = 'freecurrencyapi_exchange_rates_cache.json'

def load_cache(): #Loading cached exchange rates from a JSON file
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                loaded_cache = json.load(f)
                # Convert string keys from JSON back to tuples
                return {
                    tuple(k.split(',')): v
                    for k, v in loaded_cache.items()
                }
            except json.JSONDecodeError:
                print(
                    "Warning: Could not decode cache file. Starting with empty cache."
                )
                return {}
    return {}

def save_cache(): #Saving the current exchange rate cache to a JSON file
    stringified_cache = {
        ','.join(k): v
        for k, v in exchange_rate_cache.items()
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(stringified_cache, f)
    print(f"Cache saved to {CACHE_FILE}") # Converting tuple keys to strings for JSON serialization

exchange_rate_cache = load_cache() # Load cache when script starts
print(f"Loaded {len(exchange_rate_cache)} items from cache.")

def get_exchange_rate_freecurrencyapi(date_str,
                                      from_currency,
                                      to_currency=TARGET_BASE_CURRENCY):
    if from_currency == to_currency:
        return 1.0, False  # Return 1.0 and False (no API call needed)

    cache_key = (date_str, from_currency, to_currency)
    if cache_key in exchange_rate_cache:
        return exchange_rate_cache[
            cache_key], False  # Return from cache, no API call

# Parameters
    params = {
        'apikey': API_KEY,
        'date': date_str,
        'base_currency': to_currency,
        'currencies': from_currency
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get('data'):
            if date_str in data['data'] and from_currency in data['data'][
                    date_str]:
                rate_gbp_to_from = data['data'][date_str][from_currency]
                if rate_gbp_to_from != 0:
                    rate = 1 / rate_gbp_to_from
                    exchange_rate_cache[cache_key] = rate
                    return rate, True  # Rate fetched via API, so True
                else:
                    print(
                        f"Warning: Rate for {from_currency} to {to_currency} is zero on {date_str}. Returning None."
                    )
                    return None, False
            else:
                print(
                    f"Error: Rate for {from_currency} not found in API response for {date_str} (or date not found). Response: {data}"
                )
                return None, False
        else:
            print(
                f"API Error for {from_currency} to {to_currency} on {date_str}: {data.get('message', 'Unknown API Error')}. Response: {data}"
            )
            return None, False
    except requests.exceptions.RequestError as e:
        print(
            f"Network or API request error for {from_currency} to {to_currency} on {date_str}: {e}"
        )
        return None, False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, False


assets_df['created'] = pd.to_datetime(assets_df['created'])

# Minimising API calls and getting unique date-currency pairs first
unique_date_currency_pairs = assets_df[[
    'created', 'asset_currency'
]].drop_duplicates().sort_values('created')


# Fetching rates for all unique pairs
fetched_rates = {}
num_api_calls_made = 0
for idx, row in unique_date_currency_pairs.iterrows():
    date_str = row['created'].strftime('%Y-%m-%d')
    from_currency = row['asset_currency']

    # Call the modified function
    rate, api_called = get_exchange_rate_freecurrencyapi(
        date_str, from_currency, TARGET_BASE_CURRENCY)

    if rate is not None:
        fetched_rates[(date_str, from_currency)] = rate

    if api_called:
        num_api_calls_made += 1
        print(
            f"API call made for {from_currency} on {date_str}. Waiting {6.1} seconds. Total API calls this run: {num_api_calls_made}"
        )
        time.sleep(6.1)
    else:
        pass  # No sleep for cache hits
print(
    f"Fetched {len(fetched_rates)} rates (including cached). Total actual API calls made: {num_api_calls_made}."
)
save_cache()  # Saving cache after all potential fetches


# Applying the fetched rates to the DataFrame
def convert_value_with_fetched_rates(row):
    date_str = row.name.strftime('%Y-%m-%d')
    from_currency = row['asset_currency']
    original_value = row['asset_value']

    rate = fetched_rates.get((date_str, from_currency))

    if rate is not None:
        return original_value * rate
    else:
        return np.nan

assets_df = assets_df.set_index('created')

assets_df['asset_value_gbp'] = assets_df.apply(
    convert_value_with_fetched_rates, axis=1)

print("\nDataFrame with API-converted 'asset_value_gbp' column (first 5 rows):")
print(assets_df[['asset_value', 'asset_currency', 'asset_value_gbp']].head())
print("-" * 50)


# 1.3 Univariate analysis
# ID
# Count, max, min, mean, asset count
print(f"Number of unique IDs: {assets_df['_id'].nunique()}")
id_frequency = assets_df['_id'].value_counts()
print(f"Maximum frequency of asset allocation: {id_frequency.max()}")
print(f"Minimum frequency of asset allocation: {id_frequency.min()}")
print(f"Mean frequency of asset allocation: {id_frequency.mean():.2f}")
print(
    f"Number of unique allocation IDs: {assets_df['asset_allocation_id'].nunique()}"
)
print("-" * 50)

# Asset allocation type
# Counts
assets_allocation_frequency = assets_df['asset_allocation'].value_counts()
print(f"Counts of each asset allocation type: {assets_allocation_frequency}")

# Percentages
asset_allocation_percentage = assets_df['asset_allocation'].value_counts(
    normalize=True) * 100
print(
    f"Percentages of each asset allocation type: {asset_allocation_percentage.map('{:.2f}'.format)}"
)

# Pie chart
labels = assets_allocation_frequency.index
sizes = assets_allocation_frequency.values

plt.figure(figsize=(8, 8))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.2f%%',
    startangle=90,
    pctdistance=0.85
)
plt.title('Distribution of Asset Allocation Types')
plt.axis('equal')
print("-" * 50)

# Asset currency
# Counts
currency_frequency = assets_df['asset_currency'].value_counts()
print(f"Counts of each asset currency: {currency_frequency}")

# Percentages
currency_percentage = assets_df['asset_currency'].value_counts(
    normalize=True) * 100
print(
    f"Percentages of each asset currency: {currency_percentage.map('{:.2f}'.format)}"
)

# Bar chart
plt.figure(figsize=(10, 6))
currency_frequency.plot(kind='bar', color='skyblue')

plt.title('Frequency of Asset Currencies')
plt.xlabel('Asset Currency')
plt.ylabel('Number of Assets')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
print("-" * 50)

# Asset value
# Mean, std, min, IQR, max
assets_value_summary = assets_df['asset_value_gbp'].describe()
print(
    f"Asset value summary statistics (GBP): {assets_value_summary.map('{:.2f}'.format)}"
)

# Histogram
plt.figure(figsize=(10, 6)) 
plt.hist(assets_df['asset_value_gbp'], bins=20, edgecolor='black',
         alpha=0.7)

plt.title('Distribution of Asset Value')
plt.xlabel('Asset Value (GBP)')
plt.ylabel('Frequency (Number of Assets)')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()

# Box plot
plt.figure(figsize=(10, 4))
sns.boxplot(x=assets_df['asset_value_gbp'], color='lightcoral')
plt.title('Box Plot of Asset Value (Revealing Skew and Outliers)')
plt.xlabel('Asset Value (GBP)')

# Violin plot
plt.figure(figsize=(10, 4))
sns.violinplot(x=assets_df['asset_value_gbp'], color='lightblue')
plt.title('Violin Plot of Asset Value (Revealing Skew and Density)')
plt.xlabel('Asset Value (GBP)')
print("-" * 50)

# Time (Week needed)
# Period covered
period_start = assets_df.index.min()
period_end = assets_df.index.max()
print(f'Period covered: {period_start} to {period_end}')

# Daily counts
date_counts = pd.Series(assets_df.index.date).value_counts().sort_index()
date_df = date_counts.reset_index()
date_df.columns = ['Date', 'Frequency']
date_df['Date'] = pd.to_datetime(date_df['Date'])

print(f'Minimum total asset allocations made per day: {date_counts.min():.2f}')
print(f'Maximum total asset allocations made per day: {date_counts.max():.2f}')
print(f'Mean total asset allocations made per day: {date_counts.mean():.2f}')

# Line plot (daily)
plt.figure(figsize=(15, 7))
ax = sns.lineplot(x='Date',
                  y='Frequency',
                  data=date_df,
                  marker='o',
                  markersize=4,
                  linestyle='-',
                  color='orange',
                  alpha=0.7)
plt.title('Daily Frequency of Asset Allocations')
plt.xlabel('Date')
plt.ylabel('Number of Asset Allocations')
plt.xticks(rotation=45, ha='right')
plt.grid(True, which="both", ls="-", alpha=0.6)
plt.tight_layout()

# Weekly counts
week_counts = assets_df.index.isocalendar().week.value_counts().sort_index()
week_df = week_counts.reset_index()
week_df.columns = ['Week_Start_Date', 'Frequency']

print(f'Minimum total asset allocations made per week: {week_counts.min():.2f}')
print(f'Maximum total asset allocations made per week: {week_counts.max():.2f}')
print(f'Mean total asset allocations made per week: {week_counts.mean():.2f}')


# Monthly counts
month_counts = assets_df.index.month.value_counts().sort_index()
print(f'Minimum total asset allocations made per month: {month_counts.min():.2f}')
print(f'Maximum total asset allocations made per month: {month_counts.max():.2f}')
print(
    f'Mean total asset allocations made per month: {month_counts.mean():.2f}')

month_df = month_counts.reset_index()
month_df.columns = ['Month', 'Frequency']

month_name_to_num = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}
month_df['Month_Num_Sort'] = month_df['Month'].map(month_name_to_num)

month_df = month_df.sort_values(by='Month_Num_Sort').drop(
    columns='Month_Num_Sort')

# Bar chart (monthly)
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 7)

plt.figure(figsize=(12, 7))
sns.barplot(x='Month', y='Frequency', data=month_df)

plt.title('Frequency of Assets Allocations by Month')
plt.xlabel('Month')
plt.ylabel('Number of Asset Allocations')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
print("-" * 50)


# 1.4 Multivariate analysis
# Asset value according to asset allocation type
asset_value_gbp_by_type = assets_df.groupby(
    'asset_allocation')['asset_value_gbp'].agg(
        ['sum', 'mean', 'median', 'min', 'max'])
asset_value_gbp_by_type.columns = [
    'Total Value', 'Average Value', 'Median Value', 'Min Value', 'Max Value'
]
print("\nSummary statistics of asset values (GBP) by allocation type:")
print(asset_value_gbp_by_type.map('{:.2f}'.format))

# Bar chart
fig1, ax2 = plt.subplots(figsize=(10, 6))

sns.barplot(x=asset_value_gbp_by_type.index,
            y='Total Value',
            data=asset_value_gbp_by_type,
            ax=ax2)

ax2.set_title('Total Asset Value by Allocation Type')
ax2.set_xlabel('Asset Allocation Type')
ax2.set_ylabel('Total Value (GBP)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
fig1.tight_layout()
print("-" * 50)

# Asset value according to asset currency
asset_value_gbp_by_currency = assets_df.groupby(
    'asset_currency')['asset_value_gbp'].agg(
        ['sum', 'mean', 'median', 'min', 'max'])
asset_value_gbp_by_currency.columns = [
    'Total Value', 'Average Value', 'Median Value', 'Min Value', 'Max Value'
]
print("\nSummary statistics of asset values by currency:")
print(asset_value_gbp_by_currency.map('{:.2f}'.format))

# Bar chart (Total asset value by currency)
fig2, ax3 = plt.subplots(figsize=(10, 6))

sns.barplot(x=asset_value_gbp_by_currency.index,
            y='Total Value',
            data=asset_value_gbp_by_currency,
            ax=ax3)

ax3.set_title('Total Asset Value by Currency')
ax3.set_xlabel('Asset Currency')
ax3.set_ylabel('Total Value (GBP)')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', linestyle='--', alpha=0.7)
fig1.tight_layout()

# Bar chart (Average asset value by currency)
plt.figure(figsize=(10, 6))
sns.barplot(x=asset_value_gbp_by_currency.index,
            y='Average Value',
            data=asset_value_gbp_by_currency)
plt.title('Average Asset Value by Currency')
plt.xlabel('Asset Currency')
plt.ylabel('Average Value')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()


# Asset value over time
# Daily
daily_total_value = assets_df['asset_value_gbp'].resample('D').sum().fillna(0)
daily_total_value_df = daily_total_value.reset_index()
daily_total_value_df.columns = ['Date', 'Total_Value']

print(f'Minimum total asset value (GBP) made per day: {daily_total_value.min(): .2f}')
print(f'Maximum total asset value (GBP) made per day: {daily_total_value.max(): .2f}')
print(f'Mean total asset value (GBP) made per day: {daily_total_value.mean():.2f}')


# Weekly
weekly_total_value = assets_df['asset_value_gbp'].resample('W').sum().fillna(0)
weekly_total_value_df = weekly_total_value.reset_index()
weekly_total_value_df.columns = ['Week_End_Date', 'Total_Value']

print(f'Minimum total asset value (GBP) made per week: {weekly_total_value.min(): .2f}')
print(f'Maximum total asset value (GBP) made per week: {weekly_total_value.max(): .2f}')
print(f'Mean total asset value (GBP) made per week: {weekly_total_value.mean():.2f}')

# Monthly
monthly_total_value = assets_df['asset_value_gbp'].resample('ME').sum().fillna(0)
monthly_total_value_df = monthly_total_value.reset_index()
monthly_total_value_df.columns = ['Month', 'Total_Value']

print(f'Minimum total asset value (GBP) made per month: {monthly_total_value.min(): .2f}')
print(f'Maximum total asset value (GBP) made per month: {monthly_total_value.max(): .2f}')
print(f'Mean total asset value (GBP) made per month: {monthly_total_value.mean():.2f}')

# Line plot (Daily total asset value over time)
plt.style.use('seaborn-v0_8-darkgrid')

fig_daily, ax_daily = plt.subplots(figsize=(15, 7))
sns.lineplot(x='Date',
             y='Total_Value',
             data=daily_total_value_df,
             marker='o',
             markersize=3,
             linestyle='-',
             color='purple',
             alpha=0.7,
             ax=ax_daily)

ax_daily.set_title('Daily Total Asset Value Over Time')
ax_daily.set_xlabel('Date')
ax_daily.set_ylabel('Total Asset Value')


ax_daily.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax_daily.xaxis.set_minor_locator(
    mdates.WeekdayLocator(byweekday=mdates.MONDAY))

ax_daily.tick_params(axis='x', rotation=45)
ax_daily.grid(True, which="both", ls="-", alpha=0.6)
fig_daily.tight_layout()

# Line plot (Weekly total asset value over time)
fig_weekly, ax_weekly = plt.subplots(figsize=(15, 7))
sns.lineplot(x='Week_End_Date',
             y='Total_Value',
             data=weekly_total_value_df,
             marker='o',
             markersize=4,
             linestyle='-',
             color='green',
             alpha=0.8,
             ax=ax_weekly)

ax_weekly.set_title('Weekly Total Asset Value Over Time')
ax_weekly.set_xlabel('Week Ending Date')
ax_weekly.set_ylabel('Total Asset Value')


ax_weekly.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax_weekly.xaxis.set_minor_locator(
    mdates.WeekdayLocator(byweekday=mdates.MONDAY))

ax_weekly.tick_params(axis='x', rotation=45)
ax_weekly.grid(True, which="both", ls="-", alpha=0.6)
fig_weekly.tight_layout()

# Asset value by allocation type over time
# Line plot (Weekly average asset value by allocation type over time)
weekly_avg_value_by_type = assets_df.groupby(
    'asset_allocation')['asset_value_gbp'].resample('W').mean()
weekly_avg_value_by_type_df = weekly_avg_value_by_type.reset_index()
weekly_avg_value_by_type_df.columns = [
    'Asset_Type', 'Week_End_Date', 'Average_Value_GBP'
]


plt.style.use('seaborn-v0_8-darkgrid')
fig_type, ax_type = plt.subplots(figsize=(15, 8))

sns.lineplot(x='Week_End_Date',
             y='Average_Value_GBP',
             hue='Asset_Type',
             data=weekly_avg_value_by_type_df,
             marker='o',
             markersize=4,
             ax=ax_type)

ax_type.set_title('Weekly Average Asset Value Over Time by Asset Type')
ax_type.set_xlabel('Week Ending Date')
ax_type.set_ylabel('Average Asset Value')


ax_type.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax_type.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))

ax_type.tick_params(axis='x', rotation=45)
ax_type.grid(True, which="both", ls="-", alpha=0.6)
ax_type.legend(title='Asset Type', bbox_to_anchor=(1.05, 1),
               loc='upper left')
fig_type.tight_layout()

# Line plot (Weekly average asset value by currency over time)
weekly_avg_value_by_currency = assets_df.groupby(
    'asset_currency')['asset_value_gbp'].resample('W').mean()

weekly_avg_value_by_currency_df = weekly_avg_value_by_currency.reset_index()
weekly_avg_value_by_currency_df.columns = [
    'Asset_Currency', 'Week_End_Date', 'Average_Value'
]


fig_currency, ax_currency = plt.subplots(figsize=(15, 8))

sns.lineplot(x='Week_End_Date',
             y='Average_Value',
             hue='Asset_Currency',
             data=weekly_avg_value_by_currency_df,
             marker='o',
             markersize=4,
             ax=ax_currency)

ax_currency.set_title('Weekly Average Asset Value Over Time by Asset Currency')
ax_currency.set_xlabel('Week Ending Date')
ax_currency.set_ylabel('Average Asset Value')

ax_currency.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax_currency.xaxis.set_minor_locator(
    mdates.WeekdayLocator(byweekday=mdates.MONDAY))

ax_currency.tick_params(axis='x', rotation=45)
ax_currency.grid(True, which="both", ls="-", alpha=0.6)
ax_currency.legend(title='Asset Currency',
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left')
fig_currency.tight_layout()



# --- 2. Financial Personality Data Analysis ---
personality_df = pd.read_csv(
    '/Users/leungchinyu/Desktop/Financial Personality data.csv')

# 2.1 Initial Data Inspection
personality_df = personality_df.rename(
    columns={
        'confidence': 'Confidence',
        'risk_tolerance': 'Risk Tolerance',
        'composure': 'Composure',
        'impulsivity': 'Impulsivity',
        'impact_desire': 'Impact Desire'
    })

print("Financial Personality Data")
print("\nFirst 5 rows:")
print(personality_df.head())
print("-" * 50)

print("\nLast 5 rows:")
print(personality_df.tail())
print("-" * 50)

print("\nDataframe Information:")
print(personality_df.info())
print("-" * 50)

print(f'Number of duplicated rows: {personality_df.duplicated().sum()}')  # No duplicated rows
print("-" * 50)

# Excluding _id for analysis
numerical_cols = [
    'Confidence', 'Risk Tolerance', 'Composure', 'Impulsivity', 'Impact Desire'
]

# 2.2 Univariate Analysis ---
# Summary
print("\nPersonality Summary:")
print(personality_df[numerical_cols].describe().map('{:.2f}'.format))
print("-" * 50)

# Distributions
# Histograms
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.histplot(personality_df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()

# Box plots
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=personality_df[col].dropna())
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
plt.tight_layout()


# 2.3 Multivariate Analysis
# Correlation matrix
print("\nCorrelation Matrix")
correlation_matrix = personality_df[numerical_cols].corr()
print(correlation_matrix.map('{:.2f}'.format))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5)
plt.title('Correlation Matrix of Personality')

# Pair plots
sns.pairplot(personality_df[numerical_cols])
plt.suptitle('Pair Plots of Personality',
             y=1.02)


# --- 3. Merged Data Analysis ---

# 3.1 Initial Data Inspection
# Merging the two Financial Assets data and Financial Personality data
assets_df = assets_df.reset_index()
assets_df['_id'] = assets_df['_id'].astype(str)
personality_df['_id'] = personality_df['_id'].astype(str)
merged_df = pd.merge(assets_df, personality_df, on='_id', how='left')

# Data inspection
print("Merged Data (Financial Assets Data and Financial Personality Data)")
print("\nFirst 5 rows:")
print(merged_df.head())
print("-" * 50)

print("\nMissing Personality Data After Merge (if any)") # No missing data
print(merged_df[[
    'Confidence', 'Risk Tolerance', 'Composure', 'Impulsivity', 'Impact Desire'
]].isnull().sum())
print("-" * 50)

# Defining columns
personality_cols = [
    'Confidence', 'Risk Tolerance', 'Composure', 'Impulsivity', 'Impact Desire'
]
asset_categorical_cols = ['asset_allocation', 'asset_currency']
asset_continuous_cols = ['asset_value_gbp']


# 3.2 Multivariate Analysis
# Personality vs Asset Value
# Scatter plots
plt.figure(figsize=(18, 10))
for i, p_col in enumerate(personality_cols):
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=p_col, y='asset_value_gbp', data=merged_df, alpha=0.6)
    plt.title(f'{p_col} vs. Asset Value (GBP)')
    plt.xlabel(p_col)
    plt.ylabel('Asset Value (GBP)')
plt.tight_layout()


# Correlation matrix
corr_cols = personality_cols + asset_continuous_cols
correlation_matrix = merged_df[corr_cols].corr()
print("\nCorrelation Matrix (Personality vs Asset Value (GBP)")
print(correlation_matrix['asset_value_gbp'].drop(
    'asset_value_gbp').map('{:.2f}'.format))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5)
plt.title('Correlation Matrix (Personality & Asset Value (GBP))')
print("-" * 50)


# Personality vs Asset Allocation Type
# Box plots
plt.figure(figsize=(18, 10))
for i, p_col in enumerate(personality_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='asset_allocation', y=p_col, data=merged_df)
    plt.title(f'{p_col} by Asset Allocation Type')
    plt.xlabel('Asset Allocation Type')
    plt.ylabel(p_col)
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Personality vs Asset Currency
# Box plots
plt.figure(figsize=(18, 10))
for i, p_col in enumerate(personality_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='asset_currency', y=p_col, data=merged_df)
    plt.title(f'{p_col} by Asset Currency')
    plt.xlabel('Asset Currency')
    plt.ylabel(p_col)
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Mean personality scores of each asset allocation type
print("\nMean Personality Scores by Asset Allocation Type")
for p_col in personality_cols:
    print(f"\n{p_col} by Asset Allocation:")
    print(
        merged_df.groupby('asset_allocation')[p_col].mean().sort_values(
            ascending=False).map('{:.2f}'.format))
print("-" * 50)

# Mean personality scores of each asset currency
print("\nMean Personality Scores by Asset Currency")
for p_col in personality_cols:
    print(f"\n{p_col} by Asset Currency:")
    print(
        merged_df.groupby('asset_currency')[p_col].mean().sort_values(
            ascending=False).map('{:.2f}'.format))
print("-" * 50)


# Personality vs time
value_column = 'asset_value_gbp'
date_column = 'created'

# Loop through each personality column
for p_col in personality_cols:

    try:
        if p_col not in merged_df.columns:
            print(f"Warning: Column '{p_col}' not found in merged_df. Skipping.")
            continue

        bin_col_name = f'{p_col}_bin'

        if pd.api.types.is_numeric_dtype(merged_df[p_col]):
            try:
                merged_df[bin_col_name] = pd.qcut(
                    merged_df[p_col],
                    q=3,
                    labels=[f'Low {p_col}', f'Medium {p_col}', f'High {p_col}'],
                    duplicates='drop'
                )
            except ValueError as ve:
                print(f"Skipping {p_col} due to qcut error (likely too few unique values for 3 bins): {ve}")
                continue

            merged_df[bin_col_name] = merged_df[bin_col_name].astype('category')
        else:
            print(f"Warning: Column '{p_col}' is not numeric and cannot be binned. Skipping.")
            continue

        daily_value_by_trait_bin = merged_df.groupby(
            [date_column, bin_col_name],
            observed=False
        )[value_column].sum().reset_index() #Aggregating total asset value by date and the current personality bin (daily sums)

        df_indexed_by_date_for_weekly = daily_value_by_trait_bin.set_index(date_column)


        weekly_value_by_trait_bin = df_indexed_by_date_for_weekly.groupby([
            pd.Grouper(freq='W', level=date_column),  # Resampling the index to weekly
            bin_col_name
        ], observed=False
        )[value_column].sum().reset_index()

        weekly_value_by_trait_bin.rename(
            columns={date_column: 'week_end_date'}, inplace=True
        )


        # Line plots
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(x='week_end_date', y=value_column, hue=bin_col_name,
                     data=weekly_value_by_trait_bin, marker='o',
                     markersize=3, alpha=0.7, ax=ax)

        plt.title(f'Weekly Total Asset Value (GBP) by {p_col} Level')
        plt.xlabel('Week End Date')
        plt.ylabel('Total Asset Value (GBP)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=f'{p_col} Level')
        plt.grid(True, which="both", ls="-", alpha=0.6)
        plt.tight_layout()

    except Exception as e:
        print(f"Error processing {p_col}: {e}")
        print(f"Skipping plot for {p_col}. This might happen if a trait has too few unique values for qcut or other data issues.")


plt.show()
plt.close('all')

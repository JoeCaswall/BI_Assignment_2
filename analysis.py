import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

df = pd.read_csv('Crime_Reports_20240701.csv')

print(df.sample(5))

##################################################################################
#################################### Cleaning ####################################
##################################################################################

# Replace ranges with just the start time for 'Crime Date Time' to allow for datetime conversion
df['Crime Date Time'] = df['Crime Date Time'].str.split(' - ').str[0]

df['Date of Report'] = pd.to_datetime(df['Date of Report'], errors='coerce')
df['Crime Date Time'] = pd.to_datetime(df['Crime Date Time'], errors='coerce')

# Check conversions worked
print(df.dtypes)

# Add year column for easier analysis
df['Year'] = df['Crime Date Time'].dt.year.astype('Int64')

# Add hour column for time-based analysis
df['Hour'] = df['Crime Date Time'].dt.hour.astype('Int64')

# Filter data from 2009 onwards as there is limited data before this year

df = df[df['Year'] >= 2009]

# Add Is Violent column
violent_crimes = [
    'Aggravated Assault', 'Simple Assault', 'Homicide', 'Kidnapping',
    'Street Robbery', 'Commercial Robbery', 'Domestic Dispute', 'Stalking',
    'Threats', 'Extortion/Blackmail', 'Weapon Violations', 'Sex Offender Violation'
]
df['Is Violent'] = df['Crime'].isin(violent_crimes).astype(int)

# # Remove data from 2024 as it is incomplete
# df = df[df['Year'] <= 2023]

# Remove ', Cambridge, MA' from Location names for consistency
df['Location'] = df['Location'].str.replace(', Cambridge, MA', '', regex=False)

# Remove Date of Report column as not needed for analysis
df = df.drop(columns=['Date of Report'])

# Print list of all crimes
# print(df['Crime'].unique())

print(df.describe(include="all"))

# Drop rows with missing values in critical columns
df = df.dropna(subset=['Crime Date Time', 'Crime', 'Neighborhood', 'Reporting Area', 'Location'])

print("--------------------------------")
print("--------------------------------")
print("--------------------------------")

print(df.describe(include="all"))

##################################################################################
#################################### Analysis ####################################
##################################################################################


crime_counts_by_year = df.groupby('Year').size()
# print(crime_counts_by_year)

crime_type_trends = df.groupby(['Year', 'Crime']).size().unstack(fill_value=0)
# print(crime_type_trends)

crime_hourly = df.groupby(['Crime', 'Hour']).size().unstack(fill_value=0)
# print(crime_hourly)

most_common_crime_per_year = df.groupby('Year')['Crime'].agg(lambda x: x.value_counts().idxmax())
# print(most_common_crime_per_year)

crime_by_neighbourhood = df.groupby('Neighborhood').size().sort_values(ascending=False)
# print(crime_by_neighbourhood)

crime_type_by_neighbourhood = df.groupby(['Neighborhood', 'Crime']).size().unstack(fill_value=0)
# print(crime_type_by_neighbourhood)


######### Visualization #########

def plot_heatmap_violent_crimes_neighborhood_hour(df):
    # Filter for violent crimes only
    violent_df = df[df['Is Violent'] == 1]
    pivot = pd.crosstab(violent_df['Neighborhood'], violent_df['Hour'])
    plt.figure(figsize=(16,7))
    sns.heatmap(pivot, cmap='Reds')
    plt.title('Violent Crimes Across Neighborhoods and Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Neighborhood')
    plt.tight_layout()
    plt.show()


def plot_crime_numbers_over_years():
    plt.figure(figsize=(8,5))
    sns.barplot(x=crime_counts_by_year.index, y=crime_counts_by_year.values)
    plt.title('Crime Numbers Over Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Crimes')
    plt.xticks(rotation=75)
    plt.show()

def plot_crime_type_trends():
    plt.figure(figsize=(12,6))
    # Find the 10 crimes with the highest maximum yearly values
    top10_crimes = crime_type_trends.max().nlargest(10).index.tolist()
    crime_type_trends[top10_crimes].plot(kind='line')
    plt.title('Crime Type Trends Over Time (Top 10 by Maximum Yearly Value)')
    plt.xlabel('Year')
    plt.ylabel('Number of Crimes')
    plt.legend(title='Crime Type', loc='center left', bbox_to_anchor=(0.9, 0.8))
    plt.show()

def plot_hourly_distribution_of_crimes():
    plt.figure(figsize=(12,6))
    sns.heatmap(crime_hourly, cmap='YlGnBu')
    plt.title('Hourly Distribution of Crimes by Type')
    plt.xlabel('Hour')
    plt.ylabel('Crime Type')
    plt.show()

def plot_most_common_crime_per_year():
    years = sorted(df['Year'].dropna().unique())
    min_year = int(years[0])
    max_year = int(years[-1])
    periods = [(start, min(start+4, max_year)) for start in range(min_year, max_year+1, 5)]

    all_crimes = sorted(df['Crime'].dropna().unique())
    # Assign colors from a colormap
    cmap = cm.get_cmap('tab20', len(all_crimes))
    color_dict = {crime: cmap(i) for i, crime in enumerate(all_crimes)}
    
    n = len(periods)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    for ax, (start, end) in zip(axes, periods):
        period_df = df[(df['Year'] >= start) & (df['Year'] <= end)]
        data = period_df['Crime'].value_counts()
        # Only label the largest 5 slices
        top_five = data.nlargest(5)
        top_seven = data.nlargest(7)
        colors = [color_dict[crime] for crime in data.index]
        def autopct_func(pct, allvals=data):
            absolute = int(round(pct/100.*sum(allvals)))
            # Only show percentage if this slice is in top 7
            return '%1.1f%%' % pct if absolute in top_seven.values else ''
        labels = [label if label in top_seven.index else '' for label in data.index]
        ax.pie(data, labels=labels, autopct=autopct_func, startangle=90, colors=colors)
        ax.set_title(f'Crime Types {start}-{end}')
    # Add a legend for all crime types at the bottom
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=crime,
                          markerfacecolor=color_dict[crime], markersize=10)
               for crime in all_crimes]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(all_crimes), 6), bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.show()

def plot_crime_rate_by_neighbourhood():
    plt.figure(figsize=(10,6))
    crime_by_neighbourhood.plot(kind='bar')
    plt.title('Crime Rate by Neighbourhood')
    plt.xlabel('Neighbourhood')
    plt.ylabel('Number of Crimes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_crime_type_by_neighbourhood():
    # Find top 7 crimes overall
    top7_crimes = df['Crime'].value_counts().nlargest(7).index.tolist()
    # Filter columns to only top 7 crimes
    filtered = crime_type_by_neighbourhood[top7_crimes]
    filtered.plot(kind='bar', stacked=True, figsize=(12,8))
    plt.title('Top 7 Crime Types by Neighbourhood')
    plt.xlabel('Neighbourhood')
    plt.ylabel('Number of Crimes')
    plt.xticks(rotation=45)
    plt.legend(title='Crime Type')
    plt.tight_layout()
    plt.show()

def plot_pie_chart_crime_numbers_by_reporting_area():
    crime_counts_by_location = df['Reporting Area'].value_counts()
    plt.figure(figsize=(8,8))
    # Only label the largest 5 slices
    top_five = crime_counts_by_location.nlargest(5)
    top_seven = crime_counts_by_location.nlargest(7)
    colors = cm.tab20.colors
    def autopct_func(pct, allvals=crime_counts_by_location):
        absolute = int(round(pct/100.*sum(allvals)))
        # Only show percentage if this slice is in top 7
        return '%1.1f%%' % pct if absolute in top_seven.values else ''
    labels = [label if label in top_seven.index else '' for label in crime_counts_by_location.index]
    plt.pie(crime_counts_by_location, labels=labels, autopct=autopct_func, startangle=90, colors=colors)
    plt.title('Crime Numbers by Reporting Area')
    plt.tight_layout()
    plt.show()

def find_comparible_crime_levels_in_prior_years():
    # Find the latest date in 2024
    latest_2024_date = df[df['Year'] == 2024]['Crime Date Time'].max()

    # Filter each prior year to only include data up to that date
    cutoff_month = latest_2024_date.month
    cutoff_day = latest_2024_date.day

    # For each year before 2024, filter to only include entries up to cutoff month/day
    mask = (df['Year'] < 2024) & (
        (df['Crime Date Time'].dt.month < cutoff_month) |
        ((df['Crime Date Time'].dt.month == cutoff_month) & (df['Crime Date Time'].dt.day <= cutoff_day))
    )
    filtered_df = df[mask]

    # Group and compare crime levels for these partial years
    crime_counts = filtered_df.groupby('Year').size()
    print(crime_counts)

def get_partial_year_crime_counts(df):
    """
    Returns a Series of crime counts for each year up to the latest date in 2024.
    """
    latest_2024_date = df[df['Year'] == 2024]['Crime Date Time'].max()
    cutoff_month = latest_2024_date.month
    cutoff_day = latest_2024_date.day
    print(f"latest 2024 date: {latest_2024_date}")
    print(f"cutoff month: {cutoff_month}, cutoff day: {cutoff_day}")
    mask = (df['Year'] <= 2024) & (
        (df['Crime Date Time'].dt.month < cutoff_month) |
        ((df['Crime Date Time'].dt.month == cutoff_month) & (df['Crime Date Time'].dt.day <= cutoff_day))
    )
    filtered_df = df[mask]
    return filtered_df.groupby('Year').size()

def plot_partial_year_crime_counts(crime_counts):
    """
    Plots a bar graph of crime counts for each year up to the latest 2024 date.
    """
    if crime_counts is None or crime_counts.empty:
        print("No partial year crime counts to plot.")
        return
    plt.figure(figsize=(8,5))
    crime_counts.plot(kind='bar')
    plt.title('Crime Counts by Year (up to latest 2024 date)')
    plt.xlabel('Year')
    plt.ylabel('Number of Crimes')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.show()

def plot_heatmap_crime_vs_hour(df):
    pivot = pd.crosstab(df['Crime'], df['Hour'])
    plt.figure(figsize=(14,10))
    plt.yticks(fontsize=8, rotation=0)
    sns.heatmap(pivot, cmap='YlGnBu')
    plt.title('Crime Type vs. Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Crime Type')
    plt.tight_layout()
    plt.show()

def plot_heatmap_crime_vs_neighborhood(df):
    # Find top 5 neighborhoods by total crime count
    top5_neighborhoods = df['Neighborhood'].value_counts().nlargest(5).index.tolist()
    pivot = pd.crosstab(df['Crime'], df['Neighborhood'])
    pivot = pivot[top5_neighborhoods]
    plt.figure(figsize=(10,8))
    plt.yticks(fontsize=8, rotation=0)
    sns.heatmap(pivot, cmap='YlGnBu')
    plt.title('Crime Type vs. Top 5 Neighborhoods')
    plt.xlabel('Neighborhood')
    plt.ylabel('Crime Type')
    plt.tight_layout()
    plt.show()

def plot_heatmap_crime_vs_year(df):
    pivot = pd.crosstab(df['Crime'], df['Year'])
    plt.figure(figsize=(14,8))
    plt.yticks(fontsize=8, rotation=0)
    sns.heatmap(pivot, cmap='YlGnBu')
    plt.title('Crime Type vs. Year')
    plt.xlabel('Year')
    plt.ylabel('Crime Type')
    plt.tight_layout()
    plt.show()

def plot_heatmap_hour_vs_neighborhood(df):
    pivot = pd.crosstab(df['Neighborhood'], df['Hour'])
    plt.figure(figsize=(16,7))
    sns.heatmap(pivot, cmap='YlGnBu')
    plt.title('Hour vs. Neighborhood')
    plt.xlabel('Hour of Day')
    plt.ylabel('Neighborhood')
    plt.tight_layout()
    plt.show()

def plot_heatmap_reporting_area_vs_hour(df):
    pivot = pd.crosstab(df['Reporting Area'], df['Hour'])
    plt.figure(figsize=(16,10))
    sns.heatmap(pivot, cmap='YlGnBu')
    plt.title('Reporting Area vs. Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Reporting Area')
    plt.tight_layout()
    plt.show()

crime_counts_partial_years = get_partial_year_crime_counts(df)
# plot_partial_year_crime_counts(crime_counts_partial_years)

# plot_crime_numbers_over_years()
# plot_crime_type_trends()
# plot_hourly_distribution_of_crimes()
# plot_most_common_crime_per_year()
# plot_crime_rate_by_neighbourhood()
# plot_crime_type_by_neighbourhood()
# plot_pie_chart_crime_numbers_by_reporting_area()

# Heatmaps

# plot_heatmap_crime_vs_hour(df)
# plot_heatmap_crime_vs_neighborhood(df)
# plot_heatmap_crime_vs_year(df)
# plot_heatmap_hour_vs_neighborhood(df)
plot_heatmap_violent_crimes_neighborhood_hour(df)
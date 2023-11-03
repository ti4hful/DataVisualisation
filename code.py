# %%
pip install pandas

# %%
pip install plotly

# %%
import pandas as pd
import numpy as np

# %%
import matplotlib.pyplot as plt
import seaborn as sns


# %%
df = pd.read_csv(r'C:\Codebase\assignments\DataVisual\dataset\2019.csv')
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df.info()

# %%
df.isnull().sum()

# %%
df.describe()

# %%
#get the score of top 15 ranking countries
df.sort_values(by=['Overall rank'],ascending=True).iloc[:15][['Overall rank','Country or region','Score']]

# %%
# Extract: Read data from the source CSV file
source_file = r'C:\Codebase\assignments\DataVisual\dataset\2019.csv'
df = pd.read_csv(source_file)

# %%
target_file = "transformed_data.csv"
df.to_csv(target_file, index=False)

print("ETL process completed. Transformed data saved to", target_file)

# %%


# %%
import matplotlib.pyplot as plt

top_N = 15
top_countries = df.head(top_N)

plt.figure(figsize=(12, 6))
plt.bar(top_countries['Country or region'], top_countries['Score'])
plt.title(f'Top {top_N} Happiest Countries (2019)')
plt.xlabel('Country')
plt.ylabel('Happiness Score')
plt.xticks(rotation=45)
plt.show()


# %%
# Bar Chart: Top 15 Happiest Countries
top_N = 15
top_countries = df.head(top_N)

bar_color = 'purple'

plt.figure(figsize=(12, 6))
plt.bar(top_countries['Country or region'], top_countries['Score'], color=bar_color)
plt.title(f'Top {top_N} Happiest Countries (2019)')
plt.xlabel('Country')
plt.ylabel('Happiness Score')
plt.xticks(rotation=45)
plt.show()


# %%
#Scatter Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['GDP per capita'], df['Score'], color='pink', alpha=0.7)
plt.title('GDP vs. Happiness Score (2019)')
plt.xlabel('GDP per Capita')
plt.ylabel('Happiness Score')
plt.show()



# %%
# Correlation heatmap
numeric_columns = df.select_dtypes(include='number')

corr_matrix = numeric_columns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# %%
# Pairplot for Multiple Variable Analysis
sns.pairplot(df, vars=["Score", "GDP per capita", "Social support", "Healthy life expectancy"])
plt.suptitle('Pairplot of Happiness Factors (2019)')
plt.show()


# %%

# Try a pastel color palette
pastel_palette = ["#FFD1DC", "#FFABAB", "#FFC3A0", "#FF677D"]

# Use the custom pastel color palette in the pairplot
sns.set_palette(sns.color_palette(pastel_palette))
sns.pairplot(df, vars=["Score", "GDP per capita", "Social support", "Healthy life expectancy"])
plt.suptitle('Pairplot of Happiness Factors (2019)')
plt.show()


# %%

# Try another one "pastel" palette
sns.set_palette("pastel")

sns.pairplot(df, vars=["Score", "GDP per capita", "Social support", "Healthy life expectancy"])
plt.suptitle('Pairplot of Happiness Factors (2019)')
plt.show()


# %%
# Box Plot for Happiness Score Distribution
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Score", color='plum')
plt.title('Distribution of Happiness Scores (2019)')
plt.xlabel('Happiness Score')
plt.show()


# %%
# Bar Chart for Generosity by Country

generous_countries = df.sort_values(by="Generosity", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=generous_countries, x="Generosity", y="Country or region", color='blueviolet')
plt.title('Top 10 Most Generous Countries (2019)')
plt.xlabel('Generosity')
plt.ylabel('Country')
plt.show()



# %%
# Histogram of GDP
plt.figure(figsize=(8, 6))
plt.hist(df['GDP per capita'], bins=20, color='thistle', edgecolor='black')
plt.title('Distribution of GDP (2019)')
plt.xlabel('GDP per Capita')
plt.ylabel('Frequency')
plt.show()


# %%
# Radar Chart to Compare Happiness Factors

countries_to_compare = ["Finland", "Denmark", "Norway", "Iceland", "Netherlands"]

selected_countries = df[df['Country or region'].isin(countries_to_compare)]

# Create a radar chart
factors = ["GDP per capita", "Social support", "Healthy life expectancy", "Freedom to make life choices", "Generosity", "Perceptions of corruption"]
data = selected_countries[factors].values.T
labels = factors

angles = [n / float(len(labels)) * 2 * 3.14159265359 for n in range(len(labels))]
angles += angles[:1]

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.yticks([0.2, 0.4, 0.6, 0.8], color="lightcoral", size=10)
plt.ylim(0, 1)

for i in range(len(selected_countries)):
    values = data[:, i].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=selected_countries.iloc[i]["Country or region"])
    ax.fill(angles, values, 'b', alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Comparison of Happiness Factors by Country (2019)')
plt.show()


# %%
# Bubble Plot to Visualize Three Variables
plt.figure(figsize=(10, 6))
plt.scatter(
    x=df['GDP per capita'],
    y=df['Score'],
    c=df['Social support'],
    s=df['Healthy life expectancy'] * 10,
    cmap='viridis',
    alpha=0.7
)
plt.title('Bubble Plot: GDP, Happiness Score, and Social Support (2019)')
plt.xlabel('GDP')
plt.ylabel('Happiness Score')
plt.colorbar(label='Social Support')
plt.show()

# %%
from matplotlib import colormaps
list(colormaps)

# %%
# Bubble Plot to Visualize Three Variables with different colors
plt.figure(figsize=(10, 6))
plt.scatter(
    x=df['GDP per capita'],
    y=df['Score'],
    c=df['Social support'],
    s=df['Healthy life expectancy'] * 15,
    cmap='Purples',
    alpha=0.7
)
plt.title('Bubble Plot: GDP, Happiness Score, and Social Support (2019)')
plt.xlabel('GDP')
plt.ylabel('Happiness Score')
plt.colorbar(label='Social Support')
plt.show()



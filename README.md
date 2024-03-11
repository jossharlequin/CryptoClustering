# CryptoClustering

# %%
# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    "Resources/crypto_market_data.csv",
    index_col="coin_id")

# Display sample data
df_market_data.head(10)

# %%
# Generate summary statistics
df_market_data.describe()

# %%
# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)

# %% [markdown]
# ---

# %% [markdown]
# ### Prepare the Data

# %%
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
market_data_scaled = StandardScaler().fit_transform(df_market_data[["price_change_percentage_24h", "price_change_percentage_7d", "price_change_percentage_14d", "price_change_percentage_30d", "price_change_percentage_60d", "price_change_percentage_200d", "price_change_percentage_1y"]])

# %%
# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(market_data_scaled, columns=["price_change_percentage_24h", "price_change_percentage_7d", "price_change_percentage_14d", "price_change_percentage_30d", "price_change_percentage_60d", "price_change_percentage_200d", "price_change_percentage_1y"])

# Copy the crypto names from the original data

# Set the coinid column as index
df_market_data_scaled["coin_id"] = df_market_data.index  # Adding coin_id to the transformed DataFrame
df_market_data_scaled.set_index("coin_id", inplace=True)

# Display sample data
df_market_data_scaled.head()

# %% [markdown]
# ---

# %% [markdown]
# ### Find the Best Value for k Using the Original Data.

# %%
# Create a list with the number of k-values from 1 to 11
k = list(range(1, 11))

# %%
# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
for i in k:
    k_model = KMeans(n_clusters=i)
    k_model.fit(df_market_data_transformed)
    inertia.append(k_model.inertia_)
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list


# %%
# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_1 = pd.DataFrame(elbow_data)

# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow_1.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)


# %% [markdown]
# #### Answer the following question: 
# 
# **Question:** What is the best value for `k`?
# 
# # I chose 4 because afterwards is where the curve flattens out.

# %% [markdown]
# ---

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the Original Data

# %%
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)

# %%
# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)

# %%
# Predict the clusters to group the cryptocurrencies using the scaled data
crypto_clusters = model.predict(df_market_data_scaled)

# Print the resulting array of cluster values.
print(crypto_clusters)

# %%
# Create a copy of the DataFrame
df_market_data_predictions = df_market_data_scaled.copy()

# %%
# Add a new column to the DataFrame with the predicted clusters
df_market_data_predictions['PredictedCluster'] = crypto_clusters

# Display sample data
df_market_data_predictions[0:5]

# %%
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
scatter_plot = df_market_data_predictions.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="PredictedCluster",
    hover_cols=["coin_id"],
    width=800,
    height=500
)
# Show the scatter plot
scatter_plot

# %% [markdown]
# ---

# %% [markdown]
# ### Optimize Clusters with Principal Component Analysis.

# %%
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)

# %%
# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
market_data_pca = pca.fit_transform(df_market_data_scaled)
# View the first five rows of the DataFrame. 
market_data_pca[0:5]


# %%
# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_

# %% [markdown]
# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# # The total explained variance is around 90% which indicates this will acurately estimate the data 90% of the time.

# %%
# Creating a DataFrame with the PCA data
df_market_data_pca = pd.DataFrame(
    market_data_pca,
    columns=["PCA1", "PCA2", "PCA3"]
)

# Set the 'coin_id' column as the index
df_market_data_pca.index = df_market_data.index

# Display sample data
df_market_data_pca.head()

# %% [markdown]
# ---

# %% [markdown]
# ### Find the Best Value for k Using the PCA Data

# %%
# Create a list with the number of k-values from 1 to 11
k = list(range(1, 11))

# %%
# Create an empty list to store the inertia values


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
inertia = []

for i in k:
    k_model = KMeans(n_clusters=i)
    k_model.fit(df_market_data_pca)
    inertia.append(k_model.inertia_)


# %%
# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_2 = pd.DataFrame(elbow_data)


# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow_2.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)


# %% [markdown]
# #### Answer the following questions: 
# 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#  # I chose 4.
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
# # In this example it does not differ.

# %%


# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the PCA Data

# %%
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)

# %%
# Fit the K-Means model using the PCA data
model.fit(df_market_data_pca)

# %%
# Predict the clusters to group the cryptocurrencies using the PCA data
k_4 = model.predict(df_market_data_pca)
# Print the resulting array of cluster values.
print(k_4)

# %%
# Create a copy of the DataFrame with the PCA data
df_market_data_pca_predictions = df_market_data_pca.copy()

# Add a new column to the DataFrame with the predicted clusters
df_market_data_pca_predictions["predicted_clusters"] = k_4

# Display sample data
df_market_data_pca_predictions[0:5]


# %%
# Create a scatter plot using hvPlot by setting
 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
scatter_plot_pca = df_market_data_pca_predictions.hvplot.scatter(
    x="PCA1",
    y="PCA2",
    by="predicted_clusters",
    hover_cols=["coin_id"],
    width=800,
    height=500
)

# Show the plot
scatter_plot_pca


# %% [markdown]
# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

# %%
# Composite plot to contrast the Elbow curves
df_elbow_1.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
) + df_elbow_2.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)

# %%
# Composite plot to contrast the clusters
scatter_plot + scatter_plot_pca

# %% [markdown]
# #### Answer the following question: 
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
# # In this case, using fewer features allow the data points to be more tightly clumped together so that one can clearly see the divergence between clusters 1 and 2 in the PCA version as compared to the original scaled version.



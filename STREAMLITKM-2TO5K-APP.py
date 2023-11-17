import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import base64
#Run the app using 'streamlit run STREAMLITKM-2TO5K-APP.py' in your terminal.
#pip.exe freeze > requirements.tx 

# Global definition of centroids colors and shapes
centroids_colors = ['red', 'blue', 'green', 'purple', 'orange']
centroids_shapes = ['D', 's', '^', 'o', '*']

# Function to calculate misclassification rate
def calculate_misclassification_rate(true_labels, predicted_labels):
    # Assuming the majority class in each cluster is the correct label
    cluster_labels = {}
    for i in np.unique(predicted_labels):
        labels_in_cluster = true_labels[predicted_labels == i]
        most_common = np.bincount(labels_in_cluster).argmax()
        cluster_labels[i] = most_common

    misclassified = sum([true_label != cluster_labels[pred_label] for true_label, pred_label in zip(true_labels, predicted_labels)])
    total = len(true_labels)
    return misclassified / total

def visualize_kmeans_steps(X, true_labels, feature_names, n_clusters, max_iter, initial_centroids):
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1, max_iter=1, random_state=0)
    plots = []  # List to store plot figures
    overall_info = []

    for i in range(max_iter):
        kmeans.max_iter = i + 1
        kmeans.fit(X)

        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_


        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
        # Loop through the number of clusters for plotting
        for j in range(n_clusters):
            color = centroids_colors[j % len(centroids_colors)]
            shape = centroids_shapes[j % len(centroids_shapes)]
            # ... [plotting code] ...
            ax.scatter(centroids[j, 0], centroids[j, 1], c=color, marker=shape, edgecolor='k')

        ax.set_xlabel(feature_names[0], fontsize=8)
        ax.set_ylabel(feature_names[1], fontsize=8)
        ax.set_title(f'Iteration {i+1}', fontsize=12)
        plots.append(fig)

        # Metrics calculation
        misclassification_rate = calculate_misclassification_rate(true_labels, labels)
        inertia = round(kmeans.inertia_, 2)
        silhouette = round(silhouette_score(X, labels), 2)

        # Convert cluster frequencies to string or another simple format
        cluster_freq_str = ', '.join([f'{key}: {value}' for key, value in pd.Series(labels).value_counts().items()])

        # Include the predicted labels in the iteration_info
        iteration_info = {
            "Iteration": i + 1,
            "Cluster Means": centroids.round(2).tolist(),
            "Cluster Frequencies": pd.Series(labels).value_counts().to_dict(),
            "Misclassification Rate": misclassification_rate,
            "Inertia": inertia,
            "Silhouette Score": silhouette,
            "Cluster Frequencies": cluster_freq_str,
            "Predicted Labels": labels.tolist()  # Convert numpy array to list
        }
        overall_info.append(iteration_info)

    return overall_info, plots

#### Streamlit code 
# Function to generate a download link for a DataFrame
def get_csv_download_link(df, filename, button_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
    return href


# Streamlit app layout
st.title("K-Means VISUAL CLUSTERS: Teaching Tool")

# Input parameters using Streamlit widgets
# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')
feature_names_input = st.sidebar.text_input("Enter names for two features (comma-separated)", "Ease of Use, Sophisticated Technology")
n_samples = st.sidebar.slider("Number of data points", 50, 500, 100)
cluster_std = st.sidebar.slider("Standard deviation for clusters", 0.5, 3.0, 1.0)
num_true_clusters = st.sidebar.slider("Number of True Clusters", 2, 5, 3)
num_predict_clusters = st.sidebar.slider("Number of Predicted Clusters", 2, 5, 3)
max_iter = st.sidebar.slider("Maximum number of iterations", 2, 20, 6)
initial_centroids_input = st.sidebar.text_input("Initial centroids (x,y)", "9,8; 5,5;-2,-2")


run_simulation = st.sidebar.button("Run K-Means Simulation")

if run_simulation:
    feature_names = feature_names_input.split(',')
    initial_centroids = np.array([list(map(float, point.split(','))) for point in initial_centroids_input.split(';')])

    # Define cluster centers based on the number of true clusters
    # You can adjust these values or make them dynamic based on num_true_clusters
    if num_true_clusters == 2:
        centers = np.array([[2, 2], [8, 8]])
    elif num_true_clusters == 3:
        centers = np.array([[2, 2], [8, 8], [5, 12]])
    elif num_true_clusters == 4:
        centers = np.array([[2, 2], [8, 8], [3, 12], [10, 12]])
    elif num_true_clusters == 5:
        centers = np.array([[2, 2], [8, 8], [5, 12], [2, 6], [8, 6]])
    
    # Generate synthetic data with the specified centers
    X, true_labels = make_blobs(n_samples=n_samples, centers=centers, n_features=2, cluster_std=cluster_std)

    # Initial plot with user-specified centroids
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', marker='o', edgecolor='k')
    for centroid, color, shape in zip(initial_centroids, centroids_colors, centroids_shapes):
        ax.scatter(centroid[0], centroid[1], c=color, marker=shape, edgecolor='k')
    ax.set_xlabel(feature_names[0], fontsize=8)
    ax.set_ylabel(feature_names[1], fontsize=8)
    ax.set_title('Initial Data with Specified Centroids', fontsize=12)
    st.pyplot(fig)  # Display the initial plot immediately

    # Run K-Means clustering
    cluster_info, plots = visualize_kmeans_steps(X, true_labels, feature_names, num_predict_clusters, max_iter, initial_centroids)

    # Display plots and cluster information
    for plot in plots:
        st.pyplot(plot)

    # Convert cluster_info to DataFrame
    cluster_info_df = pd.DataFrame(cluster_info)

    # Apply formatting selectively
    # For example, if 'Inertia' and 'Silhouette Score' are the only numeric columns:
    formatted_cluster_info = cluster_info_df.style.format({
        'Inertia': "{:.2f}",
        'Silhouette Score': "{:.2f}"
    })
    st.write(formatted_cluster_info)

    # Extract last_labels correctly from the cluster_info
    last_labels = np.array(cluster_info[-1]['Predicted Labels'])  # Convert back to numpy array if needed
    
    # Create and display the hit/miss table for the last iteration
    hit_miss_table = pd.DataFrame({'True Label': true_labels, 'Predicted Label': last_labels})
    hit_miss_table['Hit/Miss'] = np.where(hit_miss_table['True Label'] == hit_miss_table['Predicted Label'], 'Hit', 'Miss')
    hit_miss_summary = pd.crosstab(hit_miss_table['True Label'], hit_miss_table['Predicted Label'], rownames=['True'], colnames=['Predicted'])
    st.write(hit_miss_summary)

# Running the Streamlit App instructions
st.markdown("""
CREATED BY DR. SUNDAR BALAKRISHNAN.
1. Save this script in a Python file (e.g., app.py).
2. Run the app using 'streamlit run STREAMLITKM-2TO5K-APP.py' in your command line or terminal.
""")

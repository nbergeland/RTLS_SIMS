# RTLS_SIMS
RTLS Simulations
# Example 1: Simulated RTLS Position Data
This code simulates location data of multiple assets moving randomly within a defined space:
#Simulated RTLS Position Data with Plotly
import numpy as np
import pandas as pd
import plotly.express as px

#Simulation parameters
np.random.seed(42)
num_assets = 10
num_timesteps = 100

#Generate random walk data for assets
positions = np.cumsum(np.random.randn(num_timesteps, num_assets, 2), axis=0)

#Create DataFrame for Plotly
data = []
for asset in range(num_assets):
    df_asset = pd.DataFrame({
        'X': positions[:, asset, 0],
        'Y': positions[:, asset, 1],
        'Time': range(num_timesteps),
        'Asset': f'Asset {asset + 1}'
    })
    data.append(df_asset)

df_positions = pd.concat(data)

#Plot asset movements
fig = px.line(df_positions, x='X', y='Y', color='Asset',
              title='Simulated RTLS Asset Movements',
              labels={'X': 'X Coordinate', 'Y': 'Y Coordinate'})

fig.update_layout(width=900, height=600)
fig.show()

# Example 2
# KMeans Clustering of RTLS Data with Plotly
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

#Simulated RTLS data
np.random.seed(42)
positions = np.random.rand(100, 2) * 100  # Random positions in a 100x100 area

#Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42).fit(positions)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

#Create DataFrame for Plotly
df_clusters = pd.DataFrame({
    'X': positions[:, 0],
    'Y': positions[:, 1],
    'Cluster': labels.astype(str)
})

#Plot clustering results with centroids
fig = px.scatter(df_clusters, x='X', y='Y', color='Cluster',
                 title='KMeans Clustering of RTLS Asset Positions',
                 labels={'X': 'X Coordinate', 'Y': 'Y Coordinate'})

#Add centroid markers explicitly
fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1],
                mode='markers', marker=dict(size=15, symbol='x', color='black'),
                name='Centroids')

fig.update_layout(width=900, height=600)
fig.show()

# Example 3
# Python RTLS Hospital Simulation (with Boundaries and Realistic Movement)
import numpy as np
import pandas as pd  # <-- Previously missing import
import plotly.express as px

#Simulation parameters
np.random.seed(42)
num_employees = 5
num_patients = 10
num_timesteps = 50

#Employee movement (larger random walk within hospital)
def simulate_employee_movement():
    positions = np.zeros((num_timesteps, num_employees, 2))
    positions[0] = np.random.uniform(20, 80, size=(num_employees, 2))

    for t in range(1, num_timesteps):
        step = np.random.randn(num_employees, 2) * 2
        positions[t] = positions[t-1] + step
        positions[t, :, 0] = np.clip(positions[t, :, 0], 0, 100)
        positions[t, :, 1] = np.clip(positions[t, :, 1], 0, 100)
    return positions

#Patient movement (restricted near room)
def simulate_patient_movement():
    positions = np.zeros((num_timesteps, num_patients, 2))
    patient_rooms = np.random.uniform(30, 70, size=(num_patients, 2))

    for t in range(num_timesteps):
        step = np.random.randn(num_patients, 2) * 0.5
        positions[t] = patient_rooms + step
        positions[t, :, 0] = np.clip(positions[t, :, 0], patient_rooms[:, 0]-3, patient_rooms[:, 0]+3)
        positions[t, :, 1] = np.clip(positions[t, :, 1], patient_rooms[:, 1]-3, patient_rooms[:, 1]+3)
    return positions

#Define parameters explicitly
num_timesteps = 100
employee_positions = simulate_employee_movement()
patient_positions = simulate_patient_movement()

#Import pandas explicitly
import pandas as pd  

#Convert data into DataFrame
def positions_to_df(positions, role):
    data = []
    num_entities = positions.shape[1]
    for entity in range(num_entities):
        df_entity = pd.DataFrame({
            'X': positions[:, entity, 0],
            'Y': positions[:, entity, 1],
            'Time': np.arange(num_timesteps),
            'Role': role,
            'ID': f'{role}_{entity+1}'
        })
        data.append(df_entity)
    return pd.concat(data)

#Parameters explicitly defined
num_timesteps = 100
num_employees = 5
num_patients = 10

#Simulate
employee_positions = simulate_employee_movement()
patient_positions = simulate_patient_movement()

#Combine into one dataframe
import pandas as pd  # Explicitly ensuring this is imported
df_employees = positions_to_df(employee_positions, 'Employee')
df_patients = positions_to_df(patient_positions, 'Patient')
df_rtls = pd.concat([df_employees, df_patients])

#Visualization using Plotly
import plotly.express as px

fig = px.scatter(df_rtls, x='X', y='Y', color='Role', 
                 animation_frame='Time', animation_group='ID',
                 title='Realistic RTLS Hospital Simulation',
                 labels={'X':'X Coordinate', 'Y':'Y Coordinate'},
                 range_x=[0, 100], range_y=[0, 100])

fig.update_layout(width=900, height=600)
fig.show()

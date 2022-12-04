import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib as plt
from matplotlib.colors import LinearSegmentedColormap
from catboost import CatBoostRegressor

# define variables
cpus = np.sort(['Intel Core i9-9900K', 'Intel Core i7-9700K',
				'Intel Core i7-8700K', 'Intel Core i7-7700K',
				'Intel Core i5-9400F', 'Intel Core i5-8600K', 'Intel Core i5-8400',
				'Intel Core i5-7600K', 'Intel Core i5-7500', 'Intel Core i3-7100',
				'AMD Ryzen 9 3900X', 'AMD Ryzen 7 3800X', 'AMD Ryzen 7 3700X',
				'AMD Ryzen 5 3600X', 'AMD Ryzen 5 3600', 'AMD Ryzen 5 2600X',
				'AMD Ryzen 5 1600X', 'AMD Ryzen 5 2600', 'AMD Ryzen 7 1700X'])
gpus = np.sort(['NVIDIA GeForce RTX 2080 Ti', 'NVIDIA GeForce RTX 2080',
				'NVIDIA GeForce GTX 1080 Ti', 'NVIDIA GeForce GTX TITAN X',
				'NVIDIA GeForce RTX 2070 SUPER', 'NVIDIA GeForce RTX 2070',
				'NVIDIA GeForce GTX 1080 11Gbps', 'NVIDIA GeForce RTX 2060 SUPER',
				'NVIDIA GeForce RTX 2060', 'NVIDIA GeForce GTX 1070 Ti',
				'NVIDIA GeForce GTX 1070 GDDR5X', 'NVIDIA GeForce GTX 1660 Ti',
				'NVIDIA GeForce GTX 1660 SUPER', 'NVIDIA GeForce GTX 980 Ti',
				'NVIDIA GeForce GTX 1660', 'NVIDIA GeForce GTX 1060 6 GB GDDR5X',
				'NVIDIA GeForce GTX 980', 'NVIDIA GeForce GTX 1060 5 GB',
				'NVIDIA GeForce GTX 970', 'NVIDIA GeForce GTX 1050 Ti',
				'NVIDIA GeForce GTX 1050 3 GB', 'AMD Radeon RX 5700 XT',
				'AMD Radeon RX 5700', 'AMD Radeon Pro Vega 64',
				'AMD Radeon RX 590', 'AMD Radeon RX 580', 'AMD Radeon RX 570'])
game_name = ['Call Of Duty WW2', 'Fortnite', 'Path Of Exile', 'Destiny 2',
			 'Radical Heights', 'League Of Legends', 'Overwatch',
			 'Player Unknowns Battlegrounds', 'Dota 2',
			 'Counter Strike Global Offensive', 'Sea Of Thieves',
			 'Apex Legends', 'Frostpunk', 'Total War 3 Kingdoms',
			 'World Of Tanks', 'Battlefield 4', 'Warframe', 'Air Mech Strike',
			 'Battletech', 'Far Cry 5', 'Starcraftc2', 'Rainbow Six Siege',
			 'Grand Theft Auto 5', 'A Way Out']
setting = ['med', 'max']
cmap = LinearSegmentedColormap.from_list('rg', ['#ff6961', '#77dd77'], N=256)

# load model
model = pickle.load(open('model.pkt', 'rb'))

# Header
st.header('🎮 Predicting FPS in games based on your hardware')
st.markdown('Here you can specify your cpu model and gpu model and get fps predictions '
		'for several games. The `med` and `max` columns indicate the number of fps at medium '
		'and maximum graphics settings in the game.')
col1, col2 = st.columns(2)
# Subheaders
col2.subheader('Your results in the table below!')
col1.subheader('Select your hardware')

# Selectboxes
with col1:
	cpu_input = st.selectbox('Select CPU:', options=cpus, help='Start typing your CPU model and '
															   'select from the drop down list 🚀')
with col1:
	gpu_input = st.selectbox('Select GPU:', options=gpus, help='Start typing your GPU model and '
															   'select from the drop down list 🚀')

# Defining output df
df = pd.DataFrame(data=None, index=game_name, columns=setting, dtype=int, copy=None)

# Filling output df
for i in df.index:
	for j in df.columns:
		pred = int(model.predict([cpu_input, gpu_input, i, j]).astype('int'))
		df[j][i] = pred
df = df.astype('int')

# Displaying output df
col2.dataframe(df.style.background_gradient(cmap=cmap, vmin=0, vmax=150), use_container_width=True)

# Final message
st.success('Thank you for interacting with this model. '
			 'You can find the source code on [my GitHub](https://github.com/VasilyBakshaev/FPS_Games)')
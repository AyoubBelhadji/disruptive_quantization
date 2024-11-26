#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .files_tools import *
#from tqdm import tqdm

def visualize_and_save_dynamics(experiment_name, c_array_trajectory, data_array):
    R,T,M,d = c_array_trajectory.shape
    animations = []  # To store animations for each repetition
    
    for r in range(R):
        fig, ax = plt.subplots()
    
        # def update(frame, current_R):
        #     ax.clear()
            
        #     ax.set_xlim([-15, 28])
        #     ax.set_ylim([-15, 28])
            
        #     # Plot static positions
        #     ax.scatter(data_array[:, 0], data_array[:, 1], color='black', label='target distribution')
        #     # Plot trajectories for each particle at this frame
        #     for k in range(M):
        #         ax.scatter(c_array_trajectory[current_R, frame, k, 0], c_array_trajectory[current_R, frame, k, 1], color='red', 
        #                    label=f'Particle {k}' if frame == 0 else "")
        #     ax.set_title(f'Repetition {current_R + 1}, Time step {frame + 1}')
        #     ax.legend()
    
    
    
        def animate(t, current_R):
            ax.clear()
            
            centroids_0 = c_array_trajectory[r,0, :, :]
            ## Change the following: the windows size should be in the config file
            
            ax.set_xlim([-15, 28])
            ax.set_ylim([-15, 28])
            # Plot all data points
            ax.scatter(data_array[:, 0], data_array[:, 1], color='black', alpha=0.5)
            
            # Plot initial centroids
            ax.scatter(centroids_0[:, 0], centroids_0[:, 1], color='red', alpha=0.5, label='Initial Centroids')
            
            # Plot moving centroids for frame t
            ax.scatter(list(c_array_trajectory[current_R,t, :, 0]), list(c_array_trajectory[current_R,t, :, 1]), color='green', alpha=1, label='Centroids')
            
            ax.set_title('Iteration t='+str(t))
            
            #ax.set_xlabel('X1')
            #ax.set_ylabel('X2')
            ax.legend()
            
            
        # Animate for the first repetition (adjust if you want all repetitions)
        #current_R = 0  # Focus on the first repetition
        ani = FuncAnimation(fig, animate, frames=T, interval=200, fargs=(r,))
        #experiment_name = "experiment_name"
        
        folder_name = "figures/"+experiment_name+"/gif/"
        
        create_folder_if_needed(folder_name)
        
        # Save animation as a GIF
        gif_path = folder_name + "particle_evolution_"+str(r)+".gif"
        ani.save(gif_path, writer=PillowWriter(fps=10))
        plt.close(fig)  # Close the figure to avoid displaying static plots

    return gif_path

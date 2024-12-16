#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:26:17 2023

@author: ayoubbelhadji
"""


import matplotlib.pyplot as plt
import numpy as np

#### Image tools
    
def show_image(x,d_1,d_2):
    f = plt.figure()
    y = x.reshape((d_1,d_2))
    plt.imshow(y,cmap='gray')
    
def show_list_of_images(x_list,d_1,d_2):
    L = len(x_list)
    for l in list(range(L)):
        print(l)
        x = x_list[l]
        f = plt.figure()
        y = x.reshape((d_1,d_2))
        plt.imshow(y,cmap='gray')
        #f.savefig("image"+str(l)+".pdf", bbox_inches='tight')



def show_list_of_images_sim2(x_lists, d_1, d_2):
    L = len(x_lists[0])  # Assuming all x_lists have the same length
    k = len(x_lists)  # Number of lists (e.g., number of images per row)

    fig, axes = plt.subplots(L, k, figsize=(15, L * 3))
    
    for l in range(L):
        for idx, x_list in enumerate(x_lists):
            x = x_list[l]
            y = x.reshape((d_1, d_2))
            
            if L == 1:  # If only one image, handle as 1D axes
                ax = axes[idx]
            elif k == 1:  # If only one list, handle as 1D axes
                ax = axes[l]
            else:
                ax = axes[l, idx]

            ax.imshow(y, cmap='gray')
            ax.axis('off')  # Turn off axis for better visualization

    plt.tight_layout()
    plt.show()



def show_list_of_images_sim(x_lists, d_1, d_2):
    L = len(x_lists[0])  # Assuming all x_lists have the same length
    k = len(x_lists)  # Number of lists (this will now be the number of rows)
    
    fig, axes = plt.subplots(k, L, figsize=(L * 3, k * 3))  # Flip dimensions for 90-degree rotation
    
    for idx, x_list in enumerate(x_lists):
        for l in range(L):
            x = x_list[l]
            y = x.reshape((d_1, d_2))
            
            if k == 1:  # Handle the case if there is only one list (1D axes)
                ax = axes[l]
            elif L == 1:  # Handle the case if there is only one image per list (1D axes)
                ax = axes[idx]
            else:
                ax = axes[idx, l]  # Transpose the layout (rows for lists, columns for images)

            ax.imshow(y, cmap='gray')
            ax.axis('off')  # Turn off axis for better visualization

    plt.tight_layout()
    plt.show()
    
    
from matplotlib.animation import FuncAnimation

def animate_images_(x_lists, d_1, d_2):
    L = len(x_lists[0])  # Assuming all x_lists have the same length
    k = len(x_lists)  # Number of lists (number of images per row)

    fig, axes = plt.subplots(L, k, figsize=(k * 3, L * 3))

    def update(frame):
        for l in range(L):
            for idx, x_list in enumerate(x_lists):
                x = x_list[l]
                y = x.reshape((d_1, d_2))
                #if frame % 2 == 0:
                    # Original image
                #    y = x.reshape((d_1, d_2))
                # else:
                #     # 90-degree rotated image
                #     y = np.rot90(x.reshape((d_1, d_2)))

                # Update the corresponding axes
                if L == 1:  # If only one row, handle axes as 1D
                    ax = axes[idx]
                elif k == 1:  # If only one list, handle axes as 1D
                    ax = axes[l]
                else:
                    ax = axes[l, idx]

                ax.imshow(y, cmap='gray')
                ax.axis('off')  # Turn off axis for better visualization

    # Create the animation
    ani = FuncAnimation(fig, update, frames=20, interval=500)

    plt.tight_layout()
    plt.show()

    # Save the animation as a GIF
    ani.save("image_flip_animation.gif", writer='imagemagick', fps=2)
    
    
def animate_images__(x_lists, d_1, d_2):
    L = len(x_lists[0])  # Number of images in each list
    k = len(x_lists)  # Number of lists (e.g., number of rows in the plot)
    
    fig, axes = plt.subplots(1, k, figsize=(k * 3, L * 3))

    def update(frame):
        print(frame)
        for l in range(k):
            x_list = x_lists[l]
            for idx, x_list in enumerate(x_lists):
                x = x_list[frame]
                y = x.reshape((d_1, d_2))
                # # Show original in even iterations, rotated in odd iterations
                # if frame % 2 == 0:
                #     y = x.reshape((d_1, d_2))  # Original image
                # else:
                #     y = np.rot90(x.reshape((d_1, d_2)))  # 90-degree rotated image

                # Update the corresponding axes
                if L == 1:  # If only one row, handle axes as 1D
                    ax = axes[idx]
                elif k == 1:  # If only one list, handle axes as 1D
                    ax = axes[l]
                else:
                    ax = axes[l, idx]

                ax.imshow(y, cmap='gray')
                ax.axis('off')  # Turn off axis for better visualization

    # Create the animation
    ani = FuncAnimation(fig, update, frames=L-1, interval=1)

    plt.tight_layout()
    plt.show()

    # Save the animation as a GIF
    ani.save("iteration_animation_.gif", writer='imagemagick', fps=1)




def animate_images(x_lists,filepath, d_1=28, d_2=28, iterations=10):
    L = len(x_lists[0])  # Number of images per list (assuming all lists have the same length)
    k = len(x_lists)  # Number of lists (number of images to be shown simultaneously)
    
    fig, axes = plt.subplots(1, k, figsize=(k * 3, 3))  # Create a single row of subplots for k images

    # Initialize the image plots
    ims = []
    for idx in range(k):
        ax = axes[idx] if k > 1 else axes  # Handle the case when k=1
        im = ax.imshow(np.zeros((d_1, d_2)), cmap='gray', vmin=0, vmax=1)  # Placeholder for the images
        ax.axis('off')  # Turn off axis for better visualization
        ims.append(im)

    def update(frame):
        for idx, x_list in enumerate(x_lists):
            # Get the image for the current frame from each x_list
            image = x_list[frame % L]  # Iterate over the available images in each list
            ims[idx].set_data(image.reshape((d_1, d_2)))  # Update the image data for the subplot

    # Create the animation
    ani = FuncAnimation(fig, update, frames=iterations, interval=500)

    plt.tight_layout()
    plt.show()

    # Save the animation as a GIF
    ani.save(filepath, writer='imagemagick', fps=1)
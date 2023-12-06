import os
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import matplotlib.gridspec as gridspec

def display_images_in_grid(images, titles, rows, cols, figsize=(6, 6), to_save=False, save_filename=None):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Create a gridspec
    #  gs = gridspec.GridSpec(rows, cols, width_ratios=[1]*cols, height_ratios=[1]*rows)

    # Reduce the spacing between subplots
    plt.subplots_adjust(wspace=0.0002, hspace=-0.04)
    #  gs.update(wspace=0.000002, hspace=-0.04)

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            img_path = os.path.join(GRAPHICS_DIR, images[i * cols + j])
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            #  ax.set_title(titles[i * cols + j])

            if i == 0:
                ax.set_title(proportions[j])
            if j == 0:
                ax.text(-0.05, 0.5, row_names[i], rotation=90, ha='center', va='center', transform=ax.transAxes)

    if to_save and save_filename:
        plt.savefig(save_filename)
    plt.show()


GRAPHICS_DIR = "MedicamentResistanceODE/graphics/"

# Specify the image file names
medicament = "Docetaxel"
ode_images = [f"ode_{medicament}_10_90.png", f"ode_{medicament}_50_50.png", f"ode_{medicament}_90_10.png"]
cancer_images = [f"cancer_{medicament}_10_90.png", f"cancer_{medicament}_50_50.png", f"cancer_{medicament}_90_10.png"]
proportions = ["10:90", "50:50", "90:10"]
row_names = ['Équations Différentielles', 'Croissance des cellules cancérigènes']
#  row_names = ['ODE', 'Cancer']

# Specify the titles for each subplot
ode_titles = ["ODE 10_90", "ODE 50_50", "ODE 90_10"]
cancer_titles = ["Cancer 10_90", "Cancer 50_50", "Cancer 90_10"]

# Display and save the combined figure
save_filename=os.path.join(GRAPHICS_DIR, "{medicament}_combined_figure.png")
display_images_in_grid(ode_images + cancer_images, ode_titles + cancer_titles, 2, 3, to_save=True, save_filename=save_filename)


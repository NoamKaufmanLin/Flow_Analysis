import datetime
import traceback
import numpy as np
from itertools import product
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm


class Analysis:

    @staticmethod
    def single_variable_histogram(data, var, bins=30, save=True):
        # Plotting the distribution of 'flow_avg'
        plt.figure(figsize=(10, 6))
        sns.histplot(data[var], bins=bins, kde=True)
        plt.title(f'Distribution of {var}')
        plt.xlabel(f'{var}')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        # Summary statistics of 'flow_avg'
        print(data[var].describe())

    @staticmethod
    def scatterplot(data, vars, subvars, target_var, save=True):
        # Creating scatter plots to visualize relationships between variables and 'flow_avg'
        fig, axes = plt.subplots(1, len(vars), figsize=(18, 5))

        # Define a colormap based on subvar values
        num_subvars = len(data[subvars[0]].unique())
        cmap = plt.cm.get_cmap('viridis', num_subvars)  # Adjust 'viridis' to change the colormap

        for i in range(len(vars)):
            # Scatter plot for vars[i] vs target_var
            sns.scatterplot(ax=axes[i], data=data, x=vars[i], y=target_var, hue=subvars[i], palette=cmap, alpha=0.3)
            axes[i].set_title(f'{target_var} vs {vars[i]}')
            axes[i].set_xlabel(vars[i])
            axes[i].set_ylabel(target_var)
            axes[i].grid(True)

            # Plot lines connecting points with the same subvar
            for size, color in zip(data[subvars[i]].unique(), cmap.colors):
                subset = data[data[subvars[i]] == size].sort_values(by=vars[i])
                axes[i].plot(subset[vars[i]], subset[target_var], alpha=0.5, color=color)

            # Add legend outside the plot area
            handles, labels = axes[i].get_legend_handles_labels()
            axes[i].legend(handles, labels, title=subvars[i], loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def heatmap(data, var1, var2, target_var, save=True):
        # Pivot the DataFrame to prepare for heatmap
        heatmap_data = data.pivot_table(values=target_var, index=var1, columns=var2, aggfunc=np.mean)

        # Generate the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=False, cmap='coolwarm')  # 'annot' annotates the values
        plt.title(f'Heatmap of {target_var} by {var1} and {var2}')
        plt.show()


if __name__ == '__main__':
    # Load the data from the uploaded CSV file
    data_path = "10_10_10_constant20_stride2.csv"  # 20_20_constant100, 40_40_constant100, 10_10_10_constant100
    data = pd.read_csv(data_path)

    # Display the first few rows of the dataframe to understand its structure
    print(data.head())

    analyzer = Analysis()
    # analyzer.single_variable_histogram(data=data, var='flow_ratio_avg', bins=20, save=True)
    # analyzer.single_variable_histogram(data=data, var='min_ratio_avg', bins=20, save=True)

    vars = ['size_2', 'size_2', 'size_3', 'size_3', 'size_4', 'size_4']
    subvars = ['size_3', 'size_4', 'size_2', 'size_4', 'size_2', 'size_3']
    # vars = ['size_2', 'size_3', 'size_4']
    analyzer.scatterplot(data=data, vars=vars, subvars=subvars, target_var='flow_ratio_avg', save=True)
    analyzer.scatterplot(data=data, vars=vars, subvars=subvars, target_var='min_ratio_avg', save=True)

    # analyzer.heatmap(data=data, var1='size_2', var2='size_3', target_var='flow_ratio_avg', save=True)
    # analyzer.heatmap(data=data, var1='size_2', var2='size_3', target_var='min_ratio_avg', save=True)
    # print()


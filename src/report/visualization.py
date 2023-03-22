import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def snsplt(train_df):
    plt.figure(figsize=(10, 10))
    sns.set_style("darkgrid")
    # Plot the countplot with the ordered categories
    label_series = pd.Series(train_df['label'])
    return sns.countplot(x=label_series)
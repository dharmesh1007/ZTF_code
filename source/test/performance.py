def plot_roc(y_train, y_test, y_prob, label_encoder, path):

    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import roc_curve, auc, RocCurveDisplay
    from itertools import cycle
    import matplotlib.pyplot as plt
    import numpy as np


    # One hot encode the class labels
    # y_onehot_test.shape (n_samples, n_classes)
    # we use a LabelBinarizer to binarize the target by one-hot-encoding in a OvR fashion. 
    # This means that the target of shape (n_samples,) is mapped to a target of shape (n_samples, n_classes)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test) 


    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()


    # Compute micro-average ROC curve and ROC area
    # ROC curve using micro-averaged OvR
    # Micro-averaging aggregates the contributions from all the classes (using np.ravel) to compute the average metrics as follows:
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    n_classes = label_encoder.classes_.shape[0]

    # Store the fpr, tpr and roc_auc for each class vs the remaining classes.
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    # Threshold values of probability
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # Average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # Construct plots
    fig, ax = plt.subplots(figsize=(8, 8))



    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["red", "orange", "grey", "lightgreen", "green", "cyan", "blue", "indigo", "violet", "darkgrey"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_prob[:, class_id],
            name=f"{label_encoder.inverse_transform(np.unique(y_train))[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) curves \nOne-vs-Rest multiclass")
    plt.legend()
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.show()



def plot_cv_results(cv_results, n_folds, n_models, path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # To generate boxplots, first filter the columns to those of interest.
    cv_results_trunc = cv_results.filter(regex='^split')

    # How many models do we wish to show in the plot?
    cv_results_trunc = cv_results_trunc.iloc[0:n_models, :]

    # Displaying results in a box plot requires some columns as rows and some rows as columns.
    # This requires using the transpose and melt functions.
    cv_results_transpose = cv_results_trunc.transpose().reset_index().rename(columns={'index':'metric'})
    for i in range(n_folds):
        cv_results_transpose['metric'] = cv_results_transpose['metric'].str.replace(f'split{i}_test_', '')

    cv_results_transpose.columns = cv_results_transpose.columns.map(str)

    cv_results_melt=pd.melt(cv_results_transpose, id_vars=['metric'],var_name='model',value_name='value')

    # Boxplots of each parameter combination.
    plt.figure(figsize=(10,5))
    for i in range(0, n_models, 2):
        plt.axvspan(i-.5, i+.5, facecolor='grey', alpha=0.1)

    sns.boxplot(x="model", y="value", hue="metric", data=cv_results_melt, showmeans=True, palette='rocket')
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.show()



# Plot feature importances
def plot_feature_importance(importance,names,model_type, path, top_n=20):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df = fi_df.iloc[0:top_n,:]

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.savefig(path, format='pdf', bbox_inches='tight')


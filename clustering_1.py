from mnist_data_prep import get_mnist_data_labels_neural
from census_data_prep import get_census_data_and_labels
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from matplotlib import pyplot as plt

SEED = 1
figure_directory = "Document/Figures/working/"


def find_K(dataset_name, X, max_k=16):
    scores = []
    K_list = list(range(2, max_k))
    for K in K_list:
        fig, ax = plt.subplots()
        # fig.suptitle(title, fontsize=16)
        model = KMeans(K, random_state=SEED)
        visualizer = SilhouetteVisualizer(model, colors="yellowbrick", ax=ax)
        visualizer.fit(X)
        visualizer.show(outpath=f"{figure_directory}{dataset_name}_silhouette_K_{K}.png")
        print(f"K:{K:2}  Silhouette Score:{visualizer.silhouette_score_}")
        scores.append(visualizer.silhouette_score_)

    print(K_list)
    print(scores)

    return K_list, scores


print("Loading Data")
(
    train_images_flattened,
    train_one_hot_labels,
    train_labels,
    test_images_flattened,
    test_one_hot_labels,
    test_labels,
) = get_mnist_data_labels_neural()

find_K("mnist", train_images_flattened)

(
    df_data,
    df_label,
    np_data_numeric,
    np_label_numeric,
    df_test_data,
    df_test_label,
    np_test_data_numeric,
    np_test_label_numeric,
    data_classes,
) = get_census_data_and_labels()

find_K("census", np_data_numeric, max_k=25)

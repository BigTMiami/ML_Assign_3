from mnist_data_prep import get_mnist_data_labels_neural
from census_data_prep import get_census_data_and_labels
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer
from matplotlib import pyplot as plt
from time import time
from charting import chart_pca_scree

SEED = 1
figure_directory = "Document/Figures/working/"


def find_K(dataset_name, X, max_k, start_k=2):
    scores = []
    K_list = list(range(start_k, max_k))
    fig, ax = plt.subplots()
    for K in K_list:
        fig.suptitle(f"{dataset_name} K Means Silhouette", fontsize=16)
        model = KMeans(K, random_state=SEED)
        visualizer = SilhouetteVisualizer(model, colors="yellowbrick", ax=ax)
        visualizer.fit(X)
        visualizer.show(outpath=f"{figure_directory}{dataset_name}_silhouette_K_{K}.png")
        print(f"K:{K:2}  Silhouette Score:{visualizer.silhouette_score_}")
        scores.append(visualizer.silhouette_score_)
        plt.clf()

    plt.close()
    print(K_list)
    print(scores)

    return K_list, scores


def find_EM(X, max_k, start_k=2):
    start_time = time()
    bic_scores = {
        "spherical": {"k": [], "bic_score": []},
        "tied": {"k": [], "bic_score": []},
        "diag": {"k": [], "bic_score": []},
        "full": {"k": [], "bic_score": []},
    }
    for k in range(start_k, max_k):
        print(f"Number of Components:{k}")
        for covariance_type in bic_scores:
            em = mixture.GaussianMixture(n_components=k, covariance_type=covariance_type)
            em.fit(X)
            score = em.bic(X)
            bic_scores[covariance_type]["k"].append(k)
            bic_scores[covariance_type]["bic_score"].append(score)
            print(f"    {covariance_type:>9}:{score:10.0f}")

    end_time = time()
    print(bic_scores)
    print(f"Time:{end_time-start_time:6.2f} seconds")
    return bic_scores


def find_PCA(train_data, test_data, sup_title, variance_threshold=0.99):
    start_time = time()
    pca = PCA(n_components=variance_threshold)
    print("Fitting Data")
    pca.fit(train_data)
    print(f"    Time:{time()-start_time:.0f} seconds")
    start_time = time()
    print("Charting Data")
    chart_pca_scree(pca, "Scree", sup_title, variance_threshold)
    print(f"    Time:{time()-start_time:.0f} seconds")
    start_time = time()
    print("Transforming Data")
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    print(f"    Time:{time()-start_time:.0f} seconds")
    print(f"Total components:{pca.n_components_} for {variance_threshold*100}% variance")
    return train_data, test_data

from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
from matplotlib import pyplot as plt
from time import time
from charting import chart_pca_scree, line_chart, clean_string, chart_bic_scores
import json
import warnings

SEED = 1
figure_directory = "Document/Figures/working/"
results_file = "clustering_results.py"


def save_results(variable_name, results):
    with open(results_file, "a") as f:
        f.write("\n")
        f.write(f"{clean_string(variable_name)} = {json.dumps(results)}")
        f.write("\n")


def find_K_elbow(dataset_name, X, max_k, start_k=2):
    clean_dataset_name = clean_string(dataset_name)
    print(f"Starting Elbow for {dataset_name}, ({start_k},{max_k})")
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(start_k, max_k))
    visualizer.fit(X)
    visualizer.show(
        outpath=f"{figure_directory}{clean_dataset_name}_elbow_max_{max_k}_best_{visualizer.elbow_value_}.png"
    )
    print(f"Best k:{visualizer.elbow_value_}")
    return


def find_K(dataset_name, X, max_k, start_k=2):
    clean_dataset_name = clean_string(dataset_name)
    scores = []
    K_list = list(range(start_k, max_k))
    for K in K_list:
        fig, ax = plt.subplots()
        fig.suptitle(f"{dataset_name} K Means Silhouette", fontsize=16)
        model = KMeans(K, random_state=SEED)
        visualizer = SilhouetteVisualizer(model, colors="yellowbrick", ax=ax)
        visualizer.fit(X)
        visualizer.show(outpath=f"{figure_directory}{clean_dataset_name}_silhouette_K_{K}.png")
        print(f"K:{K:2}  Silhouette Score:{visualizer.silhouette_score_}")
        scores.append(visualizer.silhouette_score_)
        plt.close()

    line_chart(K_list, "K", scores, "Sillhoette Score", dataset_name, "K Means Cluster Scores")

    print(K_list)
    print(scores)

    variable_name = f"{clean_dataset_name}_silhoette_scores"
    results = {"K_list": K_list, "scores": scores}
    save_results(variable_name, results)

    find_K_elbow(dataset_name, X, max_k, start_k=start_k)

    return K_list, scores


def find_EM(dataset_name, X, max_k, start_k=2):
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
    print(f"Time:{end_time-start_time:6.2f} seconds")
    print(bic_scores)

    chart_bic_scores(bic_scores, f"{dataset_name} Data", "Expecation Maximization")

    clean_dataset_name = clean_string(dataset_name)
    variable_name = f"{clean_dataset_name}_bic_scores"
    results = {"bic_scores": bic_scores}
    save_results(variable_name, results)

    return bic_scores


def find_PCA(train_data, test_data, sup_title, variance_threshold):
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
    train_reduced_data = pca.transform(train_data)
    test_reduced_data = pca.transform(test_data)
    print(f"    Time:{time()-start_time:.0f} seconds")
    print(f"Total components:{pca.n_components_} for {variance_threshold*100}% variance")
    return train_reduced_data, test_reduced_data


def find_K_means_from_PCA(dataset_name, train_data, test_data, variance_threshold, max_k):
    print(f"Finding PCA")
    sup_title = f"PCA {dataset_name} Data"
    reduced_train_data, reduced_test_data = find_PCA(
        train_data, test_data, sup_title, variance_threshold=variance_threshold
    )

    print("Finding K Means on reduced dataset using Silhouette")
    reduced_dataset_name = f"{dataset_name} Reduced PCA ({100*variance_threshold}% EV)"
    find_K(reduced_dataset_name, reduced_train_data, max_k)

    print("Finding K Means on reduced dataset using Elbow")
    find_K_elbow(reduced_dataset_name, reduced_train_data, max_k)


def find_em_from_PCA(dataset_name, train_data, test_data, variance_threshold, max_k):
    print(f"Finding PCA")
    sup_title = f"PCA {dataset_name} Data"
    reduced_train_data, reduced_test_data = find_PCA(
        train_data, test_data, sup_title, variance_threshold=variance_threshold
    )
    reduced_dataset_name = f"{dataset_name} Reduced PCA ({100*variance_threshold}% EV)"

    print("Finding EM on reduced dataset")
    find_EM(reduced_dataset_name, reduced_train_data, max_k)


def ICA_review(dataset_name, train_data, max_k, max_iter, tol, n_components):
    print(f"Fitting ICA")
    print("================================================")
    ica = FastICA(n_components=n_components, max_iter=max_iter, tol=tol)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            ica.fit(train_data)
        except Warning as e:
            print(e)
            print("ERRORED OUT")
            exit()

    print(f"Fitted in {ica.n_iter_} iterations")
    print(f"    {ica.n_features_in_} features seen in fitting")
    print(f"Transforming ICA")
    print("================================================")
    reduced_train_data = ica.transform(train_data)

    print("Finding K Means on ICA dataset using Silhouette")
    print("================================================")
    reduced_dataset_name = f"{dataset_name} ICA {n_components} components)"
    find_K(reduced_dataset_name, reduced_train_data, max_k)

    print("Finding K Means on ICA dataset using Elbow")
    print("================================================")
    find_K_elbow(reduced_dataset_name, reduced_train_data, max_k)

    print("Finding EM on ICA dataset")
    print("================================================")
    find_EM(reduced_dataset_name, reduced_train_data, max_k)


def RPA_review(dataset_name, train_data, max_k, eps, proj_type, n_components):
    print(f"Fitting RPA")
    print("================================================")
    if eps is None:
        print(f"RPA: Using n_components of {n_components} because eps not set.")
        kwargs = {"n_components": n_components}
    else:
        kwargs = {"eps": eps}

    if proj_type == "sparse":
        print("Using Sparse")
        rpa = SparseRandomProjection(**kwargs)
    elif proj_type == "gaussian":
        print("Using Gaussian")
        rpa = GaussianRandomProjection(**kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            rpa.fit(train_data)
        except Warning as e:
            print(e)
            print("ERRORED OUT")
            exit()

    print(f"Fitted {rpa.n_components_} components")

    print(f"Transforming with RPA")
    print("================================================")
    reduced_train_data = rpa.transform(train_data)

    print("Finding K Means on RPA dataset using Silhouette")
    print("================================================")
    reduced_dataset_name = f"{dataset_name} RPA Eps:{eps} to {rpa.n_components_} components)"
    find_K(reduced_dataset_name, reduced_train_data, max_k)

    print("Finding K Means on ICA dataset using Elbow")
    print("================================================")
    find_K_elbow(reduced_dataset_name, reduced_train_data, max_k)

    print("Finding EM on ICA dataset")
    print("================================================")
    find_EM(reduced_dataset_name, reduced_train_data, max_k)

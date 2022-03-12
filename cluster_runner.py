from clustering_1 import (
    find_K,
    find_EM,
    find_PCA,
    find_K_elbow,
    find_K_means_from_PCA,
    find_em_from_PCA,
    ICA_review,
    RPA_review,
)
from mnist_data_prep import get_mnist_data_labels
from census_data_prep import get_census_data_and_labels
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering Runner")
    parser.add_argument("dataset", choices=["MNIST", "Census"], help="The dataset: MNIST Census")
    parser.add_argument(
        "task",
        choices=["kmeans", "em", "pca", "elbow", "pca_kmeans", "pca_em", "ica_review", "rpa_review", "custom"],
        help="The task : kmeans, em, pca, elbow, pca_kmeans, pca_em, ica_review, rpa_review, custom",
    )
    parser.add_argument("-variance_threshold", type=float, default=0.85, help="Variance Threshold used for PAC")
    parser.add_argument("-max_k", type=int, default=10, help="Maximum K for K means or EM componenents")
    parser.add_argument("-max_iter", type=int, default=200, help="Maximum Iterations for FastICA")
    parser.add_argument("-tol", type=float, default=0.0001, help="Tolerance for FastICA")
    parser.add_argument("-n_components", type=int, default=10, help="Number of ICA components FastICA or RPA")
    parser.add_argument("-eps", type=float, help="Epsilon for Random Projection")
    parser.add_argument(
        "-proj_type", choices=["sparse", "gaussian"], default="sparse", help="Use sparse or gaussian for RPA"
    )

    args = parser.parse_args()

    if args.dataset == "MNIST":
        print("Loading MNIST Data")
        (
            train_data,
            train_labels,
            test_data,
            test_labels,
        ) = get_mnist_data_labels(scale_data=True)
    else:
        print("Loading Census Data")
        (
            census_df_data,
            census_df_label,
            train_data,
            train_labels,
            census_df_test_data,
            census_df_test_label,
            test_data,
            test_labels,
            data_classes,
        ) = get_census_data_and_labels()

    if args.task == "custom":
        print("CUSTOM TASKS NOT SET")
    elif args.task == "rpa_review":
        if args.eps is None:
            print(f"RPA: Using n_components of {args.n_components} because eps not set.")
        RPA_review(args.dataset, train_data, args.max_k, args.eps, args.proj_type, args.n_components)
    elif args.task == "ica_review":
        ICA_review(args.dataset, train_data, args.max_k, args.max_iter, args.tol, args.n_components)
    elif args.task == "pca_em":
        find_em_from_PCA(args.dataset, train_data, test_data, args.variance_threshold, args.max_k)
    elif args.task == "pca_kmeans":
        find_K_means_from_PCA(args.dataset, train_data, test_data, args.variance_threshold, args.max_k)
    elif args.task == "pca":
        reduced_train_data, reduced_test_data = find_PCA(
            train_data, test_data, f"PCA {args.dataset} Data", args.variance_threshold
        )
    elif args.task == "kmeans":
        find_K(args.dataset, train_data, args.max_k)

    elif args.task == "em":
        find_EM(args.dataset, train_data, args.max_k)

    elif args.task == "elbow":
        find_K_elbow(args.dataset, train_data, args.max_k)
    else:
        print(f"Unsupported Task of {args.task}")

    # Beep when done
    print("\a")

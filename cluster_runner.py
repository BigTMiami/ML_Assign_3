from clustering_1 import find_K, find_EM, find_PCA
from mnist_data_prep import get_mnist_data_labels_neural
from census_data_prep import get_census_data_and_labels
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering Runner")
    parser.add_argument("dataset", choices=["MNIST", "Census"], help="The dataset: MNIST Census")
    parser.add_argument("task", choices=["kmeans", "em", "pca"], help="The task : kmeans, em, pca")
    parser.add_argument("-variance_threshold", type=float, default=0.85, help="Variance Threshold used for PAC")
    parser.add_argument("-max_k", type=int, default=10, help="Maximum K for K means or EM componenents")

    args = parser.parse_args()

    if args.dataset == "MNIST":
        print("Loading MNIST Data")
        (
            train_data,
            train_one_hot_labels,
            train_labels,
            test_data,
            test_one_hot_labels,
            test_labels,
        ) = get_mnist_data_labels_neural()
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

    if args.task == "pca":
        reduced_train_data, reduced_test_data = find_PCA(
            train_data, test_data, f"PCA {args.dataset} Data", variance_threshold=args.variance_threshold
        )

    elif args.task == "kmeans":
        find_K(args.dataset, train_data, args.max_k)

    elif args.task == "em":
        find_EM(train_data, args.max_k)

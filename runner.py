from clustering_1 import find_K, find_EM, find_PCA
from mnist_data_prep import get_mnist_data_labels_neural
from census_data_prep import get_census_data_and_labels
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering Runner")

    if True:
        print("Loading MNIST Data")
        (
            train_images_flattened,
            train_one_hot_labels,
            train_labels,
            test_images_flattened,
            test_one_hot_labels,
            test_labels,
        ) = get_mnist_data_labels_neural()

        reduced_census_train_data, reduced_census_test_data = find_PCA(
            train_images_flattened, test_images_flattened, "PCA Census Data", variance_threshold=0.8
        )

    if False:
        print("Loading MNIST Data")
        (
            train_images_flattened,
            train_one_hot_labels,
            train_labels,
            test_images_flattened,
            test_one_hot_labels,
            test_labels,
        ) = get_mnist_data_labels_neural()

        find_K("MNIST", train_images_flattened, 25)

    if False:
        (
            census_df_data,
            census_df_label,
            census_np_data_numeric,
            census_np_label_numeric,
            census_df_test_data,
            census_df_test_label,
            census_np_test_data_numeric,
            census_np_test_label_numeric,
            data_classes,
        ) = get_census_data_and_labels()

        find_K("Census", census_np_data_numeric, 30, start_k=25)

    if False:
        print("Loading Data")
        (
            census_df_data,
            census_df_label,
            census_np_data_numeric,
            census_np_label_numeric,
            census_df_test_data,
            census_df_test_label,
            census_np_test_data_numeric,
            census_np_test_label_numeric,
            data_classes,
        ) = get_census_data_and_labels()

        find_EM(census_np_data_numeric, 30)

    if False:
        print("Loading MNIST Data")
        (
            train_images_flattened,
            train_one_hot_labels,
            train_labels,
            test_images_flattened,
            test_one_hot_labels,
            test_labels,
        ) = get_mnist_data_labels_neural()

        find_EM(train_images_flattened, 25)

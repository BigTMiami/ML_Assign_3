from charting import line_chart, chart_bic_scores
import clustering_results as cr


chart_bic_scores(cr.census_bic_scores, "Census Data", "Expecation Maximization")
chart_bic_scores(cr.mnist_bic_scores, "MNIST Data", "Expecation Maximization")

line_chart(
    cr.census_k_list, "K", cr.census_silhoette_scores, "Sillhoette Score", "Census Data", "K Means Cluster Scores"
)
line_chart(cr.mnist_k_list, "K", cr.mnist_silhoette_scores, "Sillhoette Score", "MNIST Data", "K Means Cluster Scores")

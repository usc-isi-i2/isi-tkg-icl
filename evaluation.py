from eval_utils import (
    EmptinessMetric,
    filter_predictions,
    get_index,
    load_data,
    print_metrics,
    update_metric,
)
from utils import HitsMetric, get_args

if __name__ == "__main__":
    args = get_args()

    output_data, train_data, valid_data, test_data = load_data(args)
    index = get_index(train_data, valid_data, test_data, args)

    hits_metric, empty_metric = HitsMetric(), EmptinessMetric()
    for x in output_data:
        filtered_predictions = filter_predictions(x, index, args)
        update_metric(x, filtered_predictions, hits_metric, empty_metric, args)

    print_metrics(hits_metric, empty_metric)

"""Module to display results.
Some Methods/Classes/Functions were directly taken from the jupyter notebook submission
on anomaly detection of the Intro to ML Safety course from the Center of AI Safety """

import math


def get_results_max(all_anomaly_results, model_name="normal"):
    all_anomaly_results[model_name]["max"] = [0, 0, 0, 0, 0]
    for key in all_anomaly_results[model_name].keys():
        if key != "max":
            index = 0
            for score in all_anomaly_results[model_name][key]:
                all_anomaly_results[model_name]["max"][index] = max(
                    score, all_anomaly_results[model_name]["max"][index]
                )
                index += 1


def compare_all_results(all_anomaly_results, dataloaders: dict):
    for model_name in all_anomaly_results:
        to_be_printed = " " * (25 - len(model_name)) + model_name
        dataset_names = [name for name, _ in dataloaders[1:]] + ["AVG"]
        for name in dataset_names:
            to_be_printed += " | " + " " * (6 - math.ceil(len(name) / 2)) + name + " " * (6 - math.floor(len(name) / 2))

        print(to_be_printed)
        print("=" * (25 + len(dataset_names) * 15))

        get_results_max(all_anomaly_results, model_name=model_name)
        for key in all_anomaly_results[model_name].keys():
            if key != "max":
                to_be_printed = " " * (25 - len(key)) + key
                index = 0
                for result in all_anomaly_results[model_name][key]:
                    if all_anomaly_results[model_name]["max"][index] == result:
                        result = "*" + "{:.2f}".format(round(result * 100, 2)) + "%"
                    else:
                        result = "{:.2f}".format(round(result * 100, 2)) + "%"
                    to_be_printed += (
                        " | "
                        + " " * (6 - math.ceil(len(result) / 2))
                        + result
                        + " " * (6 - math.floor(len(result) / 2))
                    )
                    index += 1
                print(to_be_printed)
        print()

    print("\n* highlights the maximum AUROC Score for an OOD Dataset")

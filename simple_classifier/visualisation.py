import pandas
import plotly.express
import numpy as np
import plotly.graph_objects as go

def PR_cruve(confidences, dataset):
    df = __prepare_dataframe(confidences, dataset)
    area = __calculate_area(df["precision"], df["recall"])
    fig = go.Figure(plotly.express.line(data_frame=df, x=df["recall"], y=df["precision"], custom_data=['accuracy', 'f1score'], title=str(area)),)
    fig.update_traces(
        hovertemplate="<br>".join([
            "recall: %{x}",
            "precision: %{y}",
            "accuracy: %{customdata[0]}",
            "F1-score: %{customdata[1]}",
        ])
    )

    fig.show()

def __calculate_area(presicions, recalls):
    area = 0
    for i in range(1, len(recalls)):
        area += (recalls[i] - recalls[i-1]) * presicions[i]
    return area

def __prepare_dataframe(confidences, dataset):
    precisions, recalls, accuracies, f1scores = __calculate_args(dataset["class"], confidences, np.unique(confidences))
    precisions, recalls, accuracies, f1scores = __removeBadPRPoints(precisions, recalls, accuracies, f1scores)
    return pandas.DataFrame({
        'precision': precisions,
        'recall' : recalls,
        'accuracy': accuracies,
        'f1score' : f1scores,
    })

def __calculate_args(originalClasses, confidences, taus):
    precisions = []
    recalls = []
    accuracies = []
    f1scores = []

    for tau in taus:
        TP, TN, FP, FN = 0, 0, 0, 0
        i = 0
        for confidence in confidences:
            prediction = 1 if confidence >= tau else 0
            objectClass = originalClasses[i]
            if prediction == objectClass:
                if objectClass == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if objectClass == 0:
                    FP += 1
                else:
                    FN += 1
            i += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append((TP + TN) / (TP + TN + FP + FN))
        if (precision + recall != 0):
            f1scores.append(2 * precision * recall / (precision + recall))
        else:
            f1scores.append(None)
    return precisions, recalls, accuracies, f1scores

def __removeBadPRPoints(precisions, recalls, accuracies, f1scores):
    combined = list(zip(precisions, recalls, accuracies, f1scores))
    combined.sort(key=lambda x: x[1])
    unzipped = [list(t) for t in zip(*combined)]
    precisions, recalls = unzipped[0], unzipped[1]
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    precisions.insert(0, 0)
    recalls.insert(0, 0)
    accuracies.insert(0, None)
    f1scores.insert(0, None)
    precisions.append(0)
    recalls.append(1)
    accuracies.append(None)
    f1scores.append(None)
    return precisions, recalls, accuracies, f1scores

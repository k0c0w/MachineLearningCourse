import numpy as np
from plotly.graph_objects import Figure, Scatter
import plotly.graph_objects as go

class Visualisation():

    @staticmethod
    def visualise_predicted_trace(prediction: np.ndarray, inputs: np.ndarray, targets: np.ndarray, plot_title=''):
        target_trace = go.Scatter(x=inputs, y=targets, mode='markers', name="targets")
        predictions_trace = Scatter(x=inputs, y=prediction.reshape((prediction.size,)), name='prediction')
        figure = Figure(
            data=[target_trace, predictions_trace],
            layout=dict(title=dict(text=plot_title)))
        figure.show()

    @staticmethod
    def visualise_error(model_descriptions: list):
        x = []
        y = []
        hover = []
        for model_description in model_descriptions:
            x.append(f'M = {model_description.M}, lambda = {model_description.Lambda}')
            y.append(model_description.ErrorOnValid)
            hover.append(f'test error: {model_description.ErrorOnTest}')
        trace = go.Scatter(x=x, y=y, mode='markers+text', text=hover, textposition="bottom center")
        figure = Figure(
            data=[trace])
        figure.show()

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

class Visualisation():

    @staticmethod
    def metric_changing_per_iteration(metric_history, metric_name):
        x = np.arange(len(metric_history))
        trace = go.Scatter(x=x, y=metric_history,)
        figure = go.Figure(data=[trace], layout=dict(title=dict(text=f'{metric_name} changing')))
        figure.show()

    @staticmethod
    def visualize_most_predicted_pictures(model, inputs: np.array, targets: np.array, images: np.array):
        def __get_3_most_suitable_images(demanded_value, is_correct_values, predictions, images):
            k = predictions.shape[1]
            images = images[is_correct_values == demanded_value]
            sorted_by_confidence = np.argsort(np.sort(predictions[is_correct_class == demanded_value])[:, k-1])
            images = images[sorted_by_confidence][-3:, :]
            return images

        def show_pictures(images, title):
            for i in range(images.shape[0]):
                plt.gray()
                plt.title(title + f' {i+1}')
                plt.imshow(images[i])
                plt.show()

        classes, predictions = model(inputs)
        is_correct_class = targets == classes
        most_incorrect_images = __get_3_most_suitable_images(False, is_correct_class, predictions, images)
        most_correct_images = __get_3_most_suitable_images(True, is_correct_class, predictions, images)

        show_pictures(most_incorrect_images, "most incorrect predicted image class")
        show_pictures(most_correct_images, "most correct predicted image class")

    @staticmethod
    def visualize_plot(x, y, descriptions):
        fig, ax = plt.subplots()
        scatter = plt.scatter(x=x, y=y)
        annotation = ax.annotate(text='', xy=(0,0), xytext=(15,15), textcoords='offset points',
                                 bbox={'boxstyle': 'round', 'fc':'r'}, arrowprops={'arrowstyle': '->'})
        annotation.set_visible(False)
        def motion_hover(event):
            annotation_visible = annotation.get_visible()
            if event.inaxes == ax:
                is_contained, annotation_index = scatter.contains(event)
                if is_contained:
                    data_point_loc = scatter.get_offsets()[annotation_index['ind'][0]]
                    annotation.xy = data_point_loc
                    model_ind = int(data_point_loc[0])
                    text_label = descriptions[model_ind]
                    annotation.set_text(text_label)
                    annotation.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if annotation_visible:
                        annotation.set_visible(False)
                        fig.canvas.draw_idle()
        fig.canvas.mpl_connect('motion_notify_event', motion_hover)

        plt.xlabel("random forest model (M,L1,L2)")
        plt.ylabel("accuracy on validation sample")
        plt.show()

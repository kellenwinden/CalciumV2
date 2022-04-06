"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import importlib.resources

from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog
#from magicgui import magic_factory

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, segmentation, morphology
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import json
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os
import csv

###
"""
- Moved spike.json into napari_calciumv2 directory (next to __init__.py)
- Added:
package_data = {
    'napari_calciumv2': ['*.json'],
    }
  as an argument for setup() function in the setup.py file
"""

class calcium(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()


        self.viewer = napari_viewer

        btn = QPushButton("Analyze")
        btn.clicked.connect(self._on_click)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(btn)

        self.canvas_traces = FigureCanvas(Figure(constrained_layout=False))
        self.axes = self.canvas_traces.figure.subplots()
        self.layout().addWidget(self.canvas_traces)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_files)
        self.layout().addWidget(self.save_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        self.layout().addWidget(self.clear_btn)

        ### changed to set variables in __init__
        self.img_stack = None
        self.img_name = None
        self.labels = None
        self.label_layer = None
        self.prediction_layer = None
        self.roi_dict = None
        self.roi_signal = None
        self.roi_dff = None
        self.spike_times = None
        self.img_path = None
        self.colors = []

    def _on_click(self):
        # added self.filename and added self. for most of the variables
        self.img_stack = self.viewer.layers[0].data
        self.img_name = self.viewer.layers[0].name
        self.img_path = self.viewer.layers[0].source.path
        img_size = self.img_stack.shape[1]

        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, f'unet_calcium_{img_size}.hdf5')

        self.model_unet = load_model(path, custom_objects={"K": K})
        background_layer = 0
        minsize = 100
        self.labels, self.label_layer, self.roi_dict = self.segment(self.img_stack, minsize, background_layer) ### added label_layer variable

        self.roi_signal = self.calculate_ROI_intensity(self.roi_dict, self.img_stack)
        self.roi_dff = self.calculateDFF(self.roi_signal)

        spike_templates_file = 'spikes.json'
        self.spike_times = self.find_peaks(self.roi_dff, spike_templates_file, 0.85)

        self.plot_values(self.roi_dff, self.labels, self.label_layer, self.spike_times)

        print('ROI areas:', self.get_ROI_area(self.roi_dict))
        print('ROI average prediction:', self.get_ROI_prediction(self.roi_dict, self.prediction_layer.data))

    def segment(self, img_stack, minsize, background_label):
        img_norm = np.max(img_stack,axis=0)/np.max(img_stack)
        img_predict = self.model_unet.predict(img_norm[np.newaxis,:,:])[0,:,:]
        self.prediction_layer = self.viewer.add_image(img_predict, name='Prediction')
        th = filters.threshold_otsu(img_predict)
        img_predict_th = img_predict > th
        img_predict_filtered_th = morphology.remove_small_objects(img_predict_th, min_size=minsize)
        distance = ndi.distance_transform_edt(img_predict_filtered_th)
        distance_smooth = filters.gaussian(distance, sigma=10)
        labels = segmentation.watershed(-distance_smooth, mask=img_predict_th)
        roi_dict = self.getROIpos(labels, background_label)
        label_layer = self.viewer.add_labels(labels, name='Segmentation', opacity=1)

        return labels, label_layer, roi_dict ### added label_layer as return value

    def getROIpos(self, labels, background_label):
        u_labels = np.unique(labels)
        roi_dict = {}
        for u in u_labels:
            roi_dict[u.item()] = []

        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                roi_dict[labels[x,y]].append([x,y])

        del roi_dict[background_label]

        return roi_dict

    def get_ROI_area(self, roi_dict):
        area = {}
        for r in roi_dict:
            area[r] = len(roi_dict[r])
        return area

    def get_ROI_prediction(self, roi_dict, prediction):
        avg_pred = {}
        for r in roi_dict:
            roi_coords = np.array(roi_dict[r]).T.tolist()
            avg_pred[r] = np.mean(prediction[tuple(roi_coords)])
        return avg_pred

    def calculate_ROI_intensity(self, roi_dict, img_stack):
        f = {}
        for r in roi_dict:
            f[r] = np.zeros(img_stack.shape[0])
            roi_coords = np.array(roi_dict[r]).T.tolist()
            for z in range(img_stack.shape[0]):
                img_frame = img_stack[z,:,:]
                f[r][z] = np.mean(img_frame[tuple(roi_coords)])
        return f

    def calculateDFF(self, roi_signal):
        dff = {}
        for n in roi_signal:
            background = self.calculate_background(roi_signal[n],100)
            dff[n] = (roi_signal[n] - background)/background
        return dff

    def calculate_background(self,f,window):
        background = np.zeros_like(f)
        background[0] = f[0]
        for y in range(1,len(f)):
            x = y - window
            if x < 0:
                x = 0
            lower_quantile = f[x:y] <= np.median(f[x:y])
            background[y] = np.mean(f[x:y][lower_quantile])
        return background

    def plot_values(self, dff, labels, layer, spike_times):  ### added labels parameter
        ### added this section:
        for i in range(1, np.max(labels) + 1):
            color = layer.get_color(i)
            color = (color[0], color[1], color[2], color[3])
            self.colors.append(color)
        self.axes.set_prop_cycle(color=self.colors)
        ###

        dff_max = np.zeros(len(dff))
        for dff_index, dff_key in enumerate(dff):
            dff_max[dff_index] = np.max(dff[dff_key])
        height_increment = max(dff_max)

        for height_index, d in enumerate(dff):
            self.axes.plot(dff[d] + height_index * (1.2 * height_increment))
            if len(spike_times[d]) > 0:
                self.axes.plot(spike_times[d],dff[d][spike_times[d]] + height_index * (1.2 * height_increment),
                               ms=2, color='k', marker='o', ls='')
            self.canvas_traces.draw_idle()

    def find_peaks(self, roi_dff, template_file, spk_threshold):
        # f = open(template_file) ### replaced this line with the following:
        f = importlib.resources.open_text(__package__, template_file)
        spike_templates = json.load(f)
        f.close() ### type(f) is now <class '_io.TextIOWrapper'>, it's a typing.TextIO instance. Do you need to close it?

        spike_times = {}
        for r in roi_dff:
            m = np.zeros((len(roi_dff[r]),len(spike_templates)))
            for spike_template_index, spk_temp in enumerate(spike_templates):
                for i in range((len(roi_dff[r])-len(spike_templates[spk_temp])+1)):
                    p = np.corrcoef(roi_dff[r][i:(i+len(spike_templates[spk_temp]))],spike_templates[spk_temp])
                    m[i, spike_template_index] = p[0,1]

            spike_times[r] = []
            spike_correlations = np.max(m,axis=1)
            j = 0
            while j < len(spike_correlations):
                if spike_correlations[j] > spk_threshold:
                    s_max = j
                    while spike_correlations[j+1] > spk_threshold:
                        if spike_correlations[j + 1] > spike_correlations[s_max]:
                            s_max = j + 1
                        j += 1
                    if np.max(roi_dff[r][s_max:(s_max + 20)]) > 0.01:
                        spike_times[r].append(s_max)
                j += 1

        return spike_times

    def save_files(self):
        save_path = self.img_path[0:-4]
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        raw_signal = np.zeros([len(self.roi_signal[list(self.roi_signal.keys())[0]]),len(self.roi_signal)])
        for i, r in enumerate(self.roi_signal):
            raw_signal[:,i] = self.roi_signal[r]

        with open(save_path + '/raw_signal.csv', 'w') as signal_file:
            writer = csv.writer(signal_file)
            writer.writerow(self.roi_signal.keys())
            for i in range(raw_signal.shape[0]):
                writer.writerow(raw_signal[i,:])

        dff_signal = np.zeros([len(self.roi_dff[list(self.roi_dff.keys())[0]]),len(self.roi_dff)])
        for i, r in enumerate(self.roi_dff):
            dff_signal[:,i] = self.roi_dff[r]

        with open(save_path + '/dff.csv', 'w') as dff_file:
            writer = csv.writer(dff_file)
            writer.writerow(self.roi_dff.keys())
            for i in range(dff_signal.shape[0]):
                writer.writerow(dff_signal[i,:])

        with open(save_path + '/spike_times.json', 'w') as spike_file:
            json.dump(self.spike_times, spike_file, indent="")

        self.canvas_traces.print_png(save_path + '/traces.png')

        label_array = np.stack((self.label_layer.data,)*4, axis=-1).astype(float)
        for i in range(1, np.max(self.labels) + 1):
            i_coords = np.asarray(label_array == [i, i, i, i]).nonzero()
            label_array[(i_coords[0], i_coords[1])] = self.colors[i - 1]
        # zero_coords = np.asarray(label_array == [0, 0, 0, 0]).nonzero()
        # label_array[(zero_coords[0], zero_coords[1])] = [0, 0, 0, 1]
        roi_layer = self.viewer.add_image(label_array, name='roi_image', visible=False)
        roi_layer.save(save_path + '/ROIs.png')

        """
        non_zero_coords = self.label_layer.data.nonzero()
        self.label_layer.data[non_zero_coords] = 1
        self.label_layer.save(save_path + '/ROIs.png')
        """

        roi_centers = {}
        for roi_number, roi_coords in self.roi_dict.items():
            center = np.mean(roi_coords, axis=0)
            roi_centers[roi_number] = (int(center[0]), int(center[1]))

        with open(save_path + '/roi_centers.json', 'w') as roi_file:
            json.dump(roi_centers, roi_file, indent="")

        self.prediction_layer.save(save_path + '/prediction.tif')

    def clear(self):
        i = len(self.viewer.layers) - 1
        while i >= 0:
            self.viewer.layers.pop(i)
            i -= 1

        self.img_stack = None
        self.img_name = None
        self.labels = None
        self.label_layer = None
        self.prediction_layer = None
        self.roi_dict = None
        self.roi_signal = None
        self.roi_dff = None
        self.spike_times = None
        self.img_path = None
        self.colors = []

        self.axes.cla()
        self.canvas_traces.draw_idle()

#@magic_factory
#def example_magic_widget(img_layer: "napari.layers.Image"):
#    print(f"you have selected {img_layer}")


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return calcium

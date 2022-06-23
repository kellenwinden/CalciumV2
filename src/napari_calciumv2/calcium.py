import importlib.resources

from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
# from magicgui import magic_factory

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, segmentation, morphology, feature
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import json
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os
import csv
import pandas as pd

"""
- Moved spike.json into napari_calciumv2 directory (next to __init__.py)
- Added:
package_data = {
    'napari_calciumv2': ['*.json', '*.hdf5'],
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

        self.img_stack = None
        self.img_name = None
        self.labels = None
        self.label_layer = None
        self.prediction_layer = None
        self.roi_dict = None
        self.roi_signal = None
        self.roi_dff = None
        self.median = None
        self.bg = None
        self.spike_times = None
        self.max_correlations = None
        self.max_cor_templates = None
        self.roi_analysis = None
        self.framerate = None
        self.mean_connect = None
        self.img_path = None
        self.colors = []
        # self.A = None

    def _on_click(self):
        self.img_stack = self.viewer.layers[0].data
        self.img_name = self.viewer.layers[0].name
        self.img_path = self.viewer.layers[0].source.path
        img_size = self.img_stack.shape[1]

        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, f'unet_calcium_{img_size}.hdf5')

        self.model_unet = load_model(path, custom_objects={"K": K})
        background_layer = 0
        minsize = 100
        self.labels, self.label_layer, self.roi_dict = self.segment(self.img_stack, minsize, background_layer)

        if self.label_layer:
            self.roi_signal = self.calculate_ROI_intensity(self.roi_dict, self.img_stack)
            self.roi_dff = self.calculateDFF(self.roi_signal)

            spike_templates_file = 'spikes.json'
            self.spike_times = self.find_peaks(self.roi_dff, spike_templates_file, 0.85, 0.80)
            self.roi_analysis, self.framerate = self.analyze_ROI(self.roi_dff, self.spike_times)
            self.mean_connect = self.get_mean_connect(self.roi_dff, self.spike_times)

            self.plot_values(self.roi_dff, self.labels, self.label_layer, self.spike_times)
            # print('ROI average prediction:', self.get_ROI_prediction(self.roi_dict, self.prediction_layer.data))

    def segment(self, img_stack, minsize, background_label):
        img_norm = np.max(img_stack, axis=0) / np.max(img_stack)
        img_predict = self.model_unet.predict(img_norm[np.newaxis, :, :])[0, :, :]

        if np.max(img_predict) > 0.3:
            self.prediction_layer = self.viewer.add_image(img_predict, name='Prediction')
            th = filters.threshold_otsu(img_predict)
            img_predict_th = img_predict > th
            img_predict_remove_holes_th = morphology.remove_small_holes(img_predict_th, area_threshold=minsize * 0.3)
            img_predict_filtered_th = morphology.remove_small_objects(img_predict_remove_holes_th, min_size=minsize)
            distance = ndi.distance_transform_edt(img_predict_filtered_th)
            local_max = feature.peak_local_max(distance,
                                               min_distance=10,
                                               footprint=np.ones((15, 15)),
                                               labels=img_predict_filtered_th)
            local_max_mask = np.zeros_like(img_predict_filtered_th, dtype=bool)
            local_max_mask[tuple(local_max.T)] = True
            markers = morphology.label(local_max_mask)
            labels = segmentation.watershed(-distance, markers, mask=img_predict_filtered_th)
            roi_dict, labels = self.getROIpos(labels, background_label)
            label_layer = self.viewer.add_labels(labels, name='Segmentation', opacity=1)
        else:
            self.general_msg('No ROI', 'There were no cells detected')
            labels, label_layer, roi_dict = None, None, None

        return labels, label_layer, roi_dict

    def getROIpos(self, labels, background_label):
        u_labels = np.unique(labels)
        roi_dict = {}
        for u in u_labels:
            roi_dict[u.item()] = []

        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                roi_dict[labels[x, y]].append([x, y])

        del roi_dict[background_label]

        # print("roi_dict len:", len(roi_dict))
        area_dict, roi_to_delete = self.get_ROI_area(roi_dict, 100)
        # print("area_dict:", area_dict)

        # delete roi in label layer and dict
        for r in roi_to_delete:
            coords_to_delete = np.array(roi_dict[r]).T.tolist()
            labels[tuple(coords_to_delete)] = 0
            roi_dict[r] = []

        # move roi in roi_dict
        for r in range(1, (len(roi_dict) - len(roi_to_delete) + 1)):
            i = 1
            while not roi_dict[r]:
                roi_dict[r] = roi_dict[r + i]
                roi_dict[r + i] = []
                i += 1

        # delete extra roi keys
        for r in range((len(roi_dict) - len(roi_to_delete) + 1), (len(roi_dict) + 1)):
            del roi_dict[r]

        # update label layer with new roi
        for r in roi_dict:
            roi_coords = np.array(roi_dict[r]).T.tolist()
            labels[tuple(roi_coords)] = r
        # print("new roi_dict len:", len(roi_dict))
        return roi_dict, labels

    def get_ROI_area(self, roi_dict, threshold):
        area = {}
        small_roi = []
        for r in roi_dict:
            area[r] = len(roi_dict[r])
            if area[r] < threshold:
                small_roi.append(r)
        return area, small_roi

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
                img_frame = img_stack[z, :, :]
                f[r][z] = np.mean(img_frame[tuple(roi_coords)])
        return f

    def calculateDFF(self, roi_signal):
        dff = {}
        self.median = {}
        self.bg = {}
        for n in roi_signal:
            background, self.median[n] = self.calculate_background(roi_signal[n], 200)
            self.bg[n] = background.tolist()
            dff[n] = (roi_signal[n] - background) / background
            dff[n] = dff[n] - np.min(dff[n])
        return dff

    def calculate_background(self, f, window):
        background = np.zeros_like(f)
        background[0] = f[0]
        median = [background[0]]
        for y in range(1, len(f)):
            x = y - window
            if x < 0:
                x = 0
            lower_quantile = f[x:y] <= np.median(f[x:y])
            background[y] = np.mean(f[x:y][lower_quantile])
            median.append(np.median(f[x:y]))
        return background, median

    def plot_values(self, dff, labels, layer, spike_times):
        for i in range(1, np.max(labels) + 1):
            color = layer.get_color(i)
            color = (color[0], color[1], color[2], color[3])
            self.colors.append(color)

        roi_to_plot = []
        colors_to_plot = []
        for i, r in enumerate(spike_times):
            if len(spike_times[r]) > 0:
                roi_to_plot.append(r)
                colors_to_plot.append(self.colors[i])

        if len(roi_to_plot) > 0:
            print('Active ROI:', roi_to_plot)
            self.axes.set_prop_cycle(color=colors_to_plot)

            dff_max = np.zeros(len(roi_to_plot))
            for dff_index, dff_key in enumerate(roi_to_plot):
                dff_max[dff_index] = np.max(dff[dff_key])
            height_increment = max(dff_max)

            for height_index, d in enumerate(roi_to_plot):
                self.axes.plot(dff[d] + height_index * (1.2 * height_increment))
                if len(spike_times[d]) > 0:
                    self.axes.plot(spike_times[d], dff[d][spike_times[d]] + height_index * (1.2 * height_increment),
                                   ms=2, color='k', marker='o', ls='')
                self.canvas_traces.draw_idle()
        else:
            self.general_msg('No activity', 'No calcium events were detected for any ROI')

    def find_peaks(self, roi_dff, template_file, spk_threshold, reset_threshold):
        f = importlib.resources.open_text(__package__, template_file)
        spike_templates = json.load(f)
        spike_times = {}
        self.max_correlations = {}
        self.max_cor_templates = {}
        max_temp_len = max([len(temp) for temp in spike_templates.values()])

        for r in roi_dff:
            # print("\n", r)
            m = np.zeros((len(roi_dff[r]), len(spike_templates)))
            roi_dff_pad = np.pad(roi_dff[r], (0, (max_temp_len - 1)), mode='constant')
            for spike_template_index, spk_temp in enumerate(spike_templates):
                for i in range(len(roi_dff[r])):
                    p = np.corrcoef(roi_dff_pad[i:(i + len(spike_templates[spk_temp]))],
                                    spike_templates[spk_temp])
                    m[i, spike_template_index] = p[0, 1]

            spike_times[r] = []
            spike_correlations = np.max(m, axis=1)
            self.max_correlations[r] = spike_correlations
            self.max_cor_templates[r] = np.argmax(m, axis=1) + 1

            j = 0
            while j < len(spike_correlations):
                if spike_correlations[j] > spk_threshold:
                    s_max = j
                    loop = True
                    # print(f'start loop at {j}')
                    while loop:
                        while spike_correlations[j + 1] > reset_threshold:
                            if spike_correlations[j + 1] > spike_correlations[s_max]:
                                s_max = j + 1
                            j += 1
                        if spike_correlations[j + 2] > reset_threshold:
                            j += 1
                        else:
                            loop = False
                    # print(f'end loop at {j} with s_max of {s_max}')
                    window_start = max(0, (s_max - 5))
                    window_end = min((len(roi_dff[r]) - 1), (s_max + 15))
                    window = roi_dff[r][window_start:window_end]
                    peak_height = np.max(window) - np.min(window)
                    # print(peak_height)
                    if peak_height > 0.02:
                        spike_times[r].append(s_max)
                j += 1

            if len(spike_times[r]) >= 2:
                for k in range(len(spike_times[r]) - 1):
                    if spike_times[r][k] is not None:
                        if (spike_times[r][k + 1] - spike_times[r][k]) <= 10:
                            spike_times[r][k + 1] = None
                spike_times[r] = [spk for spk in spike_times[r] if spk is not None]

        return spike_times

    def analyze_ROI(self, roi_dff, spk_times):
        metadata_file = self.img_path[0:-8] + '_metadata.txt'
        framerate = 0

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = f.readlines()

            for line in metadata:
                line = line.strip()
                if line.startswith('"Exposure-ms": '):
                    exposure = float(line[15:-1]) / 1000  # exposure in seconds
                    framerate = 1 / exposure  # frames/second
                    break
        # print('framerate is:', framerate, 'frames/second')

        amplitude_info = self.get_amplitude(roi_dff, spk_times)
        time_to_rise = self.get_time_to_rise(amplitude_info, framerate)
        max_slope = self.get_max_slope(roi_dff, amplitude_info)
        IEI = self.analyze_IEI(spk_times, framerate)
        roi_analysis = amplitude_info

        for r in roi_analysis:
            roi_analysis[r]['spike_times'] = spk_times[r]
            roi_analysis[r]['time_to_rise'] = time_to_rise[r]
            roi_analysis[r]['max_slope'] = max_slope[r]
            roi_analysis[r]['IEI'] = IEI[r]

        # print(roi_analysis)
        return roi_analysis, framerate

    def get_amplitude(self, roi_dff, spk_times, deriv_threhold=0.01, reset_num=17, neg_reset_num=2, total_dist=40):
        amplitude_info = {}

        # for each ROI
        for r in spk_times:
            amplitude_info[r] = {}
            amplitude_info[r]['amplitudes'] = []
            amplitude_info[r]['peak_indices'] = []
            amplitude_info[r]['base_indices'] = []

            if len(spk_times[r]) > 0:
                dff_deriv = np.diff(roi_dff[r])
                # print(f'ROI {r} spike times: {spk_times[r]}')

                # for each spike in the ROI
                for i in range(len(spk_times[r])):
                    # Search for starting index for current spike
                    searching = True
                    under_thresh_count = 0
                    total_count = 0
                    start_index = spk_times[r][i]
                    if start_index > 0:
                        while searching:
                            start_index -= 1
                            total_count += 1
                            # If collide with a new spike
                            if start_index in spk_times[r]:
                                subsearching = True
                                negative_count = 0
                                while subsearching:
                                    start_index += 1
                                    if dff_deriv[start_index] < 0:
                                        negative_count += 1
                                    else:
                                        negative_count = 0
                                    if negative_count == neg_reset_num:
                                        subsearching = False
                                break
                            if dff_deriv[start_index] < deriv_threhold:
                                under_thresh_count += 1
                            else:
                                under_thresh_count = 0
                            if under_thresh_count == reset_num or start_index == 0 or total_count == total_dist:
                                searching = False

                    # Search for ending index for current spike
                    searching = True
                    under_thresh_count = 0
                    total_count = 0
                    end_index = spk_times[r][i]
                    if end_index < (len(dff_deriv) - 1):
                        while searching:
                            end_index += 1
                            total_count += 1
                            # If collide with a new spike
                            if end_index in spk_times[r]:
                                subsearching = True
                                negative_count = 0
                                while subsearching:
                                    end_index -= 1
                                    if dff_deriv[end_index] < 0:
                                        negative_count += 1
                                    else:
                                        negative_count = 0
                                    if negative_count == neg_reset_num:
                                        subsearching = False
                                break
                            if dff_deriv[end_index] < deriv_threhold:
                                under_thresh_count += 1
                            else:
                                under_thresh_count = 0
                            if under_thresh_count == reset_num or end_index == (len(dff_deriv) - 1) or \
                                    total_count == total_dist:
                                searching = False
                    # print(f'ROI {r} spike {i} - start_index: {start_index}, end_index: {end_index}')

                    # Save data
                    spk_to_end = roi_dff[r][spk_times[r][i]:(end_index + 1)]
                    start_to_spk = roi_dff[r][start_index:(spk_times[r][i] + 1)]
                    amplitude_info[r]['amplitudes'].append(np.max(spk_to_end) - np.min(start_to_spk))
                    amplitude_info[r]['peak_indices'].append(int(spk_times[r][i] + np.argmax(spk_to_end)))
                    amplitude_info[r]['base_indices'].append(int(spk_times[r][i] -
                                                                 (len(start_to_spk) - (np.argmin(start_to_spk) + 1))))

        # for r in amplitude_info:
        #     print('ROI', r)
        #     print('amp:', amplitude_info[r]['amplitudes'])
        #     print('peak:', amplitude_info[r]['peak_indices'])
        #     print('base:', amplitude_info[r]['base_indices'])
        return amplitude_info

    def get_time_to_rise(self, amplitude_info, framerate):
        time_to_rise = {}
        for r in amplitude_info:
            time_to_rise[r] = []
            if len(amplitude_info[r]['peak_indices']) > 0:
                for i in range(len(amplitude_info[r]['peak_indices'])):
                    peak_index = amplitude_info[r]['peak_indices'][i]
                    base_index = amplitude_info[r]['base_indices'][i]
                    frames = peak_index - base_index + 1
                    if framerate:
                        time = frames / framerate  # frames * (seconds/frames) = seconds
                        time_to_rise[r].append(time)
                    else:
                        time_to_rise[r].append(frames)

        # print('time to rise:', time_to_rise)
        return time_to_rise

    def get_max_slope(self, roi_dff, amplitude_info):
        max_slope = {}
        for r in amplitude_info:
            max_slope[r] = []
            dff_deriv = np.diff(roi_dff[r])
            if len(amplitude_info[r]['peak_indices']) > 0:
                for i in range(len(amplitude_info[r]['peak_indices'])):
                    peak_index = amplitude_info[r]['peak_indices'][i]
                    base_index = amplitude_info[r]['base_indices'][i]
                    slope_window = dff_deriv[base_index:(peak_index + 1)]
                    max_slope[r].append(np.max(slope_window))

        # print('max slope:', max_slope)
        return max_slope

    def analyze_IEI(self, spk_times, framerate):
        IEI = {}
        for r in spk_times:
            IEI[r] = []
            if len(spk_times[r]) > 1:
                IEI_frames = np.mean(np.diff(np.array(spk_times[r])))
                if framerate:
                    IEI_time = IEI_frames / framerate # in seconds
                    IEI[r].append(IEI_time)
                else:
                    IEI[r].append(IEI_frames)
        # print('IEI:', IEI)
        return IEI

    def analyze_active(self, spk_times):
        active = 0
        for r in spk_times:
            if len(spk_times[r]) > 0:
                active += 1
        active /= len(spk_times)
        return active

    def get_mean_connect(self, roi_dff, spk_times):
        A = self.get_connect_matrix(roi_dff, spk_times)

        if A is not None:
            if len(A) > 1:
                mean_connect = np.mean(np.sum(A, axis=0) - 1) / (len(A) - 1)
            else:
                mean_connect = 'N/A - Only one active ROI'
        else:
            mean_connect = 'No calcium events detected'

        return mean_connect

    def get_connect_matrix(self, roi_dff, spk_times):
        active_roi = [r for r in spk_times if len(spk_times[r]) > 0]

        if len(active_roi) > 0:
            print('phases:')
            phases = {}
            for r in active_roi:
                phases[r] = self.get_phase(len(roi_dff[r]), spk_times[r])
                print(r)
                print(phases[r])

            connect_matrix = np.zeros((len(active_roi), len(active_roi)))
            for i, r1 in enumerate(active_roi):
                for j, r2 in enumerate(active_roi):
                    connect_matrix[i, j] = self.get_sync_index(phases[r1], phases[r2])
            # self.A = connect_matrix
            np.set_printoptions(linewidth=10000, edgeitems=6)
            print('A:')
            print(connect_matrix)
        else:
            connect_matrix = None

        return connect_matrix

    def get_sync_index(self, x_phase, y_phase):
        phase_diff = self.get_phase_diff(x_phase, y_phase)
        sync_index = np.sqrt((np.mean(np.cos(phase_diff)) ** 2) + (np.mean(np.sin(phase_diff)) ** 2))

        return sync_index

    def get_phase_diff(self, x_phase, y_phase):
        x_phase = np.array(x_phase)
        y_phase = np.array(y_phase)
        phase_diff = np.mod(np.abs(x_phase - y_phase), (2 * np.pi))
        # print('phase_diff', phase_diff)

        return phase_diff # Numpy array

    def get_phase(self, total_frames, spks):
        spikes = spks.copy()
        if len(spikes) == 0 or spikes[0] != 0:
            spikes.insert(0, 0)
        if spikes[-1] != (total_frames - 1):
            spikes.append(total_frames - 1)

        phase = []
        for k in range(len(spikes) - 1):
            t = spikes[k]
            while t < spikes[k + 1]:
                instant_phase = (2 * np.pi) * ((t - spikes[k]) / (spikes[k+1] - spikes[k])) + (2 * np.pi * k)
                phase.append(instant_phase)
                t += 1
        phase.append(2 * np.pi * (len(spikes) - 1))

        return phase # Python list

    def save_files(self):
        if self.roi_dict:
            save_path = self.img_path[0:-4]
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            raw_signal = np.zeros([len(self.roi_signal[list(self.roi_signal.keys())[0]]), len(self.roi_signal)])
            for i, r in enumerate(self.roi_signal):
                raw_signal[:, i] = self.roi_signal[r]

            with open(save_path + '/raw_signal.csv', 'w') as signal_file:
                writer = csv.writer(signal_file)
                writer.writerow(self.roi_signal.keys())
                for i in range(raw_signal.shape[0]):
                    writer.writerow(raw_signal[i, :])

            dff_signal = np.zeros([len(self.roi_dff[list(self.roi_dff.keys())[0]]), len(self.roi_dff)])
            for i, r in enumerate(self.roi_dff):
                dff_signal[:, i] = self.roi_dff[r]

            with open(save_path + '/dff.csv', 'w') as dff_file:
                writer = csv.writer(dff_file)
                writer.writerow(self.roi_dff.keys())
                for i in range(dff_signal.shape[0]):
                    writer.writerow(dff_signal[i, :])

            with open(save_path + '/medians.json', 'w') as median_file:
                json.dump(self.median, median_file, indent="")

            with open(save_path + '/background.json', 'w') as bg_file:
                json.dump(self.bg, bg_file, indent="")

            with open(save_path + '/spike_times.json', 'w') as spike_file:
                json.dump(self.spike_times, spike_file, indent="")

            with open(save_path + '/roi_analysis.json', 'w') as analysis_file:
                json.dump(self.roi_analysis, analysis_file, indent="")

            max_cor = np.zeros([len(self.max_correlations[list(self.max_correlations.keys())[0]]),
                                len(self.max_correlations)])
            for i, r in enumerate(self.max_correlations):
                max_cor[:, i] = self.max_correlations[r]

            with open(save_path + '/max_correlations.csv', 'w') as cor_file:
                writer = csv.writer(cor_file)
                writer.writerow(self.max_correlations.keys())
                for i in range(max_cor.shape[0]):
                    writer.writerow(max_cor[i, :])

            max_cor_temps = np.zeros([len(self.max_cor_templates[list(self.max_cor_templates.keys())[0]]),
                                      len(self.max_cor_templates)])
            for i, r in enumerate(self.max_cor_templates):
                max_cor_temps[:, i] = self.max_cor_templates[r]

            with open(save_path + '/max_cor_templates.csv', 'w') as cor_temp_file:
                writer = csv.writer(cor_temp_file)
                writer.writerow(self.max_cor_templates.keys())
                for i in range(max_cor_temps.shape[0]):
                    writer.writerow(max_cor_temps[i, :])

            self.canvas_traces.print_png(save_path + '/traces.png')

            label_array = np.stack((self.label_layer.data,) * 4, axis=-1).astype(float)
            for i in range(1, np.max(self.labels) + 1):
                i_coords = np.asarray(label_array == [i, i, i, i]).nonzero()
                label_array[(i_coords[0], i_coords[1])] = self.colors[i - 1]
            roi_layer = self.viewer.add_image(label_array, name='roi_image', visible=False)
            roi_layer.save(save_path + '/ROIs.png')

            roi_centers = {}
            for roi_number, roi_coords in self.roi_dict.items():
                center = np.mean(roi_coords, axis=0)
                roi_centers[roi_number] = (int(center[0]), int(center[1]))

            with open(save_path + '/roi_centers.json', 'w') as roi_file:
                json.dump(roi_centers, roi_file, indent="")

            self.prediction_layer.save(save_path + '/prediction.tif')

            self.generate_summary(save_path)

            # with open(save_path + '/S.csv', 'w') as s_file:
            #     writer = csv.writer(s_file)
            #     for i in range(self.s.shape[0]):
            #         writer.writerow(self.s[i, :])
        else:
            self.general_msg('No ROI', 'Cannot save data')

    def generate_summary(self, save_path):
        total_amplitude = []
        total_time_to_rise = []
        total_max_slope = []
        total_IEI = []

        for r in self.roi_analysis:
            if len(self.roi_analysis[r]['amplitudes']) > 0:
                total_amplitude.extend(self.roi_analysis[r]['amplitudes'])
                total_time_to_rise.extend(self.roi_analysis[r]['time_to_rise'])
                total_max_slope.extend(self.roi_analysis[r]['max_slope'])
            if len(self.roi_analysis[r]['IEI']) > 0:
                total_IEI.extend(self.roi_analysis[r]['IEI'])

        if any(self.spike_times.values()):
            avg_amplitude = np.mean(np.array(total_amplitude))
            avg_max_slope = np.mean(np.array(total_max_slope))
            if self.framerate:
                units = 'seconds'
            else:
                units = 'frames'
            avg_time_to_rise = np.mean(np.array(total_time_to_rise))
            avg_time_to_rise = f'{avg_time_to_rise} {units}'
            if len(total_IEI) > 0:
                avg_IEI = np.mean(np.array(total_IEI))
                avg_IEI = f'{avg_IEI} {units}'
                std_IEI = np.std(np.array(total_IEI))
            else:
                avg_IEI = 'Only one event per ROI'
        else:
            avg_amplitude = 'No calcium events detected'
            avg_max_slope = 'No calcium events detected'
            avg_time_to_rise = 'No calcium events detected'
            avg_IEI = 'No calcium events detected'
        percent_active = self.analyze_active(self.spike_times)

        with open(save_path + '/summary.txt', 'w') as sum_file:
            sum_file.write(f'File: {self.img_path}\n')
            if self.framerate:
                sum_file.write(f'Framerate: {self.framerate} frames/seconds\n')
            else:
                sum_file.write('No framerate detected\n')
            sum_file.write(f'Total ROI: {len(self.roi_dict)}\n')
            sum_file.write(f'Percent Active ROI: {percent_active}\n')
            sum_file.write(f'Average Amplitude: {avg_amplitude}\n')
            sum_file.write(f'Average Max Slope: {avg_max_slope}\n')
            sum_file.write(f'Average Time to Rise: {avg_time_to_rise}\n')
            sum_file.write(f'Average Interevent Interval (IEI): {avg_IEI}\n')
            if len(total_IEI) > 0:
                sum_file.write(f'\tIEI Standard Deviation: {std_IEI}\n')
            sum_file.write(f'Mean Global Connectivity: {self.mean_connect}')

    # Taken from napari-calcium plugin by Federico Gasparoli
    def general_msg(self, message_1: str, message_2: str):
        msg = QMessageBox()
        # msg.setStyleSheet("QLabel {min-width: 250px; min-height: 30px;}")
        msg_info_1 = f'<p style="font-size:18pt; color: #4e9a06;">{message_1}</p>'
        msg.setText(msg_info_1)
        msg_info_2 = f'<p style="font-size:15pt; color: #000000;">{message_2}</p>'
        msg.setInformativeText(msg_info_2)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

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
        self.median = None
        self.bg = None
        self.spike_times = None
        self.max_correlations = None
        self.max_cor_templates = None
        self.roi_analysis = None
        self.framerate = None
        self.mean_connect = None
        self.img_path = None
        self.colors = []
        # self.A = None

        self.axes.cla()
        self.canvas_traces.draw_idle()


# @magic_factory
# def example_magic_widget(img_layer: "napari.layers.Image"):
#    print(f"you have selected {img_layer}")


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return calcium

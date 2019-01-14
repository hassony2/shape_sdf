from collections import defaultdict, OrderedDict
from datetime import datetime
import os
import logutils
import plotly.offline as py
import plotly.tools as pytools
import plotly.graph_objs as go

args_save = ['batch_size', 'mini_factor', 'optimizer', 'sdf_point_nb', 'lr', 'momentum', 'hidden_layer_nb', 'hidden_neuron_nb']
args_corresps = {'batch_size': 'bs',
        'mini_factor': 'mf',
        'sdf_point_nb': 'sdf_pn',
        'hidden_layer_nb': 'lay_nb',
        'hidden_neuron_nb': 'neur_nb',
        'momentum': 'mom',
        'optimizer': 'optim'}

def get_save_folder(root_folder, args, args_corresps=args_corresps, args_save=args_save):
    save_params = {}
    for arg_name in args_save:
        if hasattr(args, arg_name):
            arg_val = getattr(args, arg_name)
            if arg_name in args_corresps:
                arg_name = args_corresps[arg_name]
            save_params[arg_name] = arg_val
        else:
            raise ValueError('Param {} not in args {}'.format(arg_name, args))

    now = datetime.now()
    save_string = '{}/{:02d}_{:02d}_{}'.format( args.dataset, now.year, int(now.month), int(now.day))
    for arg_name in sorted(save_params.keys()):
        save_string = '{}_{}-{}'.format(save_string, arg_name, save_params[arg_name])

    folder = os.path.join(root_folder, save_string)
    return folder


class Monitor():
    def __init__(self, checkpoint, plotly=True, hosting_folder=None):
        self.checkpoint = checkpoint
        self.train_path = os.path.join(self.checkpoint, 'train.txt')
        self.val_path = os.path.join(self.checkpoint, 'val.txt')
        logutils.create_log_file(self.train_path)
        logutils.create_log_file(self.val_path)

        self.plotly = plotly
        self.hosting_folder = hosting_folder
        os.makedirs(self.hosting_folder, exist_ok=True)
        self.metrics = Metrics(checkpoint, hosting_folder=self.hosting_folder)

    def log_train(self, epoch, errors):
        logutils.log_errors(epoch, errors, self.train_path)

    def log_val(self, epoch, errors):
        logutils.log_errors(epoch, errors, self.val_path)


class Metrics():
    def __init__(self, checkpoint, hosting_folder=None):
        self.checkpoint = checkpoint
        self.hosting_folder = hosting_folder
        self.evolution = defaultdict(lambda: defaultdict(OrderedDict))

    def save_metrics(self, epoch, metric_dict, val=False):
        for loss_name, loss_dict in metric_dict.items():
            for split_name, val in loss_dict.items():
                self.evolution[loss_name][split_name][epoch] = val

    def plot_metrics(self):
        """For plotting"""
        metric_traces = defaultdict(list)
        for loss_name, loss_dict in self.evolution.items():
            for split_name, vals in loss_dict.items():
                trace = go.Scatter(
                        x=list(vals.keys()),
                        y=list(vals.values()),
                        mode='lines',
                        name=split_name)
                metric_traces[loss_name].append(trace)

        metric_names = list(metric_traces.keys())
        fig = pytools.make_subplots(
                rows=1,
                cols=len(metric_traces),
                subplot_titles=tuple(metric_names))

        for metric_idx, metric_name in enumerate(metric_names):
            traces = metric_traces[metric_name]
            for trace in traces:
                fig.append_trace(trace, 1, metric_idx + 1)
        plotly_path = os.path.join(self.checkpoint, 'plotly.html')
        py.plot(fig, filename=plotly_path, auto_open=False)
        if self.hosting_folder is not None:
            hosted_plotly_path = os.path.join(self.hosting_folder,
                    'plotly.html')
            py.plot(fig, filename=hosted_plotly_path, auto_open=False)

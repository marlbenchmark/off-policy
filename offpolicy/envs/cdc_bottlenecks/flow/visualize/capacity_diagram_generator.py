"""Generates capacity diagrams for the bottleneck.

This method accepts as input a csv file containing the inflows and outflows
from several simulations as created by the file `examples/sumo/density_exp.py`,
e.g.

    1000, 978
    1000, 773
    1500, 1134
    ...

And then uses this data to generate a capacity diagram, with the x-axis being
the inflow rates and the y-axis is the outflow rate.

Usage
-----
::
    python capacity_diagram_generator.py </path/to/file>.csv
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.palettes import Spectral
from matplotlib import rc
import numpy as np


def import_data_from_csv(fp):
    r"""Import inflow/outflow data from the predefined csv file.

    Parameters
    ----------
    fp : string
        file path

    Returns
    -------
    dict
        "inflows": list of all the inflows \n
        "outflows" list of the outflows matching the inflow at the same index
    """
    inflows = []
    outflows = []
    with open(fp, 'rt') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            inflows.append(float(row[0]))
            outflows.append(float(row[1]))

    return {'inflows': inflows, 'outflows': outflows}

def import_rets_from_csv(fp):
    r"""Import inflow/outflow data from the predefined csv file.

    Parameters
    ----------
    fp : string
        file path

    Returns
    -------
    dict
        "inflows": list of all the inflows \n
        "outflows" list of the outflows matching the inflow at the same index
        "percent_congested":
    """
    inflows = []
    outflows = []
    velocities = []
    percent_congested = []
    bottleneckdensities = []
    with open(fp, 'rt') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            inflows.append(float(row[0]))
            outflows.append(float(row[1]))
            percent_congested.append(float(row[3]))

    return {'inflows': inflows, 'outflows': outflows, 'percent_congested':percent_congested}


def get_capacity_data(data):
    r"""Compute the unique inflows and subsequent outflow statistics.

    Parameters
    ----------
    data : dict
        "inflows": list of all the inflows \n
        "outflows" list of the outflows matching the inflow at the same index

    Returns
    -------
    as_array
        unique inflows
    as_array
        mean outflow at given inflow
    as_array
        std deviation of outflow at given inflow
    """
    unique_vals = sorted(list(set(data['inflows'])))
    sorted_outflows = {inflow: [] for inflow in unique_vals}

    for inflow, outlfow in zip(data['inflows'], data['outflows']):
        sorted_outflows[inflow].append(outlfow)

    mean = np.asarray([np.mean(sorted_outflows[val]) for val in unique_vals])
    std = np.asarray([np.std(sorted_outflows[val]) for val in unique_vals])

    return unique_vals, mean, std


def get_ret_data(data):
    r"""Compute the unique inflows and ret values.

    Parameters
    ----------
    data : dict
        "inflows": list of all the inflows \n
        "outflows" list of the outflows matching the inflow at the same index
        "percent_congested"

    Returns
    -------
    as_array
        unique inflows
    as_array
        mean congestion at given inflow
    as_array
        std deviation of congestion at given inflow
    """
    unique_vals = sorted(list(set(data['inflows'])))
    sorted_idx = {inflow: [] for inflow in unique_vals}

    for inflow, outlfow in zip(data['inflows'], data['percent_congested']):
        sorted_idx[inflow].append(outlfow)

    mean = np.asarray([np.mean(sorted_idx[val]) for val in unique_vals])
    std = np.asarray([np.std(sorted_idx[val]) for val in unique_vals])

    return unique_vals, mean, std


def create_parser():
    """Create an argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Generates capacity diagrams for the bottleneck.',
        epilog="python capacity_diagram_generator.py </path/to/file>.csv")

    parser.add_argument('file', type=str, help='path to the csv file. If you pass a folder the script will iterate'
                                               'through every file in the folder.')
    parser.add_argument('--plot_congestion', action="store_true", help='path to the csv file. If you pass a folder the script will iterate'
                                               'through every file in the folder.')
    parser.add_argument('--bokeh', action="store_true", help='Create the plot with an interactive legend in bokeh.')
    parser.add_argument('--all', action="store_true", help='Plot all files in folder')

    return parser


if __name__ == '__main__':
    # import parser arguments
    parser = create_parser()
    args = parser.parse_args()

    # some plotting parameters
    #rc('text', usetex=True)
    #font = {'weight': 'bold', 'size': 18}
    #rc('font', **font)
    plt.figure(figsize=(27, 9))
    p = figure(plot_width=1600, plot_height=1000)
    # import the csv file
    if os.path.isdir(args.file):
        if args.all:
            files = glob.glob(os.path.join(args.file, "*"))
        else:
            files = glob.glob(os.path.join(args.file, "rets*n*_fcoeff*_qinit*_*"))
        # files = glob.glob(os.path.join(args.file, "*"))
    else:
        files = [args.file]

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(files)))

    for i, file in enumerate(files):
        if args.plot_congestion and 'rets' in file:
            data = import_rets_from_csv(file)
            unique_inflows, congested_mean, congested_std = get_ret_data(data)
            # perform plotting operation
            plt.plot(unique_inflows, congested_mean, linewidth=2, c=colors[i])
            p.line(unique_inflows, congested_mean, alpha=0.8, legend = file, line_width=6, color=Spectral[11][i % len(Spectral[11])])
            # if not os.path.isdir(args.file):
            plt.fill_between(unique_inflows, congested_mean - congested_std,
                                congested_mean + congested_std, alpha=0.25, color=colors[i])
        elif not args.plot_congestion:
            data = import_data_from_csv(file)

            # compute the mean and std of the outflows for all unique inflows
            unique_inflows, mean_outflows, std_outflows = get_capacity_data(data)

            # perform plotting operation
            plt.plot(unique_inflows, mean_outflows, linewidth=2, c=colors[i])
            p.line(unique_inflows, mean_outflows, alpha=0.8, line_width=6, legend = file, color=Spectral[11][i % len(Spectral[11])])
            # if not os.path.isdir(args.file):
            plt.fill_between(unique_inflows, mean_outflows - std_outflows,
                                mean_outflows + std_outflows, alpha=0.25, color=colors[i])
    legend_names = files
    # legend_names = [file.split('outflows_')[1].split('.')[0] for file in files]
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Outflow' + r'$ \ \frac{vehs}{hour}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()
    lgd = plt.legend(legend_names, loc='upper left', borderaxespad=0.)
    # plt.tight_layout(pad=7)
    
    if args.bokeh:
        p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'
        show(p)
    else:
        plt.show()
    # if len(files) > 1:
    # 	plt.savefig('trb_data/alinea_data/alinea_comp.png')
    # else:
        # plt.savefig('trb_data/alinea_data/{}.png'.format(legend_names[0]))

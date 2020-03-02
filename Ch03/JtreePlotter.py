import matplotlib.pyplot as plt


decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_node(node_text, center_ptr, parent_ptr, node_type):
    createPlot.ax1.annotate(node_text, xy=parent_ptr, xycoords='axes fraction', )
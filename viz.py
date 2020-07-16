
"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import data_utils
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
  """
  Visualize a 2d skeleton

  Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(data_utils.H36M_NAMES)*2, "channels should have 64 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )

  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
    ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

  # Get rid of the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Get rid of tick labels
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

  RADIUS = 200 # space around the subject
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")

  ax.set_aspect('equal')



def show2Dpose_seq(channels_seq, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
  """
  Visualize a 2d skeleton

  Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """
  len_seq = 20
  print('channels_seq',channels_seq.shape)
  for ind in range(len_seq):
          channels =  channels_seq[ind]
          assert channels.size == len(data_utils.H36M_NAMES)*2, "channels should have 64 entries, it has %d instead" % channels.size
          vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )

          I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
          J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
          LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

          # Make connection matrix
          interval = 150
          for i in np.arange( len(I) ):
              x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
              ax.plot(x+ind*interval, y, lw=2, c=lcolor if LR[i] else rcolor)

          # Get rid of the ticks
          ax.set_xticks([])
          ax.set_yticks([])

          # Get rid of tick labels
          ax.get_xaxis().set_ticklabels([])
          ax.get_yaxis().set_ticklabels([])

          x_RADIUS = 3000 # space around the subject
          y_RADIUS = 200
          xroot, yroot = vals[0,0], vals[0,1]

          margin = 100
          ax.set_xlim([xroot-margin, x_RADIUS+xroot])
          ax.set_ylim([-y_RADIUS+yroot, y_RADIUS+yroot])
          if add_labels:
               ax.set_xlabel("x")
               ax.set_ylabel("z")

    
  ax.set_aspect('equal')

def show2DposeOnImage(img, channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
  """
  Visualize a 2d skeleton

  Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(data_utils.H36M_NAMES)*2, "channels should have 64 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )

  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  ax.imshow(img)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
    ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

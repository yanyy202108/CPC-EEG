import scipy.interpolate
import scipy.io
import numpy as np
import scipy.misc

import matplotlib 
import matplotlib.pyplot as plt
import math

from PIL import Image

def fig2data ( fig ):
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = ( w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


d1 = scipy.io.loadmat('wxj.mat')
#d1 = scipy.io.loadmat('IVa_aa350.mat')

EEG_PSD = np.array(d1['feature_st'])
#EEG_PSD = np.array(d1['train_x'])
# print EEG_PSD.shape

# sensor_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2','P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
# koord = [[1,4],[0,3],[1,3],[0.5,2.5],[0,2],[0,1],[1,0],[3,0],[4,1],[4,2],[3.5,2.5],[3,3],[4,3],[3,4]] 
#sensor_names = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
#koord = [[1,4],[1,3.5],[1,3],[0.25,3],[0.5,2.5],[1.5,2.5],[1,2],[0,2],[0.5,1.5],[1.5,1.5],[1,1],[0.25,1],[1,0.5],[1,0],[2,0],[2,1],[3,4],[3,3.5],[2,3],[3,3],[3.75,3],[3.5,2.5],[2.5,2.5],[2,2],[3,2],[4,2],[3.5,1.5],[2.5,1.5],[3,1],[3.75,1],[3,0.5],[3,0]]
sensor_name = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',	'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','O1','OZ','O2']
koord = [[1,4],[2,4],[3,4],[1,3.5],[3,3.5],[0.25,3],[0.5,3],[1,3],[1.5,3],[2,3],[2.5,3],[3,3],[3.5,3],[3.75,3],[0.25,2.5],[0.5,2.5],[1,2.5],[1.5,2.5],[2,2.5],[2.5,2.5],[3,2.5],[3.5,2.5],[3.75,2.5],[0,2],[0.5,2],[1,2],[1.5,2],[2,2],[2.5,2],[3,2],[3.5,2],[4,2],[0.25,1.5],[0.5,1.5],[1,1.5],[1.5,1.5],[2,1.5],[2.5,1.5],[3,1.5],[3.5,1.5],[3.75,1.5],[0.25,1],[0.5,1],[1,1],[1.5,1],[2,1],[2.5,1],[3,1],[3.5,1],[3.75,1],[0.25,0.5],[0.5,0.5],[1,0.5],[2,1],[3,1],[3.5,1],[3.75,1],[1,0],[2,0],[3,0]]
N_points = 300
xy_center = (2,2)
radius = 2

x, y = [], []

for i in koord:
    x.append(i[0])
    y.append(i[1])

xi = np.linspace(-2,6,N_points)
yi = np.linspace(-2,6,N_points)
#zi = scipy.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

# For theta, alpha and beta bands
fig_b = plt.figure()
fig_m = plt.figure()


ax_b = fig_b.add_subplot(111, aspect = 1)
ax_m = fig_m.add_subplot(111, aspect = 1)
#ax_b = fig_b.add_subplot(111, aspect = 1)
# ax_t.scatter(x, y, marker = 'o', c = 'b', s = 15, zorder = 3)

# remove the ticks
ax_b.set_xticks([])
ax_b.set_yticks([])
ax_m.set_xticks([])
ax_m.set_yticks([])
#ax_b.set_xticks([])
#ax_b.set_yticks([])

ax_b.set_xlim(0, 6)
ax_b.set_ylim(0, 6)
ax_m.set_xlim(0, 6)
ax_m.set_ylim(0, 6)
#ax_b.set_xlim(0, 4)
#ax_b.set_ylim(0, 4)

max_val = np.array([0.0,0.0,0.0])

for i in range (0, len(EEG_PSD)):
    if math.isnan(EEG_PSD[i,0]):
        print ('Got NaN.')
        continue
        
    # For theta, alpha and beta bands
    #zi_t = scipy.interpolate.griddata((x, y), EEG_PSD[i,0:32], (xi[None,:], yi[:,None]), method = 'cubic')
    #zi_a = scipy.interpolate.griddata((x, y), EEG_PSD[i,32:64], (xi[None,:], yi[:,None]), method = 'cubic')
    #zi_b = scipy.interpolate.griddata((x, y), EEG_PSD[i,64:96], (xi[None,:], yi[:,None]), method = 'cubic')
    zi_b = scipy.interpolate.griddata((x, y), EEG_PSD[i,0:60], (xi[None,:], yi[:,None]), method = 'cubic')
    zi_m = scipy.interpolate.griddata((x, y), EEG_PSD[i,60:120], (xi[None,:], yi[:,None]), method = 'cubic')
    #max_val[0] = np.nanmax(zi_t)
    #max_val[1] = np.nanmax(zi_a)
    #max_val[2] = np.nanmax(zi_b)

    #max_ind = np.argmax(max_val)
    #max_val_scaled = max_val / max_val[max_ind]
    # print max_val_scaled

    #CS_t = ax_t.contourf(xi, yi, zi_t, 60, cmap = plt.cm.Reds, zorder = 1)
    #CS_a = ax_a.contourf(xi, yi, zi_a, 60, cmap = plt.cm.Greens, zorder = 1)
    #CS_b = ax_b.contourf(xi, yi, zi_b, 60, cmap = plt.cm.Blues, zorder = 1)
    CS_b = ax_b.contourf(xi, yi, zi_b, 60, cmap = plt.cm.Reds, zorder = 1)
    CS_m = ax_m.contourf(xi, yi, zi_m, 60, cmap = plt.cm.Greens, zorder = 1)
    
    #im_t = fig2img(fig_t)
    #im_a = fig2img(fig_a)
    #im_b = fig2img(fig_b)
    im_b = fig2img(fig_b)
    im_m = fig2img(fig_m)

    im_np_b = np.array(im_b)
    #im_np_b = np.rot90(im_np_b, 1)  
    im_np_b = im_np_b[::-1]
    #im_np_t = im_np_t[136:520, 48:432, :]
    im_np_b = im_np_b[0:576, 0:423, :]

    im_np_m = np.array(im_m)
    #im_np_m = np.rot90(im_np_m, 1)
    im_np_m = im_np_m[::-1]
    #im_np_a = im_np_a[136:520, 48:432, :]
    im_np_m = im_np_m[0:576, 0:423, :]

    #im_np_b = np.array(im_b)
    #im_np_b = np.rot90(im_np_b, 1)
    #im_np_b = im_np_b[::-1]
    ##im_np_b = im_np_b[136:520, 48:432, :]
    #im_np_b = im_np_b[0:576, 0:423, :]

    #added_img = im_np_t + im_np_a + im_np_b
    added_img = im_np_b + im_np_m
    filename = '.\\wxj\\' + str(i+1) + '.png'
    scipy.misc.imsave(filename, added_img)

    if i % 100 == 0:
        print (i)


    # Free memory
    CS_b, CS_m = None, None
    zi_b, zi_m = None, None
    im_b, im_m = None, None
    im_np_b, im_np_m = None, None
    filename = None


    # plt.show() 

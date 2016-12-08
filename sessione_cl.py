import itertools
import thread
import glob
import cv2
import re
import numpy as np
import scipy.interpolate as si
import scipy.signal as signal
import scipy.fftpack as fft
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib import gridspec
from matplotlib.cbook import get_sample_data
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib 
matplotlib.rcParams.update({'font.size': 17})
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os.path

from math import sqrt  # per latexify
SPINE_COLOR = 'gray'   # per latexify

# presa da http://nipunbatra.github.io/2014/08/latexify/
def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """


    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'text.fontsize': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'sans-serif'
    }

    matplotlib.rcParams.update(params)

def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax

def range_brace(x_min, x_max, mid=0.75, 
                beta1=50.0, beta2=100.0, height=1, 
                initial_divisions=11, resolution_factor=1.5):
    # determine x0 adaptively values using second derivitive
    # could be replaced with less snazzy:
    #   x0 = np.arange(0, 0.5, .001)
    x0 = np.array(())
    tmpx = np.linspace(0, 0.5, initial_divisions)
    tmp = beta1**2 * (np.exp(beta1*tmpx)) * (1-np.exp(beta1*tmpx)) / np.power((1+np.exp(beta1*tmpx)),3)
    tmp += beta2**2 * (np.exp(beta2*(tmpx-0.5))) * (1-np.exp(beta2*(tmpx-0.5))) / np.power((1+np.exp(beta2*(tmpx-0.5))),3)
    for i in range(0, len(tmpx)-1):
        t = int(np.ceil(resolution_factor*max(np.abs(tmp[i:i+2]))/float(initial_divisions)))
        x0 = np.append(x0, np.linspace(tmpx[i],tmpx[i+1],t))
    x0 = np.sort(np.unique(x0)) # sort and remove dups
    # half brace using sum of two logistic functions
    y0 = mid*2*((1/(1.+np.exp(-1*beta1*x0)))-0.5)
    y0 += (1-mid)*2*(1/(1.+np.exp(-1*beta2*(x0-0.5))))
    # concat and scale x
    x = np.concatenate((x0, 1-x0[::-1])) * float((x_max-x_min)) + x_min
    y = np.concatenate((y0, y0[::-1])) * float(height)
    return (x,y)

class simulatedAndSetup():
	def __init__(self):
		# confronto spettri
		fromAle = DATA_PATH+'/elab_video/simulatedWhisker_byAle/transffunct_D21_bw1000Hz_sim.txt' #ampvsfreq_D21_rel.txt'  #transffunct_D21_bw1000Hz_sim.txt  #ampvsfreq_D21_damp001.txt
		a = sessione('d21','12May','_NONcolor_','/ratto1/0_acciaio_no_rot/',(260, 780, 0, 205),33,True, False)

		a.calcoloTransferFunction(False)
		spettroVero = a.TFM 
		spettroSim = np.flipud(np.loadtxt(fromAle))
		spettroSim = spettroSim[:-2,3:]
		# figura
		f = plt.figure(figsize=(20,12))
		a1 = f.add_subplot(2,2,1)
		a3 = f.add_subplot(2,2,3)
		cax1 = a1.imshow(np.log10(spettroSim) ,aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')	
		cbar1 = f.colorbar(cax1,ax=a1)
		cax3 =a3.imshow(np.log10(spettroVero),aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')	
		cbar3 = f.colorbar(cax3,ax=a3)
		a3.set_xlabel('Frequency [Hz]') 
		a1.set_yticks([])
		a3.set_yticks([])
		a1.set_ylabel(r'Base        $\longrightarrow$         tip')
		a3.set_ylabel(r'Base        $\longrightarrow$         tip')
		# setup
		'''
		#spezzoni di codice -- forse meglio inserire il setup nel postprocessing
		setupFile =	'Setup_color_transparent_background.png'
		aw1.imshow(LB)
		self.luceBluFiltroPLung = directory+'IMG_0236.JPG'
		LB=mpimg.imread(self.luceBianca)
		'''	
		f.savefig(DATA_PATH+'/elab_video/simulationAndSetup.pdf')
		
	

class PickleAsciiTimeTrendsConversion(): # prendo i dati di un baffo dal pickle e salvo i trend punto a punto in un file ascii
	def __init__(self):
		Pickle2Ascii = True # se True from pickle to ascii
							# se False from ascii to pickle
		self.sample = 'filo_acciaio' #'d21' #
		self.date = '13Apr' #'12May' #
		self.label = '_NONcolor_'
		self.path = '/media/jaky/DATI BAFFO/elab_video/'
		if Pickle2Ascii:
			self.fext = 'pickle'
			self.pickle2ascii_conversion()	

	def pickle2ascii_conversion(self):
		elabSessione = False
		ROI_inutile = (310, 629, 50, 210)
		THs_inutile = 29
		s = sessione(self.sample,self.date,self.label,self.path,ROI_inutile,THs_inutile,True,elabSessione,False,True) # carico la sessione senza elabolarla
		s.loadTracking()
		k = 1
		for v in s.V:
			f = open(self.path+self.sample+'_'+self.date+'_'+self.label+'_video_'+str(k)+'.txt','wb') 
			k = k+1
			for dbaffo in v.wst: # punti baffo
				for yt in dbaffo:
					f.write(str(yt)+'\t')
				f.write('\n') 
			f.close()
			
	
class creoImageProcessing_Stacked(): # 
	def __init__(self): 

		elabSessione = False
		s = sessione('d21','12May','_NONcolor_',DATA_PATH+'/ratto1/d2_1/',(310, 629, 50, 210),29,True,elabSessione,False,True) # carico la sessione senza elabolarla
		s.resolvePath(s.path)
		avi = s.aviList[0]
		v = video(avi,(0,650-340,0,235-100),29,False,False,False)
		cap = cv2.VideoCapture(avi) 	
		_,Read = cap.read()
		Read = Read[110:235,358:650]   	
		Frame_raw = cv2.cvtColor(Read, cv2.COLOR_BGR2GRAY) 						
		Frame_blur = cv2.medianBlur(Frame_raw,3)  											
		Frame_ths = cv2.threshold(Frame_blur,v.videoThs,255,cv2.THRESH_BINARY)[1]  	     # ths 
		Frame_Bspline = cv2.threshold(Frame_blur,v.videoThs,255,cv2.THRESH_BINARY)[1]  	 # blur 
		x,y = v.get_whisker(Frame_Bspline) 
		x,y = v.norm_whisker(x,y,35,3)
		x = x[:-2] # tolgo la base dalla stima (NON ho definito una ROI qui...) 
		y = y[:-2] # tolgo la base dalla stima (NON ho definito una ROI qui...) 
		Frame_Bspline/=5 #3.0 
		for i,j in zip(x,y): 
			if not np.isnan(j):
				i = int(i)
				j = int(j)
				for k1 in range(-2,2):
					for k2 in range(-2,2):
						Frame_Bspline[j+k1][i+k2] = 255
		if 1: # in caso le salvo
			cv2.imwrite(DATA_PATH+'/elab_video/ImageProc_Raw.jpg',			Frame_raw)     
			cv2.imwrite(DATA_PATH+'/elab_video/ImageProc_Blurred.jpg',		Frame_blur)     
			cv2.imwrite(DATA_PATH+'/elab_video/ImageProc_Thresholded.jpg',	Frame_ths)     
			cv2.imwrite(DATA_PATH+'/elab_video/ImageProc_Bspline.jpg',		Frame_Bspline)     
	
		# plot	
		Layers = [] 
		Layers.append(Frame_raw)     
		#Layers.append(Frame_blur)    
		Layers.append(Frame_ths)     
		Layers.append(Frame_Bspline) 

		x_offset, y_offset = 90, 90  # Number of pixels to offset each image.
		r = Frame_raw
		new_shape = ((Layers.__len__() - 1)*y_offset + r.shape[0],
					 (Layers.__len__() - 1)*x_offset + r.shape[1])  # the last number, i.e. 4, refers to the 4 different channels, being RGB + alpha

		stacked = 100*np.ones(new_shape, dtype=np.float)
		for layer,L in zip(range(Layers.__len__()),Layers):
			stacked[layer*y_offset:layer*y_offset + r.shape[0],
					layer*x_offset:layer*x_offset + r.shape[1], 
					...] = L*1./Layers.__len__()
		
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)		
		for spine in ('top','bottom','left','right'):
			ax.spines[spine].set_visible(False)
		'''
		# NON mi piazzi...
		ax.annotate("",
            xy=(160, 300), 
            xytext=(40, 150), 
			xycoords='data',
			textcoords='data',
            arrowprops=dict(arrowstyle="fancy", #linestyle="dashed",
                            color="0",
                            shrinkB=5,
                            connectionstyle="arc3,rad=0.3",
                            ),
        )
		ax.text(40,250,"Processing",fontsize=FS,color='k') 
		'''
		FS = 22
		ax.text(100-10,20,"Raw Image",fontsize=FS,color='white') 
		ax.text(100+y_offset-15,20+x_offset,"Thresholding",fontsize=FS,color='white') 
		ax.text(100+2*y_offset-30,20+2*x_offset,"Bspline modeling",fontsize=FS,color='white') 
		ax.set_xticks([])
		ax.set_yticks([])
		ax.imshow(stacked,cmap='gray')
		fig.savefig(DATA_PATH+'/elab_video/stacked.png') # <--- serve per la base della figura 2
		fig.savefig(DATA_PATH+'/elab_video/stacked.pdf')

class stampo_lunghezza_whiskers(): # calcolo le lunghezze dei baffi e le stampo a video
	def __init__(self): 
		a = confrontoBaffiDiversi('baffi_12May','diversiBaffi',False)
		for w,l in zip(a.ROOT,a.integrale_lunghezza):
			print w, ' : ', l

class creoSpettriBaffi(): # carico i dati per riplottare gli spettri
	def __init__(self): 
		a = confrontoBaffiDiversi('baffi_12May','diversiBaffi',False) # per le lunghezze dei baffi 
		# calcolo le transfer functions dato che non sono salvate... 
		'''
		# avro` gia` cambiato quella classe per computare la transfer function?? 
		'''
		ba = sessione('c12','12May','_NONcolor_',DATA_PATH+'/ratto1/0_acciaio_no_rot/',(260, 780, 0, 205),33,True, False)
		bb = sessione('c22','12May','_NONcolor_',DATA_PATH+'/ratto1/0_acciaio_no_rot/',(260, 780, 0, 205),33,True, False)
		bc = sessione('a31','12May','_NONcolor_',DATA_PATH+'/ratto1/0_acciaio_no_rot/',(260, 780, 0, 205),33,True, False)
		for baffo in [ba,bb,bc]:
			baffo.calcoloTransferFunction(False)

		if 1: # 3 spettri (due simili uno diverso), due distanze relative e gli scatter calcolati pixel a pixel
			ig1 = 3 # gruppo 1
			ig2 = 4 # gruppo 1
			ig3 = 10 # gruppo 3
			maxFrame = 4500
			Np = maxFrame/2. 		# numero di campioni frequenze positive
			bw = 2000.0 			# frame/sec - bandwidth
			df = (bw/2)/Np			# df 
			freq = [int(df*c) for c in xrange(0,801)]	
			# prendo i dati
			if 0: #semplice spettro
				g1 = a.AvSp_ncol[ig1]  # gruppo 1
				g2 = a.AvSp_ncol[ig2]  # gruppo 2 
				g3 = a.AvSp_ncol[ig3] # gruppo 3
				g1 = g1[:,:800]/np.max(g1[:,:800])
				g2 = g2[:,:800]/np.max(g2[:,:800]) 
				g3 = g3[:,:800]/np.max(g3[:,:800])
				g1_to_plot = np.log10(g1)
				g2_to_plot = np.log10(g2)
				g3_to_plot = np.log10(g3)
				g1_to_plot -= np.min(g1_to_plot)
				g2_to_plot -= np.min(g2_to_plot)
				g3_to_plot -= np.min(g3_to_plot)
				g1_to_plot /= np.max(g1_to_plot)
				g2_to_plot /= np.max(g2_to_plot)
				g3_to_plot /= np.max(g3_to_plot)
			else: # transfer function
				g1 = ba.TFM[:,:350]
				g2 = bb.TFM[:,:350]
				g3 = bc.TFM[:,:350]
				g1_to_plot = np.log10(g1)
				g2_to_plot = np.log10(g2)
				g3_to_plot = np.log10(g3)

			def doFigura(a,wLog,f): 
				a1 = f.add_subplot(2,3,1)
				a2 = f.add_subplot(2,3,2)
				a3 = f.add_subplot(2,3,3)
				a4 = f.add_subplot(2,3,4)
				a5 = f.add_subplot(2,3,5)
				#gs = gridspec.GridSpec(4,3)
				#a61 = plt.subplot(gs[8])
				#a62 = plt.subplot(gs[11])
				a6 = f.add_subplot(2,3,6)
				# spettri
				cax1 = a1.imshow(g1_to_plot,aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')	
				cax2 = a2.imshow(g2_to_plot,aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')
				cax3 = a3.imshow(g3_to_plot,aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')	
				d12 = np.power(g2_to_plot-g1_to_plot,2)
				d12 /= np.max(d12)
				d13 = np.power(g3_to_plot-g1_to_plot,2)
				d13 /= np.max(d13)
				cax4 = a4.imshow(d12,aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')	
				cax5 = a5.imshow(d13,aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')	
				cbar1 = f.colorbar(cax1,ax=a1)
				cbar1.set_ticks(np.arange(0,1.1,.1))
				cbar1.ax.tick_params(labelsize=FS)
				cbar2 = f.colorbar(cax2,ax=a2)
				cbar2.set_ticks(np.arange(0,1.1,.1))
				cbar2.ax.tick_params(labelsize=FS)
				cbar3 = f.colorbar(cax3,ax=a3)
				cbar3.set_ticks(np.arange(0,1.1,.1))
				cbar3.ax.tick_params(labelsize=FS)
				cbar4 = f.colorbar(cax4,ax=a4)
				cbar4.set_ticks(np.arange(0,1.1,.1))
				cbar4.ax.tick_params(labelsize=FS)
				cbar5 = f.colorbar(cax5,ax=a5)
				cbar5.set_ticks(np.arange(0,1.1,.1))
				cbar5.ax.tick_params(labelsize=FS)
				#
				def shorten(l):
					l = np.round(l*100)
					return str(l/100)
				a1.set_title('W1 = '+str(shorten(a.integrale_lunghezza[ig1]))+'[mm]',fontsize=FS)
				a2.set_title('W2 = '+str(shorten(a.integrale_lunghezza[ig2]))+'[mm]',fontsize=FS)
				a3.set_title('W3 = '+str(shorten(a.integrale_lunghezza[ig3]))+'[mm]',fontsize=FS)
				a4.set_title('W1-W2',fontsize=FS)
				a5.set_title('W1-W3',fontsize=FS)
				a5.set_xlabel('Frequency [Hz]',fontsize=FS)
				idx = [freq.index(0),freq.index(100),freq.index(200),freq.index(300)]
				for a in (a1,a2,a3,a4,a5):
					a.set_yticks([])
					a.set_xticks(idx)
					a.set_xticklabels([freq[i] for i in idx])
				a1.set_ylabel(r'Base        $\longrightarrow$         tip',fontsize=FS)

				# scatter
				g1r = np.reshape(g1,g1.__len__()*g1[0].__len__())
				g2r = np.reshape(g2,g1.__len__()*g1[0].__len__())
				g3r = np.reshape(g3,g1.__len__()*g1[0].__len__())
				idx = np.random.permutation(len(g1r))[0:10000]
				w13 = a6.scatter(g3r[idx],g1r[idx],s=6**2,facecolor='#dbc65e',color='#dbc65e', alpha=0.4, rasterized=True)
				w12 = a6.scatter(g2r[idx],g1r[idx],s=6**2,facecolor='#ef725f',color='#ef725f',marker='x', alpha=0.4, rasterized=True)
				a6.legend((w12,w13), ('similar','diverse'), scatterpoints=1,markerscale=2, loc='upper left',fontsize=FS)
				# regressione 
				def getLine(g,h): 
					slope, intercept, r_value, p_value, std_err = stats.linregress(g,h)
					r2 = r_value**2
					xv = [a for a in np.arange(0,1.1,.1)]
					yv = [a*slope+intercept for a in xv]
					if np.sign(intercept)>0:
						segno='+'
					else:
						segno='-'
					text = 'y = '+str(np.floor(slope*100)/100)+'x'+segno+str(np.abs(np.floor(intercept*100)/100))+'\n    (R2 = '+str(np.floor(r2*100)/100)+')'
					return xv,yv,r_value**2, text
				x12,y12,r12,pp12 = getLine(g2r,g1r)
				x13,y13,r13,pp13 = getLine(g3r,g1r)
				a6.plot(x12,y12,color='#6d2a2a')
				a6.plot(x13,y13,color='#6d622a')
				a6.text(0.28,0.6,pp12,fontsize=14)
				a6.text(0.5,0.31,pp13,fontsize=14)
				for spine in ('top','bottom','left','right'):
					a6.spines[spine].set_visible(False)
				#a6.set_xlim([0,1])
				#a6.set_ylim([0,1])
				a6.tick_params(labelsize=FS) 
				a1.tick_params(labelsize=FS) 
				a2.tick_params(labelsize=FS) 
				a3.tick_params(labelsize=FS) 
				a4.tick_params(labelsize=FS) 
				a5.tick_params(labelsize=FS) 
				return a1,a2,a3,a4,a5,a6

			# faccio figura
			FS = 20 # la dimensione del font dipende dalla dimensione della figura
			f1 = plt.figure(figsize=(20,12))
			a11,a12,a13,a14,a15,a16 = doFigura(a,True, f1)
			#f1.savefig(DATA_PATH+'/elab_video/DiffSpectra.pdf')
			f1.savefig(DATA_PATH+'/elab_video/DiffTransferFunction.pdf')

		else: # NON sara` piu` cosi` questa figura  
			print a.ROOT, ' len=', a.ROOT.__len__()
			print a.ROOT[3]
			print a.ROOT[5]
			print a.ROOT[11]
			ig1 = 3
			ig2 = 7 
			ig3 = 10 
			g1 = a.AvSp_ncol[ig1]  # gruppo 1
			g2 = a.AvSp_ncol[ig2]  # gruppo 2 
			g3 = a.AvSp_ncol[ig3] # gruppo 3

			# figura
			f1 = plt.figure()
			aa = f1.add_subplot(1,1,1)
			f2 = plt.figure(figsize=(7,6))
			gs = gridspec.GridSpec(1,3,wspace=0.2)
			ax1 = plt.subplot(gs[0,0])
			ax2 = plt.subplot(gs[0,1])
			ax3 = plt.subplot(gs[0,2])
			#f2,(ax1,ax2,ax3) = plt.subplots(1,3,sharex=True)
			# plotto i dati
			maxFrame = 4500
			Np = maxFrame/2. 		# numero di campioni frequenze positive
			bw = 2000.0 			# frame/sec - bandwidth
			df = (bw/2)/Np			# df 
			freq = [int(df*c) for c in xrange(0,801)]	
			g1_to_plot = np.log10(g1[:,:800])
			g2_to_plot = np.log10(g2[:,:800]) 
			g3_to_plot = np.log10(g3[:,:800])
			g1_to_plot -= np.min(g1_to_plot)
			g2_to_plot -= np.min(g2_to_plot)
			g3_to_plot -= np.min(g3_to_plot)
			g1_to_plot /= np.max(g1_to_plot)
			g2_to_plot /= np.max(g2_to_plot)
			g3_to_plot /= np.max(g3_to_plot)

			cax  = aa.imshow(g1_to_plot,aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')	
			cax1 = ax1.imshow(g1_to_plot,aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')	
			cax2 = ax2.imshow(g2_to_plot,aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')
			cax3 = ax3.imshow(g3_to_plot,aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')	
			#
			cbar = f1.colorbar(cax,ax=aa)
			cbar.set_ticks([])
			cbar1 = f2.colorbar(cax1,ax=ax1)
			cbar1.set_ticks(np.arange(0,1.1,.1))
			cbar1.ax.tick_params(labelsize=10)
			cbar2 = f2.colorbar(cax2,ax=ax2)
			cbar2.set_ticks(np.arange(0,1.1,.1))
			cbar2.ax.tick_params(labelsize=10)
			cbar3 = f2.colorbar(cax3,ax=ax3)
			cbar3.set_ticks(np.arange(0,1.1,.1))
			cbar3.ax.tick_params(labelsize=10)
			#
			def shorten(l):
				l = np.round(l*100)
				return str(l/100)
			ax1.set_title(str(shorten(a.integrale_lunghezza[ig1]))+'[mm]',fontsize=14)
			ax2.set_title(str(shorten(a.integrale_lunghezza[ig2]))+'[mm]',fontsize=14)
			ax3.set_title(str(shorten(a.integrale_lunghezza[ig3]))+'[mm]',fontsize=14)
			#
			aa.set_xlabel('Frequency [Hz]',fontsize=14)
			ax2.set_xlabel('Frequency [Hz]',fontsize=14)
			idx = [freq.index(0),freq.index(100),freq.index(200),freq.index(300)]
			for a in (aa,ax1,ax2,ax3):
				a.set_xticks(idx)
				a.set_xticklabels([freq[i] for i in idx])
			aa.set_ylabel(r'Base - - - - $\longrightarrow$  - - - - tip',fontsize=14)
			ax1.set_ylabel(r'Base        $\longrightarrow$         tip',fontsize=14)
			ax1.tick_params(labelsize=10) 
			ax2.tick_params(labelsize=10) 
			ax3.tick_params(labelsize=10) 
			aa.set_yticks([])
			ax1.set_yticks([])
			ax2.set_yticks([])
			ax3.set_yticks([])
			#f.text(0.02, 0.5, 'Whisker dynamic response \n pippo', ha='center', va='center', rotation='vertical')
			latexify()
			f1.tight_layout()
			#f2.tight_layout()
			#f2.subplots_adjust(wspace=0.05,hspace=0.6)
			f1.savefig(DATA_PATH+'/elab_video/OneSpectrum.pdf')
			f2.savefig(DATA_PATH+'/elab_video/baseFigura5.pdf')

class mergeComparisonsResults():
	def __init__(self):

		typeComparison = 'transferFunction'  # 'spettri' # 

		# carico dati
		a = confrontoBaffiDiversi('baffi_12May','diversiBaffi',False)
		a.loadWhiskersInfo()
		a.compareWhiskers(typeComparison) 
		b = confrontoBaffiDiversi('baffi_12May','diversiTempi',False)    
		b.compareWhiskers(typeComparison) 

		# axes arrangements
		f = plt.figure(figsize=(10,6))
		UnDyed = f.add_subplot(2,3,1)
		ColorC = f.add_subplot(2,3,2)
		Dyed = f.add_subplot(2,3,3)
		TimeC = f.add_subplot(2,3,5)
		WhiskerGroup = f.add_subplot(2,3,4)
		gs  = gridspec.GridSpec(4,3,hspace=1)
		ColorD = plt.subplot(gs[3,2])
		TimeSD = plt.subplot(gs[2,2],sharey=ColorD)

		# plot stuff
		self.sizeWhiskerGroups(a,WhiskerGroup)
		self.colorComparison(a,ColorC)
		cax = self.timeComparison(b,TimeC)
		cbar3 = f.colorbar(cax,ax=ColorC)
		cbar3.ax.tick_params(labelsize=10)
		cbar4 = f.colorbar(cax,ax=TimeC)
		cbar4.ax.tick_params(labelsize=10)
		self.diagColComp(a,TimeSD)
		self.supradiagTimeComp(b,ColorD)
		self.UndyedWhiskersComparison(a,UnDyed)
		cbar5 = f.colorbar(cax,ax=UnDyed)
		cbar5.ax.tick_params(labelsize=10)
		self.DyedWhiskersComparison(a,Dyed)
		cbar6 = f.colorbar(cax,ax=Dyed)
		cbar6.ax.tick_params(labelsize=10)

		f.tight_layout()
		f.subplots_adjust(wspace=0.45,hspace=0.6)
		f.savefig(DATA_PATH+'/elab_video/baseFigura4_'+typeComparison+'.pdf')


	def diagColComp(self,cbd,a51):
		# calcolo le lunghezze per metterle in ordine
		def shorten(l):
			l = np.round(l*100)
			return str(l/100)
		lengths = [shorten(l) for l in cbd.integrale_lunghezza]
		l_sort = [i[0] for i in reversed(sorted(enumerate(lengths), key=lambda x:x[1]))]
		lengths = [lengths[i] for i in l_sort] 
		#
		d_c2 = []
		for i in xrange(0,cbd.CORR2.__len__()):
			d_c2.append(cbd.CORR2[i][i])
		x = np.linspace(0,12,13)
		a51.plot(x,d_c2,'k.',markersize=5)
		a51.plot(x,d_c2,'k')
		a51.set_xticks(np.arange(0,len(cbd.ROOT),1))
		a51.set_yticks(np.arange(0.6,1.1,0.2))
		a51.axis([-0.2, len(cbd.ROOT)-0.8, 0.1, 1])
		#a51.set_xticklabels([])
		a51.set_ylabel('Similarity', color='k',fontsize=14, y=-.5,x = 0.3) # e` condiviso
		a51.set_title('Dye effect', color='k',fontsize=12)
		for spine in ('top','bottom','left','right'):
			a51.spines[spine].set_visible(False)
		a51.tick_params(labelsize=10) 
		a51.set_xticklabels(lengths,rotation=90) #cbd.ROOT[0:14],rotation=90)
		a51.set_xlim([-.5, 12.5])
		a51.set_ylim([0.5, 1.1])
		#a51.axes.get_xaxis().set_visible(False)

	def supradiagTimeComp(self,cbd,a51):
		ROOT  	= [re.sub('[$]','',cbd.ROOT[i]) for i in xrange(0,cbd.ROOT.__len__()) if cbd.group3[i] == 0] # uso le regular expression per togliere i $ che mi servono per l'interpreter latex per fare il corsivo 
		d_c2 = []
		for i in xrange(0,12):
			d_c2.append(cbd.CORR2[i][i+1])
		a51.plot(d_c2,'k.',markersize=5)
		a51.plot(d_c2,'k')
		a51.axis([-0.2, len(cbd.CORR2)-0.8, 0.1, 1])
		a51.set_yticks(np.arange(0.2,1.2,0.2))
		a51.set_xticks(xrange(0,len(cbd.CORR2)-1))
		#a51.set_ylabel('Similarity', color='k',fontsize=14) # e` condiviso
		a51.set_title('Time effect', color='k',fontsize=12)
		for tl in a51.get_yticklabels():
			tl.set_color('k')
		for spine in ('top','bottom','left','right'):
			a51.spines[spine].set_visible(False)
		a51.tick_params(labelsize=10) 
		#a51.set_xticklabels(ROOT[1:])
		a51.set_xticklabels(ROOT[1:13],rotation=90)
		a51.set_xlim([-.5, 11.5])
		#a51.axes.get_xaxis().set_visible(False)

	def colorComparison(self,cbd,a2):
		# calcolo le lunghezze per metterle in ordine
		def shorten(l):
			l = np.round(l*100)
			return str(l/100)
		lengths = [shorten(l) for l in cbd.integrale_lunghezza]
		l_sort = [i[0] for i in reversed(sorted(enumerate(lengths), key=lambda x:x[1]))]
		lengths = [lengths[i] for i in l_sort] 
		# riordino la matrice di interesse
		CORR2 = cbd.CORR2
		for i in xrange(0,CORR2.__len__()): 
			j = l_sort[i]
			cbd.CORR2[j] = CORR2[i]
		# faccio il plot
		cax2 = a2.imshow(cbd.CORR2,aspect='equal', interpolation="nearest",clim=(0,1))
		a2.set_xticks(np.arange(len(cbd.ROOT)))
		a2.set_xticklabels(lengths,rotation=90)#cbd.ROOT,rotation=90)
		a2.set_yticks(np.arange(len(cbd.ROOT)))
		a2.set_yticklabels(lengths)#cbd.ROOT)
		a2.set_ylabel('Length [mm]',fontsize=14)
		#a2.set_xlabel('Length [mm]',fontsize=14)
		a2.text(-.25, -.25, 'Undyed \n  Dyed',horizontalalignment='center',verticalalignment='center',rotation=45,transform=a2.transAxes,fontsize=12)
		a2.annotate('', xy=(-0.42, -0.42), xycoords='axes fraction', xytext=(0, 0), arrowprops=dict(arrowstyle="-", color='k'))
		#a2.set_ylabel('Undyed')
		#a2.set_xlabel('Dyed')
		a2.tick_params(labelsize=10) 
		return cax2

	def DyedWhiskersComparison(self,cbd,a2):
		# calcolo le lunghezze per metterle in ordine
		def shorten(l):
			l = np.round(l*100)
			return str(l/100)
		lengths = [shorten(l) for l in cbd.integrale_lunghezza]
		l_sort = [i[0] for i in reversed(sorted(enumerate(lengths), key=lambda x:x[1]))]
		lengths = [lengths[i] for i in l_sort] 
		# riordino la matrice di interesse
		CORR2 = cbd.CORR2_dyed
		for i in xrange(0,CORR2.__len__()): 
			j = l_sort[i]
			cbd.CORR2_dyed[j] = CORR2[i]
		# faccio il plot
		cax2 = a2.imshow(cbd.CORR2_dyed,aspect='equal', interpolation="nearest",clim=(0,1))
		a2.set_xticks(np.arange(len(cbd.ROOT)))
		a2.set_xticklabels(lengths,rotation=90)#cbd.ROOT,rotation=90)
		a2.set_yticks(np.arange(len(cbd.ROOT)))
		a2.set_yticklabels(lengths)#cbd.ROOT)
		a2.set_ylabel('Length [mm]',fontsize=14)
		#a2.set_xlabel('Length [mm]',fontsize=14)
		a2.text(-.25, -.25, 'Dyed \n  Dyed',horizontalalignment='center',verticalalignment='center',rotation=45,transform=a2.transAxes,fontsize=12)
		a2.annotate('', xy=(-0.42, -0.42), xycoords='axes fraction', xytext=(0, 0), arrowprops=dict(arrowstyle="-", color='k'))
		#a2.set_ylabel('Dyed')
		#a2.set_xlabel('Dyed')
		a2.tick_params(labelsize=10) 
		return cax2


	def UndyedWhiskersComparison(self,cbd,a2):
		# calcolo le lunghezze per metterle in ordine
		def shorten(l):
			l = np.round(l*100)
			return str(l/100)
		lengths = [shorten(l) for l in cbd.integrale_lunghezza]
		l_sort = [i[0] for i in reversed(sorted(enumerate(lengths), key=lambda x:x[1]))]
		lengths = [lengths[i] for i in l_sort] 
		# riordino la matrice di interesse
		CORR2 = cbd.CORR2_undyed
		for i in xrange(0,CORR2.__len__()): 
			j = l_sort[i]
			cbd.CORR2_undyed[j] = CORR2[i]
		# faccio il plot
		cax2 = a2.imshow(cbd.CORR2_undyed,aspect='equal', interpolation="nearest",clim=(0,1))
		a2.set_xticks(np.arange(len(cbd.ROOT)))
		a2.set_xticklabels(lengths,rotation=90)#cbd.ROOT,rotation=90)
		a2.set_yticks(np.arange(len(cbd.ROOT)))
		a2.set_yticklabels(lengths)#cbd.ROOT)
		a2.set_ylabel('Length [mm]',fontsize=14)
		#a2.set_xlabel('Length [mm]',fontsize=14)
		a2.text(-.25, -.25, 'Undyed \n  Undyed',horizontalalignment='center',verticalalignment='center',rotation=45,transform=a2.transAxes,fontsize=12)
		a2.annotate('', xy=(-0.42, -0.42), xycoords='axes fraction', xytext=(0, 0), arrowprops=dict(arrowstyle="-", color='k'))
		#a2.set_ylabel('Undyed')
		#a2.set_xlabel('Undyed')
		a2.tick_params(labelsize=10) 
		return cax2


	def timeComparison(self,cbd,a1):
		ROOT  	= [re.sub('[$]','',cbd.ROOT[i]) for i in xrange(0,cbd.ROOT.__len__()) if cbd.group3[i] == 0] # uso le regular expression per togliere i $ che mi servono per l'interpreter latex per fare il corsivo 
		CORR2 	= cbd.CORR2[0:ROOT.__len__(),0:ROOT.__len__()]
		cax1 = a1.imshow(CORR2,aspect='equal', interpolation="nearest",clim=(0,1))
		a1.set_xticks(np.arange(len(ROOT)))
		a1.set_xticklabels(ROOT)
		a1.set_xticklabels(ROOT,rotation=90)
		a1.set_yticks(np.arange(len(ROOT)))
		a1.set_yticklabels(ROOT)
		#a1.set_xlabel('Whisker')
		a1.set_ylabel('Over time',fontsize=14) # C3 o 37.99mm
		a1.tick_params(labelsize=10) 
		return cax1


	def sizeWhiskerGroups(self,cbd,a1):
		dist  = []
		corr2 = []
		corr2c = []
		corr2nc = []
		for i in xrange(0,cbd.CORR2.__len__()):
			for j in xrange(0,cbd.CORR2.__len__()):
				dist.append(np.abs(cbd.integrale_lunghezza[i]-cbd.integrale_lunghezza[j]))
				corr2.append(cbd.CORR2[i,j])
				corr2c.append(cbd.CORR2_dyed[i,j])
				corr2nc.append(cbd.CORR2_undyed[i,j])
		a1.scatter(dist, corr2, s=6**2, color='0.3', alpha=0.5)
		#a1.scatter(dist, corr2, s=3**2, color='r', alpha=0.5)   # <--- vengono identici
		#a1.scatter(dist, corr2, s=2**2, color='g', alpha=0.5)   # <--- vengono identici
		a1.set_xlabel('Length Difference',fontsize=14)
		a1.set_ylabel('Similarity',fontsize=14)
		a1.set_ylim([-0.05, 1.05])
		a1.set_xlim([-2, 42])
		a1.tick_params(labelsize=10) 
		for spine in ('top','bottom','left','right'):
			a1.spines[spine].set_visible(False)

		'''
		gLengths = [[],[],[],[]]
		for l,n in zip(cbd.integrale_lunghezza,cbd.ROOT):
			if n in cbd.group1:
				gLengths[0].append(l)
			if n in cbd.group2:
				gLengths[1].append(l)
			if n in cbd.group3:
				gLengths[2].append(l)
			if n in cbd.group4:
				gLengths[3].append(l)
		gLenMean = [np.mean(g) for g in gLengths if g] 
		gLenSem = [stats.sem(g) for g in gLengths if g] 
		bars = a1.bar([3.3,6.3,8.9],gLenMean,0.5,color='gray',yerr=gLenSem,error_kw=dict(elinewidth=2,ecolor='gray'))
		a1.text(6,50,'Length [mm]',fontsize=14)
		def autolabel(ax,rects):
			# attach some text labels
			for rect in rects:
				height = rect.get_height()
				ax.text(rect.get_x() + rect.get_width()/2., 3+height,'%d' % int(height),ha='center', va='bottom',fontsize=14)
		autolabel(a1,bars)
		#a1.set_xticklabels([])
		a1.spines['right'].set_visible(False)
		a1.spines['left'].set_visible(False)
		a1.spines['bottom'].set_visible(False)
		a1.spines['top'].set_visible(False)
		a1.axes.get_xaxis().set_visible(False)
		a1.axes.get_yaxis().set_visible(False)
		a1.set_xlim([0,len(cbd.ROOT)])
		a1.set_yticks(xrange(0,66,5))
		#a1.ylabel('Length [mm]')
		# grafe
		x1,y1 = range_brace(1.9,5.2)
		x2,y2 = range_brace(5.4,7.7)
		x3,y3 = range_brace(7.9,10.3)
		a1.plot(x1,6*(y1-1.2),color='gray')
		a1.plot(x2,6*(y2-1.2),color='gray')
		a1.plot(x3,6*(y3-1.2),color='gray')
		'''

class confrontoAddestramento: # confronto le performance di 4 ratti, pre/post-anestesia, due di controllo due con colorazione nel post-anestesia
	def __init__(self):
		#latexify()
		self.initData()
		self.trendData(True)

	def initData(self):
		# colore: ratto 1 e 3
		self.ratto_1_pre  = [80,87.80,80.60,83.60,85.60,82,84.60,77.90,77.60,80,80,84,86,84.70,86.60,83,85.50,85.90,87.90]
		self.ratto_1_post = [82.70,83.80,84.50,81.80,80,84.60,80.50,81,88,84,84,86.40,86,82.70,80.50,90,80,85.50,90,78,80,79.80,83.90,87] 
		self.ratto_3_pre  = [79,83.70,77.30,77.60,74.70,80.60,81.40,78,78.30,82,80.50,81,81,82,80.50,81.70,85.90,80.40,82.80]	
		self.ratto_3_post = [80,83.90,84.70,80.50,80.70,83,79.50,87,84,80.70,81.60,76,81.30,81.30,83.80,85.50,78.80,83.70,85,85,82.60,81.80,80.80,82.30]
		# controllo: ratto 2 e 4
		self.ratto_2_pre  = [83,83,77.80,81,85,81.70,82.40,85.30,80.80,80.70,82,84,83.80,79,81.40,81.40,85.70,83,82]
		self.ratto_2_post = [82,79,78.50,83,81.50,83,87.60,84,81.50,81,83.70,81,81,88,88.30,86,82.80,81.30,81,77.90,76.40,85.50,80,83.80]
		self.ratto_4_pre  = [80.50,78.40,81.90,78.40,84.90,79.40,77.50,75.30,83,75.60,79.30,84,79,79.50,76.80,76.50,80.70,83,81.70]
		self.ratto_4_post = [78.70,79,81.30,83.80,81,83.00,84,77,81,82,79.40,82.80,80.70,80,79.80,83.30,76,82,80.50,82,81,81.40,81.70,82.60]

	def trendData(self,salva=False):
		colors = ['b','c','m','g'] #cm.rainbow(np.linspace(0, 1, 4)) # 4 gruppi 

		def annotatingPatches(ax, info):
			stylename= 'wedge'
			x,y,dx,dy,xc,yc,color,commento,fontsize = info
			#xc = x+dx/2.
			#yc = y+dy+5
			print x,y,dx,dy,commento,fontsize
			p = ax.add_patch(
					patches.Rectangle(
						(x, y),   # (x,y)
						dx,          # width
						dy,          # height
					)
				)
			p.set_alpha(0.2)
			p.set_color(color)
			ax.annotate(commento, (x+dx/2., y+dy),
						(xc, yc),
						#xycoords="figure fraction", textcoords="figure fraction",
						ha="right", va="center",
						size=fontsize,
						arrowprops=dict(arrowstyle=stylename,
										patchB=p,
										shrinkA=5,
										shrinkB=5,
										fc="w", ec="k",
										connectionstyle="arc3,rad=-0.05",
										),
						)

		def creo_a1(a1):
			# disegno i trend
			area = np.pi*(1.5**2)
			Rats_name = ['colored rat','control rat','colored rat','control rat']
			Rats_pre  = [self.ratto_1_pre, self.ratto_2_pre, self.ratto_3_pre, self.ratto_4_pre]
			Rats_post = [self.ratto_1_post, self.ratto_2_post, self.ratto_3_post, self.ratto_4_post]
			xx = 0
			for rat_pre,rat_post,n,c in zip(Rats_pre,Rats_post,Rats_name,colors): 
				a1.scatter(np.linspace(0, 9.5, rat_pre.__len__()), rat_pre, area,color=c,alpha=0.8)
				a1.scatter(np.linspace(10.5, 20, rat_post.__len__()),rat_post,area,color=c,alpha=0.5)
				a1.plot(np.linspace(0, 9.5, rat_pre.__len__()), rat_pre, color=c, linewidth=1.2,alpha=0.5)
				a1.plot(np.linspace(10.5, 20, rat_post.__len__()),rat_post, color=c, linewidth=1.2,alpha=0.5)
				a1.plot(np.linspace(9.5, 10.5, 2), [rat_pre[-1],rat_post[0]], color=c, linestyle='--',alpha=0.5)
				#plt.annotate(n,(2.05,rat_post[-1]),arrowprops=dict(facecolor=c, shrink=0.05)) 
				#a1.text(xx, 90, n, bbox={'facecolor':c, 'alpha':0.5})
				xx += 0.25
			a1.spines['right'].set_visible(False)
			a1.spines['left'].set_visible(False)
			a1.spines['bottom'].set_visible(False)
			a1.spines['top'].set_visible(False)
			a1.get_xaxis().tick_top()
			a1.get_yaxis().tick_left()
			a1.set_ylabel('Performance[%]', fontsize=14)
			a1.set_yticks([75,80,85,90]) # niente
			if 0:
				a1.axvline(x=1, color='gray', linestyle='-.')	
				a1.set_xticks([0.5,1.5])
				a1.set_xticklabels(['before','after']) # niente
			else:
				annotatingPatches(a1,(-.25,74,10,17,5,96,'gray','before',12))
				annotatingPatches(a1,(10.25,74,10,17,15.5,96,'gray','after ',12))
				a1.set_xticks([]) # niente
			a1.set_xlabel('Training Sessions',fontsize=14)
			a1.set_xlim([-.5, 21.2])
			a1.set_ylim([70, 95])
			a1.tick_params(labelsize=10) 

		def creo_a2(a2):
			a2.spines['right'].set_visible(False)
			a2.spines['left'].set_visible(False)
			a2.spines['bottom'].set_visible(False)
			a2.spines['top'].set_visible(False)
			#a2.axes.get_yaxis().set_visible(False)
			a2.set_xticks([]) 
			#
			if 0: # faccio delle gaussiane
				#	
				c = colors #cm.rainbow(np.linspace(0, 1, 4)) # 4 gruppi 
				x = np.linspace(0,100,1000)
				if 1:
					x11 = np.linspace(81,86,1000)
					x12 = np.linspace(81,86,1000)
					x21 = np.linspace(80,85,1000)
					x22 = np.linspace(80,85,1000)
					x31 = np.linspace(78,83,1000)
					x32 = np.linspace(80,85,1000)
					x41 = np.linspace(77,82,1000)
					x42 = np.linspace(79.5,84.5,1000)
				else:
					x11 = x12 = x21 = x22 = x31 = x32 = x41 = x42 = x
				r11 = mlab.normpdf(x11,np.mean(self.ratto_1_pre), stats.sem(self.ratto_1_pre)) 
				r12 = mlab.normpdf(x12,np.mean(self.ratto_1_post), stats.sem(self.ratto_1_post)) 
				r21 = mlab.normpdf(x21,np.mean(self.ratto_2_pre), stats.sem(self.ratto_2_pre)) 
				r22 = mlab.normpdf(x22,np.mean(self.ratto_2_post), stats.sem(self.ratto_2_post)) 
				r31 = mlab.normpdf(x31,np.mean(self.ratto_3_pre), stats.sem(self.ratto_3_pre)) 
				r32 = mlab.normpdf(x32,np.mean(self.ratto_3_post), stats.sem(self.ratto_3_post)) 
				r41 = mlab.normpdf(x41,np.mean(self.ratto_4_pre), stats.sem(self.ratto_4_pre)) 
				r42 = mlab.normpdf(x42,np.mean(self.ratto_4_post), stats.sem(self.ratto_4_post)) 
				a2.plot(r11+0, x11, color = c[0], linestyle='-', linewidth=1.5,alpha=0.8)
				a2.plot(r21+2, x21, color = c[1], linestyle='-', linewidth=1.5,alpha=0.8)
				a2.plot(r31+1, x31, color = c[2], linestyle='-', linewidth=1.5,alpha=0.8)
				a2.plot(r41+3, x41, color = c[3], linestyle='-', linewidth=1.5,alpha=0.8)
				a2.plot(r12+0, x12, color = c[0], linestyle='-', linewidth=1,alpha=0.5)
				a2.plot(r22+2, x22, color = c[1], linestyle='-', linewidth=1,alpha=0.5)
				a2.plot(r32+1, x32, color = c[2], linestyle='-', linewidth=1,alpha=0.5)
				a2.plot(r42+3, x42, color = c[3], linestyle='-', linewidth=1,alpha=0.5)

				xx1,yy1 = range_brace(-0.5,2.2)
				xx2,yy2 = range_brace(1.8,4)
				a2.plot(xx1,1.5*(yy1)+89,color='gray')
				a2.plot(xx2,1.5*(-yy2)+75,color='gray')
				a2.text(-0.6,93,"dyed rats",fontsize=6)
				a2.text(-0.6,69,"control rats",fontsize=6)

				#a2.set_xlabel('PDF')
				#a2.get_yaxis().tick_left()
				a2.set_ylim([78, 85])
			else: # faccio dei barplot
				def permutoPerf(pre,post):
					dp = []
					for r in itertools.product(pre,post):	
						dp.append(r[1]-r[0])
					return dp
				dr1 = permutoPerf(self.ratto_1_pre,self.ratto_1_post)
				dr2 = permutoPerf(self.ratto_2_pre,self.ratto_2_post)
				dr3 = permutoPerf(self.ratto_3_pre,self.ratto_3_post)
				dr4 = permutoPerf(self.ratto_4_pre,self.ratto_4_post)
				dr = [dr2,dr4,dr1,dr3] 
				# calcolo intervallo di confidenza
				def bootstrap(data, num_samples, statistic, alpha):
					"""Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
					n = len(data)
					idx = np.random.randint(0, n, (num_samples, n))
					print n, '\n----\n----\n', idx
					samples = [data[i] for i in idx] 
					stat = np.sort(statistic(samples, 1))
					return (stat[int((alpha/2.0)*num_samples)],
							stat[int((1-alpha/2.0)*num_samples)])
				ci = []
				ci.append(bootstrap(np.asarray(dr2), 1000, np.mean, 0.05))
				ci.append(bootstrap(np.asarray(dr4), 1000, np.mean, 0.05))
				ci.append(bootstrap(np.asarray(dr1), 1000, np.mean, 0.05))
				ci.append(bootstrap(np.asarray(dr3), 1000, np.mean, 0.05))
				print ci # due ratti hanno migliorato le performance in modo significativo...
				# faccio il violinplot
				violin_parts = a2.violinplot(dr,showextrema=False,showmeans=False,showmedians=True)
				for pc,c in zip(violin_parts['bodies'],colors):
					pc.set_facecolor(c)
					pc.set_edgecolor('black')
				annotatingPatches(a2,( .6,-12,1.8,25,1.5,20,'gray','sham\nrats',12))
				annotatingPatches(a2,(2.6,-12,1.8,25,3.5,20,'red','dyed\nrats',12))
				a2.set_ylabel('Difference [%]',fontsize=14)
				a2.set_yticks(np.arange(-10,11,5))
				a2.tick_params(labelsize=11) 
				a2.set_ylim([-12, 15])
				a2.set_xlim([0, 5])

		# luce bianca - luce blu filtro rosso - luce blu  filtro passa lungo 
		# 				effetto sulla behavioral performance
		directory = DATA_PATH+'/analisi-baffo/'
		self.luceBianca 		= directory+'IMG_0239.JPG'
		self.luceBluFiltroRosso = directory+'IMG_0238.JPG'
		self.luceBluFiltroPLung = directory+'IMG_0236.JPG'
		LB=mpimg.imread(self.luceBianca)
		FR=mpimg.imread(self.luceBluFiltroRosso)
		FPL=mpimg.imread(self.luceBluFiltroPLung)
		#textSize, labelSize = fontSizeOnFigures(True)
		FS = (10,6) # dimensione figura
		fW = plt.figure(figsize=FS)
		aw1 = fW.add_subplot(2,3,1)
		aw2 = fW.add_subplot(2,3,2)
		aw3 = fW.add_subplot(2,3,3)
		aw1.imshow(LB)
		aw2.imshow(FPL)
		aw3.imshow(FR)
		def unvisibleAxes(ax):
			ax.axes.get_xaxis().set_visible(False)
			ax.axes.get_yaxis().set_visible(False)
		[unvisibleAxes(ax) for ax in [aw1,aw2,aw3]] 
		gs = gridspec.GridSpec(2,3,width_ratios=[0,6,1])
		aw41 = fW.add_subplot(gs[4])
		aw42 = fW.add_subplot(gs[5]) #,sharey=aw41)
		creo_a2(aw42)
		creo_a1(aw41)

		#	
		fW.tight_layout()
		fW.subplots_adjust(wspace=0.15,hspace=0.5)
		if salva:
			fW.savefig(DATA_PATH+'/elab_video/baseFigura1.pdf',dpi=300)
		else:
			plt.show()

class confrontoBaffiDiversi: # elaboro le diverse sessioni fra loro
	def __init__(self,name,testType,stampoFigura):
		self.name					= name
		self.testType 				= testType
		self.completeName 			= self.name+'_'+self.testType
		self.pickleEndTracking		= '.pickle'
		self.pickleEndSpectrum		= '_spectrum.pickle'
		self.pickleEndTransFun		= '_transferFunction.pickle'
		self.pickleNameInfoWhiskers = DATA_PATH+'/elab_video/'+self.completeName+'_infoWhiskers.pickle'
		self.integrale_lunghezza = []
		self.integrale_absAngolo = []

		#if os.path.isfile(self.pickleName):
		#	print 'il file '+self.pickleName+' esiste'
		#	self.loadWhiskers()

		# modifiche al volo di variabili che non devo precalcolare
		if self.testType == 'diversiBaffi':
			self.listaWhisker = [\
								#DATA_PATH+'/elab_video/a11_12May_',\
								#DATA_PATH+'/elab_video/c11_12May_',\
								DATA_PATH+'/elab_video/c12_12May_',\
								DATA_PATH+'/elab_video/c22_12May_',\
								DATA_PATH+'/elab_video/d11_12May_',\
								DATA_PATH+'/elab_video/c21_12May_',\
								DATA_PATH+'/elab_video/c31_12May_',\
								DATA_PATH+'/elab_video/d21_12May_',\
								DATA_PATH+'/elab_video/d22_12May_',\
								DATA_PATH+'/elab_video/a31_12May_',\
								DATA_PATH+'/elab_video/b11_12May_',\
								DATA_PATH+'/elab_video/c41_12May_',\
								# 			in d31 e` comparsa una imperfezione dopo la colorazione
								#DATA_PATH+'/elab_video/d31_12May_',\
								#DATA_PATH+'/elab_video/a41_12May_',\
								#DATA_PATH+'/elab_video/c51_12May_',\
								]
			# creo le due liste di cose da confrontare
			self.listaWhisker1 = []
			self.listaWhisker2 = []
			for lW in self.listaWhisker:
				self.listaWhisker1.append(lW+'_NONcolor_')
			for lW in self.listaWhisker:
				self.listaWhisker2.append(lW+'_color_')
			
			self.group1 = ['$A1_L$','$C1_L$','$C1_R$','$C2_R$','$D1_L$']
			self.group2 = ['$C2_L$','$C3_L$','$D2_L$','$D2_R$']
			self.group3 = ['$A3_L$','$B1_L$','$C4_L$','$D3_L$']
			self.group4 = ['$A4_L$','$C5_L$']
			self.ROOT = self.group1 + self.group2 + self.group3 #+ self.group4
			'''
			for lW in self.listaWhisker:
				self.ROOT.append(lW[14:17])
			'''
		elif self.testType == 'diversiTempi': 
			
			self.group1 = ['$A1_L$','$C1_L$','$C2_R$','$D1_L$'] # ne manca uno, perche`???`
			self.group2 = ['$C2_L$','$C3_L$','$D2_L$','$D2_R$']
			self.group3 = ['$A3_L$','$B1_L$','$C4_L$','$D3_L$']
			self.group4 = ['$A4_L$','$C5_L$']
			self.ROOT =	['$cut$','$+1h$','$+2h$','$+3h$','$+4h$','$+5h$','$+6h$','$+7h$','$+1d$','$dye$','$+2M$','$+3M$','$pol$']+self.group1 + self.group2 + self.group3 + self.group4
			'''
			self.ROOT =	['@cut','+1h','+2h','+3h','+4h','+5h','+6h','+7h','+1d','@color','+2M','+3M','@polish',\
						 'a11','c11','c22','d11','c21','c31','d21','d22','a31','b11','c41','d31','a41','c51']
			'''
			self.listaWhisker1 = [	DATA_PATH+'/elab_video/c31_11May_hour1__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour2__NONcolor_',\
								  	DATA_PATH+'/elab_video/c31_11May_hour3__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour4__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour5__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour6__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour7__NONcolor_',
									DATA_PATH+'/elab_video/c31_11May_hour8__NONcolor_',\
									\
									DATA_PATH+'/elab_video/c31_12May__NONcolor_',\
									DATA_PATH+'/elab_video/c31_12May__color_',\
									DATA_PATH+'/elab_video/c31_6Jul__color_',\
									DATA_PATH+'/elab_video/c31_2Ago_senzaSmaltoTrasparente__color_',\
									DATA_PATH+'/elab_video/c31_2Ago_conSmaltoTrasparente__color_',\
									\
									DATA_PATH+'/elab_video/a11_12May__color_',\
									DATA_PATH+'/elab_video/c11_12May__color_',\
									DATA_PATH+'/elab_video/c22_12May__color_',\
									DATA_PATH+'/elab_video/d11_12May__color_',\
									DATA_PATH+'/elab_video/c21_12May__color_',\
									DATA_PATH+'/elab_video/c31_12May__color_',\
									DATA_PATH+'/elab_video/d21_12May__color_',\
									DATA_PATH+'/elab_video/d22_12May__color_',\
									DATA_PATH+'/elab_video/a31_12May__color_',\
									DATA_PATH+'/elab_video/b11_12May__color_',\
									DATA_PATH+'/elab_video/c41_12May__color_',\
									DATA_PATH+'/elab_video/d31_12May__color_',\
									DATA_PATH+'/elab_video/a41_12May__color_',\
									DATA_PATH+'/elab_video/c51_12May__color_']
			self.listaWhisker2 = [	DATA_PATH+'/elab_video/c31_11May_hour1__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour2__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour3__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour4__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour5__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour6__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour7__NONcolor_',\
									DATA_PATH+'/elab_video/c31_11May_hour8__NONcolor_',\
									\
									DATA_PATH+'/elab_video/c31_12May__NONcolor_',\
									DATA_PATH+'/elab_video/c31_12May__color_',\
									DATA_PATH+'/elab_video/c31_6Jul__color_',\
									DATA_PATH+'/elab_video/c31_2Ago_senzaSmaltoTrasparente__color_',\
									DATA_PATH+'/elab_video/c31_2Ago_conSmaltoTrasparente__color_',\
									\
									DATA_PATH+'/elab_video/a11_6Jul__color_',\
									DATA_PATH+'/elab_video/c11_6Jul__color_',\
									DATA_PATH+'/elab_video/c22_6Jul__color_',\
									DATA_PATH+'/elab_video/d11_6Jul__color_',\
									DATA_PATH+'/elab_video/c21_6Jul__color_',\
									DATA_PATH+'/elab_video/c31_6Jul__color_',\
									DATA_PATH+'/elab_video/d21_6Jul__color_',\
									DATA_PATH+'/elab_video/d22_6Jul__color_',\
									DATA_PATH+'/elab_video/a31_6Jul__color_',\
									DATA_PATH+'/elab_video/b11_6Jul__color_',\
									DATA_PATH+'/elab_video/c41_6Jul__color_',\
									DATA_PATH+'/elab_video/d31_6Jul__color_',\
									DATA_PATH+'/elab_video/a41_6Jul__color_',\
									DATA_PATH+'/elab_video/c51_6Jul__color_']

			self.group1 = ['0  m','12 m','24 m','36 m','48 m','60 m','72 m','84 m','1 day','@color','2 M'] 	# stesso baffo dati nel tempo
			self.group2 = ['another 2 M']																	# baffi diversi dati prima dopo 2 mesi
			self.group3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]								# 0 dati nel tempo 1 dati diversi baffi
			self.group4 = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]								# 0 dati giorno di estrazione e colorazione meno uno (per fare le differenze) 
			# piccolo check
			if self.ROOT.__len__() is not self.listaWhisker1.__len__():
				print 'err1'
				err1
			if self.ROOT.__len__() is not self.listaWhisker2.__len__():
				print 'err2' 
				err2
			if self.ROOT.__len__() == (self.group1.__len__()+self.group2.__len__()):
				print 'err3'
				err3
			if self.ROOT.__len__() is not self.group3.__len__():
				print 'err4'
				err4
			if self.ROOT.__len__() is not self.group4.__len__():
				print 'err5'
				err5
			for l,m,n in zip(self.listaWhisker1, self.listaWhisker2, self.ROOT):
				print l,m,n
				
		else:
			print 'questo test NON esiste. Correggere self.testType'
			return

		if stampoFigura:
			self.doComparisons()


	def checkBaseTracking(self): # alcune misure vengono male, non e` che dipende da un errore nella stima della base?
		def loadTracking(fname): 
			with open(fname, 'rb') as f:
				return pickle.load(f)[0]
		for lW1,lW2 in zip(self.listaWhisker1,self.listaWhisker2): # non color
			noncolorTrack = loadTracking(lW1+self.pickleEndTracking)
			colorTrack    = loadTracking(lW2+self.pickleEndTracking)
			noncolorBaseTrack = []
			colorBaseTrack = []
			baseVarNoncolor = []
			baseVarColor = []
			for v1,v2 in zip(noncolorTrack,colorTrack): # scorro i video
				traiettorie,nPunti,nCampioni = (v1.wst,v1.wst.__len__(),v1.wst[0].__len__())
				base = traiettorie[nPunti-1]
				baseVarNoncolor.append(np.var(base))
				noncolorBaseTrack.append(base)
				traiettorie,nPunti,nCampioni = (v2.wst,v2.wst.__len__(),v2.wst[0].__len__())
				base = traiettorie[nPunti-1]
				baseVarColor.append(np.var(base))
				colorBaseTrack.append(base)
			print lW1, np.mean(baseVarNoncolor)
			print lW2, np.mean(baseVarColor)
			print '\n~~~~~~~~~~~~~~\n'
			f = plt.figure()
			a1 = f.add_subplot(2,1,1)	
			a2 = f.add_subplot(2,1,2)	
			a1.plot(np.asarray(noncolorBaseTrack).T)
			a2.plot(np.asarray(colorBaseTrack).T)
			a1.set_title(lW1)
			lW = lW1[:-len('__NONcolor_')]
			plt.savefig(lW1+'mecojoni.pdf')
			#colorTrack
			#
		

	def doComparisons(self):
		# carico o calcolo le informazioni geometriche del whisker - self.getInfoWhiskers
		def loadSessionVideos(fname):
			with open(fname, 'rb') as f:
				return pickle.load(f)[0][0] # prendo il primo fra i video
		if self.testType == 'diversiBaffi':
			if os.path.isfile(self.pickleNameInfoWhiskers):
				self.loadWhiskersInfo()
			else:
				for lW1 in self.listaWhisker1: # le due liste sono ridondanti
					v = loadSessionVideos(lW1+self.pickleEndTracking)
					#print v.avi[0:10] 	# NON voglio ricalcolare tutto, ma potrebbero esserci path sbagliati nei pickle. 
										# li correggo a manazza
					if v.avi[0:10] == '../ratto1/': # lasciare per cortesia tutta la stringa per capire cosa faccio
						v.avi = v.avi[10:] 			 # eliminato quel prefisso
						v.avi = DATA_PATH+'/ratto1/'+v.avi # aggiunto il nuovo prefisso
					#print lW1+self.pickleEndTracking
					#print v.avi
					if v.avi.find('_NONcolor')>-1:
						self.getInfoWhiskers(v) 	# calcolo le informazioni sul baffo dalla deformata
				#print self.integrale_lunghezza
				self.saveWhiskersInfo()
				self.loadWhiskersInfo()


		'''	
		if not os.path.isfile(self.pickleName):
			self.saveWhiskers()
			self.loadWhiskers()
		'''
		self.compareWhiskers()

	def getInfoWhiskers(self,video):
		print 'elaboro la deformata da questo video: ',video.avi
		X,Y = video.getBeamShape()
		nans, x = np.isnan(X), lambda z: z.nonzero()[0]
		X[nans] = np.interp(x(nans), x(~nans), X[~nans])	
		nans, x = np.isnan(Y), lambda z: z.nonzero()[0]
		Y[nans] = np.interp(x(nans), x(~nans), Y[~nans])	
		px_mm = 6.9 # calcolato con matlab, inquadratura fissa
					# 1 mm sono circa 7 pixel
		X,Y = (X/px_mm, Y/px_mm)
		l=0  # integrale lunghezza
		s1=0 # integrale angolo normalizzato
		for i in xrange(1,X.__len__()):
			x, xp = (X[i],X[i-1])
			y, yp = (Y[i],Y[i-1])
			dwhisk = np.sqrt(np.power(x-xp,2)+np.power(y-yp,2))
			angle = np.arcsin((y-yp)/dwhisk)*dwhisk # e se lo pesassi con il delta baffo?
			l += dwhisk 
			s1 += np.abs(angle)
		angle0 = np.arcsin((Y[-1]-Y[0])/l)*l # angolo medio... (se il baffo e` dritto ma montato non orizzontale)
		self.integrale_lunghezza.append(l)
		self.integrale_absAngolo.append(s1-angle0)

	def compareWhiskers(self,var2compare='transferFunction'): #'spettri'):  #
		def loadPickle(fname):
			with open(fname, 'rb') as f:
				return pickle.load(f)	

		if var2compare == 'spettri': # pickleEndSpectrum indici: lista sessioni - variabili (freq,spectrum) - dimensioni variabile... 
			lista1 = [loadPickle(f+self.pickleEndSpectrum) for f in self.listaWhisker1]
			lista2 = [loadPickle(f+self.pickleEndSpectrum) for f in self.listaWhisker2]
			print len(lista1), len(lista1[0][1]), len(lista1[0][1][0]) 
			var2compare1 = [l[1][:,:800] for l in lista1]
			var2compare2 = [l[1][:,:800] for l in lista2]
		elif var2compare == 'transferFunction': 
			lista1 = [loadPickle(f+self.pickleEndTransFun) for f in self.listaWhisker1]
			lista2 = [loadPickle(f+self.pickleEndTransFun) for f in self.listaWhisker2]
			var2compare1 = [l[0][:,:350] for l in lista1] # fino a 350Hz
			var2compare2 = [l[0][:,:350] for l in lista2] # fino a 350Hz
		else: 
			print 'quale variabile si vuole comparare?'
			errore
		self.CORR2_undyed 	= self.comparisonKernel(var2compare1,var2compare1)
		self.CORR2_dyed 	= self.comparisonKernel(var2compare2,var2compare2)
		self.CORR2 			= self.comparisonKernel(var2compare1,var2compare2)


		# FIXME!!!

		f = plt.figure()
		a1 = f.add_subplot(2,2,1)
		a2 = f.add_subplot(2,2,2)
		a3 = f.add_subplot(2,2,3)
		a4 = f.add_subplot(2,2,4)
		a1.imshow(self.CORR2_undyed,aspect='equal', interpolation="nearest",clim=(0,1))
		a2.imshow(self.CORR2       ,aspect='equal', interpolation="nearest",clim=(0,1))
		a3.imshow(self.CORR2_dyed  ,aspect='equal', interpolation="nearest",clim=(0,1))
		a4.plot(([M[i] for M,i in zip(self.CORR2,xrange(0,len(self.CORR2)))]))

		plt.savefig(DATA_PATH+'/elab_video/maialedeh'+var2compare+self.testType+'.pdf')
		

	def comparisonKernel(self,var1,var2): 
		N = var1.__len__()
		CORR2 = np.zeros((N,N))
		print N
		for i in xrange(0,var1.__len__()):
			for j in xrange(0,var2.__len__()): 
				CORR = np.corrcoef( var1[i].reshape(-1),\
									var2[j].reshape(-1))[0,1]
				CORR2[i,j] = np.power(CORR,2) 
		return CORR2

		'''
		N = self.ROOT.__len__()
		CORR2 			= np.zeros((N,N))
		CORR2_undyed	= np.zeros((N,N))
		CORR2_dyed		= np.zeros((N,N))
		DIFF_PERC_MAT  = np.ones((N,N))
		for i1 in xrange(0,N):
			print '\n baffo da confrontare = ', self.ROOT[i1], i1 
			data_c = data_c_i1  = var2[i1].reshape(-1)
			data_nc_i1 = var1[i1].reshape(-1)
			for i2 in xrange(0,N):
				data_c_i2  = var2[i2].reshape(-1)
				data_nc = data_nc_i2 = var1[i2].reshape(-1)
				# CORR2	
				CORR2[i1,i2] = np.power(np.corrcoef(data_c,data_nc)[0,1],2) 	# calcolo il quadrato cosi` la metrica va da 0 ad 1 
				CORR2_undyed[i1,i2] = np.power(np.corrcoef(data_nc_i1,data_nc_i2)[0,1],2) # confronto solo non colorati
				CORR2_dyed[i1,i2]   = np.power(np.corrcoef(data_c_i1,data_c_i2)[0,1],2) # confronto solo colorati
				# DIFF_PERC_MAT
				data_c /= np.max(data_c)
				data_nc /= np.max(data_nc)
				norma = np.sqrt(np.sum(np.power(data_c - data_nc,2))) 				# questa e` la norma
				DIFF_PERC_MAT[i1,i2] = norma/np.sqrt(data_c.__len__())			# normalizzo la norma per il suo valore massimo, ovvero sqrt(dim)
				print ' ',i2,
		return (CORR2,CORR2_undyed,CORR2_dyed), DIFF_PERC_MAT # la diff e` obsoleta...
		'''	

	def saveWhiskersInfo(self):
		with open(self.pickleNameInfoWhiskers, 'w') as f:
			pickle.dump([self.integrale_lunghezza,self.integrale_absAngolo], f)	

	def loadWhiskersInfo(self):
		with open(self.pickleNameInfoWhiskers, 'rb') as f:
			self.integrale_lunghezza, self.integrale_absAngolo = pickle.load(f)



	'''
	def calcoloTransferFunctionMedia(self,Videos):
		TFS = []
		for v in self.V: # scorro i video
			traiettorie,nPunti,nCampioni = (v.wst,v.wst.__len__(),v.wst[0].__len__())
			ingresso = traiettorie[nPunti-1] 
			Ingresso = (2.0/nCampioni)*np.abs(fft.fft(ingresso))
			f,Sxx = signal.csd(ingresso,ingresso,2000.0,nperseg=2000,scaling='spectrum')
			TF = []
			for t in traiettorie: 
				f,Syy = signal.csd(t,t,2000.0,nperseg=2000,scaling='density')          #scaling='spectrum'
				f,Syx = signal.csd(t,ingresso,2000.0,nperseg=2000,scaling='density')
				f,Sxy = signal.csd(ingresso,t,2000.0,nperseg=2000,scaling='density')
				H1 = [ syx/sxx  for sxx,syx in zip(Sxx,Syx)]
				H2 = [ syy/sxy  for syy,sxy in zip(Syy,Sxy)]
				H = [ np.sqrt(h1*h2) for h1,h2 in zip(H1,H2)]
				TF.append(np.abs(H))
			TFS.append(TF)
		return np.mean(TFS,axis=0)

	def calcoloSpettroMedio(self,Videos):
		spettri = []
		for v,i in zip(Videos,xrange(0,Videos.__len__())):
			# gia che ci sono prendiamo i dati sul primo frame
			#print v.wst.__len__()   	# 100
			#print v.wst[0].__len__()	# 4500
			#v.wst[:,0]	# <-- primo frame 
			if self.doInfoWhiskersFig:
				if i==0 and v.avi.find('_NONcolor')>-1:
					self.getInfoWhiskers(v) 	# calcolo le informazioni sul baffo dalla deformata
			spettri.append(v.WSF) 
		return np.mean(spettri,axis=0)[0:-1,0:1400]

	def saveWhiskers(self):
		self.AvSp_ncol = []
		self.AvSp_col = []
		for lW1,lW2 in zip(self.listaWhisker1,self.listaWhisker2):
			SM = self.calcoloSpettroMedio(self.getOneOnlyVar(lW1))
			self.AvSp_ncol.append(SM)
			if lW1 == lW2: # XXX questa cosa succede per come e` scritto il codice per self.testType == 'diversiTempi'
				self.AvSp_col.append(SM)
			else:
				self.AvSp_col.append(self.calcoloSpettroMedio(self.getOneOnlyVar(lW2)))
		with open(self.pickleName, 'w') as f:
			pickle.dump([self.ROOT, self.AvSp_col, self.AvSp_ncol, self.integrale_lunghezza, self.integrale_absAngolo, self.group1, self.group2, self.group3, self.group4], f)	

	def loadWhiskers(self):
		with open(self.pickleName, 'rb') as f:
			self.ROOT, self.AvSp_col, self.AvSp_ncol, self.integrale_lunghezza, self.integrale_absAngolo, self.group1, self.group2, self.group3, self.group4 = pickle.load(f)

	def getOneOnlyVar(self,pickleName,idVar=0):
		with open(pickleName,'rb') as f:
			toLoad = pickle.load(f)
		return toLoad[idVar]
	'''

class sessione: # una sessione e` caratterizzata da tanti video
	def __init__(self,whiskerName,recordingDate,colorNonColor_status,path,ROI,videoThs,videoShow=True,go=True,justPlotRaw=False,overWriteElab=False):	
		self.name    	= whiskerName                               		# campione
		self.date    	= recordingDate                             		# giorno
		self.status  	= colorNonColor_status                      		# stato campione
		self.path 		= path
		self.ROI 		= ROI
		self.videoThs  	= videoThs
		self.videoShow 	= videoShow
		self.justPlotRaw = justPlotRaw
		self.overWriteElab = overWriteElab
		if self.justPlotRaw is True:
			self.overWriteElab = False
		self.id_name 					= self.name+'_'+self.date+'_'+self.status 
		self.pickleNameTracking			= DATA_PATH+'/elab_video/'+self.id_name+'.pickle'
		self.pickleNameSpectrum			= DATA_PATH+'/elab_video/'+self.id_name+'_spectrum.pickle'
		self.pickleNameTransFun			= DATA_PATH+'/elab_video/'+self.id_name+'_transferFunction.pickle'
		self.spettroMedName 			= DATA_PATH+'/elab_video/'+self.id_name+'_spectrum.pdf'
		self.transferFunctionMedName 	= DATA_PATH+'/elab_video/'+self.id_name+'_transferFunction.pdf'
		self.fig1Name 					= DATA_PATH+'/elab_video/'+self.id_name+'_test_fig1'
		if go: #  tutto in un colpo solo
			self.elaboroFilmati()
			self.doTestFig1()				
			self.calcoloTransferFunction()  
			self.calcoloSpettroMedio()		

	def elaboroFilmati(self): 
		if self.justPlotRaw is False:
			if self.overWriteElab is False:
				if os.path.isfile(self.pickleNameTracking):
					print 'il file '+self.pickleNameTracking+' esiste'
					return False
		self.resolvePath(self.path)                                   						# identifico quali sono i video da analizzare 
		print self.path
		self.V = [video(al,self.ROI,self.videoThs,self.videoShow,self.justPlotRaw) for al in self.aviList]	# XXX PROCESSING XXX: analizzo i filmati # TODO aggiungere il multithreading		
		self.saveTracking()
		return True

	def doTestFig1(self):
		self.loadTracking() # carico i dati
		for v,i in zip(self.V,xrange(0,self.V.__len__())): 
			fname = self.fig1Name+'_tr'+str(i)+'.pdf'
			if os.path.isfile(fname):
				print 'il file '+fname+' esiste'
			else:
				v.test_fig1(True,fname) 

	def calcoloTransferFunction(self,evalFigure=True): 
		if os.path.isfile(self.pickleNameTransFun):
			print 'il file '+self.pickleNameTransFun+' esiste'
		else:
			self.loadTracking() # carico i dati
			self.TF = []
			for v in self.V: # scorro i video
				traiettorie,nPunti,nCampioni = (v.wst,v.wst.__len__(),v.wst[0].__len__())
				ingresso = traiettorie[nPunti-1] 
				Ingresso = (2.0/nCampioni)*np.abs(fft.fft(ingresso))
				f,Sxx = signal.csd(ingresso,ingresso,2000.0,nperseg=2000,scaling='spectrum')
				TF = []
				for t in traiettorie: 
					if 0: # provo a togliere la media
						t = t-ingresso
						t = t-np.mean(t)
					f,Syy = signal.csd(t,t,2000.0,nperseg=2000,scaling='density')          #scaling='spectrum'
					f,Syx = signal.csd(t,ingresso,2000.0,nperseg=2000,scaling='density')
					f,Sxy = signal.csd(ingresso,t,2000.0,nperseg=2000,scaling='density')
					H1 = [ syx/sxx  for sxx,syx in zip(Sxx,Syx)]
					H2 = [ syy/sxy  for syy,sxy in zip(Syy,Sxy)]
					H = [ np.sqrt(h1*h2) for h1,h2 in zip(H1,H2)]
					TF.append(np.abs(H))
				self.TF.append(TF)
			self.TFM = np.mean(self.TF,axis=0)
			self.saveTransferFunction()
		self.loadTransferFunction()
		if evalFigure:
			f0 = [tf[0] for tf in self.TFM]
			f = plt.figure()
			a1 = f.add_subplot(1,1,1)
			cax1 = a1.imshow(np.log10(self.TFM),aspect='auto', interpolation="gaussian",cmap='RdBu_r')#'OrRd')	
			cbar1 = f.colorbar(cax1,ax=a1)
			plt.savefig(self.transferFunctionMedName)
	
	def calcoloSpettroMedio(self,evalFigure=True):
		if os.path.isfile(self.pickleNameSpectrum):
			print 'il file '+self.pickleNameSpectrum+' esiste'
		else:
			self.loadTracking()
			spettri = []
			for v in self.V:
				spettri.append(v.WSF)
			self.spettro_medio = np.mean(spettri,axis=0)
			self.freq = self.V[0].freq
			self.saveSpectrum()
		self.loadSpectrum()	
		if evalFigure:
			ff,a1 = plt.subplots(1)
			a1.imshow(np.log10(self.spettro_medio),aspect='auto', interpolation="nearest")	
			a1.set_xlabel('Freq [Hz]')
			#a1.set_title(self.id_name)
			plt.savefig(self.spettroMedName)

	def resolvePath(self,path):	# trovo gli della sessione richiesta
		self.aviList = glob.glob(path+'*'+self.status+'*.avi')

	def saveTransferFunction(self): 
		with open(self.pickleNameTransFun, 'w') as f:
			pickle.dump([self.TFM], f)	

	def loadTransferFunction(self): 
		with open(self.pickleNameTransFun, 'rb') as f:
			data = pickle.load(f)	
		self.TFM = data[0]

	def saveSpectrum(self): 
		with open(self.pickleNameSpectrum, 'w') as f:
			pickle.dump([self.freq,self.spettro_medio], f)	

	def loadSpectrum(self): 
		with open(self.pickleNameSpectrum, 'rb') as f:
			data = pickle.load(f)	
		self.freq          = data[0]
		self.spettro_medio = data[1]

	def saveTracking(self): 
		with open(self.pickleNameTracking, 'w') as f:
			pickle.dump([self.V], f)	

	def loadTracking(self): 
		with open(self.pickleNameTracking, 'rb') as f:
			data = pickle.load(f)	
		self.V = data[0]

	def printSessionName(self): # test...
		print self.id_name

class video: # ogni fideo va elaborato
	def __init__(self,avi,ROI,videoThs,videoShow,justPlotRaw=False,processAllVideo=True):	
		self.avi= avi						# path del filmato
		self.ROI = ROI						# ROI ottimale per quel filmato 
		self.videoThs = videoThs			# soglia ottimale per quel filmato (binarizzazione)
		self.videoShow = videoShow			# mostro filmato o no (debug)
		self.justPlotRaw = justPlotRaw		# bypasso tutto per mostrare il filmato con box senza nessuna operazione
		self.bw = 2000.0 					# frame/sec - bandwidth
		cap = cv2.VideoCapture(self.avi) 
		self.maxFrame = int(cap.get(7))		# contatore dei frame
		if 0: # quanti frame ho ? 
			print self.maxFrame # == 4500
			fermati
		Np = int(self.maxFrame/2) 			# numero di campioni frequenze positive
		df = (self.bw/2)/Np								# df 
		dt = 1.0/self.bw								# dt
		self.time = [dt*c for c in xrange(0,self.maxFrame)]	
		self.freq = [df*c for c in xrange(0,self.maxFrame/2)]	
		self.N = 100 # XXX era 100 									# punti equidistanziati per il tracking del baffo
		if processAllVideo:
			self.wst = np.zeros((self.N,self.maxFrame)) 	# whisker samples time per ogni cap
			self.elaboroFilmato(cap)				# faccio il tracking
			self.postProcessing()					# abbellisco il tracking con intorpolazione dei NaN e antialias (media mobile)
			self.WSF = self.trasformataBaffo()		# eseguo la trasformata di Fourier dei punti del baffo ricostruiti
			#self.test_fig1() 						# due subplot con overlap tracking e trends nel tempo
			#self.test_fig2()						# un subplot con lo spettro del baffo

	def test_fig1(self,salva=False,name=''): 
		ff, (a1,a2) = plt.subplots(1,2)
		a1.plot(self.time,self.wst.transpose())	
		a2.plot(self.wst)	
		if salva:
			plt.savefig(name)
		else:
			plt.show()

	def test_fig2(self): 
		ff,a1 = plt.subplots(1)
		a1.imshow(self.WSF,aspect='auto', interpolation="nearest")	
		rng = xrange(0,self.freq.__len__(),120)
		idx  = [i for i in rng] 
		freq = [int(self.freq[i]) for i in rng] 
		a1.set_xticks(idx)
		a1.set_xticklabels(freq)
		a1.set_xlabel('Freq [Hz]')
		plt.show()

	def postProcessing(self):
		# print self.wst.__len__()  # 100
		# print self.wst[0].__len__()  # 4500
		self.wst = self.interpoloNan(self.wst.transpose()).transpose() # lungo il tempo 
		self.antiAliasing()

	def interpoloNan(self,Mat):
		res = np.copy(Mat)
		for i in xrange(0,res.__len__()): # elimino il problema dei nan
			nans, x= np.isnan(res[i]), lambda z: z.nonzero()[0]
			res[i][nans]= np.interp(x(nans), x(~nans), res[i][~nans])	
		return res
	
	def antiAliasing(self):
		for i in xrange(0,self.wst.__len__()): 
			self.wst[i] = np.convolve(self.wst[i], np.ones((3,))/3,mode='same')
	
	def elaboroFilmato(self,cap): 
		#print aviPaths
		fn=-1
		while fn<self.maxFrame-1: # 100: #	
			fn+=1
			#~~~~~~~~~~ INIZIO ANALISI ~~~~~~~~~~#
			_,Read = cap.read() 
			if self.justPlotRaw:
				Frame = Read
			else:
				Read,_,Frame = self.doElabFrame(fn,self.N,Read)
			#~~~~~~~~~~ FINE   ANALISI ~~~~~~~~~~#
			if self.videoShow: # mostro i filmati 
				x,y,w,h = ( self.ROI[0]			   ,self.ROI[2],\
							self.ROI[1]-self.ROI[0],self.ROI[3]-self.ROI[2])
				cv2.rectangle(Frame, (x,y), (x+w,y+h), 255, 2)
				cv2.imshow('rip',Frame) 
			if not fn%50: # mostro a quale frame mi trovo
				print 'fn == 100'
				print 'Video'+self.avi,' Frame #',fn, '-> videoThs ~= ', int(1.12*np.median(Read)), self.videoThs # XXX prima era cosi`
			if cv2.waitKey(1) & 0xFF == ord('q'): # indispensabile per il corretto funzionamento 
				break
		self.maxFrame = fn

	def getBeamShape(self):
		cap = cv2.VideoCapture(self.avi) 										# calcolo al volo, quindi il cap non lo ho
		_,Read = cap.read()
		Read,_,Frame = self.doElabFrame(0,self.N,Read)
		cv2.imwrite("../elab_video/%s.jpg"%re.sub('/','',self.avi)[2:-4],Read)        # immagine del primo frame salvata su disco
		return self.puntiBaffoEquidistanziati(self.doROI(Frame),0,self.N,3) 	# ri-calcolo i punti sul Frame

	def doElabFrame(self,fn,N,Read): 
		#Frame = Read[1]   														# prendo i frame da tutti i filmati
		Frame = Read   															# prendo i frame da tutti i filmati
		Frame = Frame[10:-1,1:-1]   											# tolgo il frame number
		Frame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY) 						# transformo in scala di grigi
		Frameb = Frame
		Frame_blur = Frameb #cv2.medianBlur(Frameb,3)  											# se filtro l'immagine qui aiuta 
		Frame_ths = cv2.threshold(Frame_blur,self.videoThs,255,cv2.THRESH_BINARY)[1]  	# soglia e divido 
		self.puntiBaffoEquidistanziati(self.doROI(Frame_ths),fn,N,3) 					# calcolo i punti sul Frame
		return Frameb,Frame_blur,Frame_ths


	def doROI(self,f): # inquadro
		return f[self.ROI[2]:self.ROI[3],self.ROI[0]:self.ROI[1]]

	def puntiBaffoEquidistanziati(self,frame,fn,N,Degree): 
		x,y = self.get_whisker(frame)
		x,y = self.norm_whisker(x,y,N,Degree)
		for c in xrange(0,N):
			self.wst[c][fn] = y[c] # qui sto raccogliendo i dati
		if 1: # cambiare il frame serve solo per il plot 
			frame/=3 #10.0 
			for i,j in zip(x,y): 
				if not np.isnan(j):
					frame[j][i] = 255
		return x, y

	def get_whisker(self,frame):
		frame = frame.transpose()
		x = np.zeros(frame.__len__())
		y = np.zeros(frame.__len__())
		c = 0
		for i in frame:
			ths = np.sum(i)
			if ths > 10:
				M = cv2.moments(i)
				try: 
					y[c] = int(M['m01']/M['m00'])
				except :
					if y[c-1]:
						y[c] = y[c-1]
					else: 
						y[c] = np.nan
			else:
				y[c] = np.nan
			x[c] = c
			c += 1
		frame = frame.transpose()
		if 0:
			plt.imshow(frame)
			plt.plot(x,y,'k.')
			plt.show()
			fermati
		return x,y

	def norm_whisker(self,x,y,N,Degree): 
		# elimino la parte di ROI - piena di NaN - a sinistra del baffo
		c = 0
		for i in y: 
			if np.isnan(i):
				c+=1
			else: 
				break
		x = x[c+1:]
		y = y[c+1:]
		cv = self.bspline(zip(x,y), N, Degree, False) # mai periodico (e` un baffo non un anello) 
		cv = cv.transpose()
		return cv[0],cv[1]

	def bspline(self,cv, n=100, degree=3, periodic=False):
		""" Calculate n samples on a bspline
			cv :      Array ov control vertices
			n  :      Number of samples to return
			degree:   Curve degree
			periodic: True - Curve is closed
					  False - Curve is open
		"""
		# If periodic, extend the point array by count+degree+1
		cv = np.asarray(cv)
		count = len(cv)
		if periodic:
			factor, fraction = divmod(count+degree+1, count)
			cv = np.concatenate((cv,) * factor + (cv[:fraction],))
			count = len(cv)
			degree = np.clip(degree,1,degree)
		# If opened, prevent degree from exceeding count-1
		else:
			degree = np.clip(degree,1,count-1)
		# Calculate knot vector
		kv = None
		if periodic:
			kv = np.arange(0-degree,count+degree+degree-1,dtype='int')
		else:
			kv = np.array([0]*degree + range(count-degree+1) + [count-degree]*degree,dtype='int')
		# Calculate query range
		u = np.linspace(periodic,(count-degree),n)
		# Calculate result
		arange = np.arange(len(u))
		points = np.zeros((len(u),cv.shape[1]))
		for i in xrange(cv.shape[1]):
			points[arange,i] = si.splev(u, (kv,cv[:,i],degree))
		return points

	def trasformataBaffo(self): # non voglio fare figure ma ritornare lo spettro per fare analisi prima/dopo la colorazione
		# print self.wst.__len__() # 100
		# print self.wst[0].__len__() # 4500
		traiettorie,nPunti,nCampioni = (self.wst,self.wst.__len__(),self.wst[0].__len__())
		window = (1.0/0.54)*signal.hamming(nCampioni) # coefficiente di ragguaglio = 0.54
		spettri_abs = np.zeros((nPunti,nCampioni/2)) 
		spettri_phs = np.zeros((nPunti,nCampioni/2)) 
		if 0: # provo a non togliere il riferimento alla base
			traiettorie = [t-traiettorie[nPunti-1] for t in traiettorie] # osservo dal punto di vista dello shaker
			traiettorie = [window*(t-np.mean(t)) for t in traiettorie] # finestro usando hamming
		return self.fft_whisker(traiettorie,nCampioni)

	def fft_whisker(self,x,Np): 
		Xabs = np.zeros((x.__len__(),Np/2))
		Xphs = np.zeros((x.__len__(),Np/2))
		for i in range(0,x.__len__()): # si possono accorpare i cicli di nan-removal ed FFT	
			f = fft.fft(x[i])
			Xabs[i] = (2.0/Np)*np.abs(f[0:Np/2])
		return Xabs

	def prova(self):
		print 'stampami questo!'


# i test vanno qui
if __name__ == '__main__': 
	
	# definitiamo i PATH come varaibili globali
	global ELAB_PATH 
	global DATA_PATH 
	ELAB_PATH = os.path.abspath(__file__)[:-len(os.path.basename(__file__))] # io sono qui
	DATA_PATH = '/media/jaky/DATI BAFFO/'
	print '~~~~~~~~~~~~\nNOTA BENE:'
	print 'ELAB_PATH = '+ELAB_PATH
	print 'DATA_PATH = '+DATA_PATH
	print '~~~~~~~~~~~~\n:'

	# ---- PRE - PROCESSING ---- #
	# TRACKING 11 MAGGIO -- c31 nel tempo -- 
	#sessione('c31','11May_hour1','_NONcolor_',DATA_PATH+'/ratto1/c3_1/11May2016/_hour1_/',(331, 625, 120, 245),32,True)   	# tracking molto bello
	#sessione('c31','11May_hour2','_NONcolor_',DATA_PATH+'/ratto1/c3_1/11May2016/_hour2_/',(331, 625, 120, 245),32,True,True,False)   	# tracking molto bello
	#sessione('c31','11May_hour3','_NONcolor_',DATA_PATH+'/ratto1/c3_1/11May2016/_hour3_/',(331, 625, 120, 245),32,True,True,False)   	# tracking molto bello
	#sessione('c31','11May_hour4','_NONcolor_',DATA_PATH+'/ratto1/c3_1/11May2016/_hour4_/',(331, 625, 120, 245),32,True,True,False)   	# tracking molto bello
	#sessione('c31','11May_hour5','_NONcolor_',DATA_PATH+'/ratto1/c3_1/11May2016/_hour5_/',(331, 625, 120, 245),32,True,True,False)   	# tracking molto bello
	#sessione('c31','11May_hour6','_NONcolor_',DATA_PATH+'/ratto1/c3_1/11May2016/_hour6_/',(331, 625, 120, 245),32,True,True,False)   	# tracking molto bello
	#sessione('c31','11May_hour7','_NONcolor_',DATA_PATH+'/ratto1/c3_1/11May2016/_hour7_/',(331, 625, 120, 245),32,True,True,False)   	# tracking molto bello
	#sessione('c31','11May_hour8','_NONcolor_',DATA_PATH+'/ratto1/c3_1/11May2016/_hour8_/',(331, 625, 120, 245),32,True,True,False)   	# tracking molto bello
	# TRACKING 12 MAGGIO
	sessione('a11','12May','_NONcolor_',DATA_PATH+'/ratto1/a1_1/',(280, 630, 0, 200),32,True) #,True,False,True)
	#sessione('a11','12May','_color_',DATA_PATH+'/ratto1/a1_1/',(280, 625, 0, 200),33,True,True,False,True) 		# tracking molto bello
	#sessione('a31','12May','_NONcolor_',DATA_PATH+'/ratto1/a3_1/',(450, 629, 145, 210),32,True)
	#sessione('a31','12May','_color_',DATA_PATH+'/ratto1/a3_1/',(450, 638, 145, 230),32,True)  		# tracking molto bello
	#sessione('a41','12May','_NONcolor_',DATA_PATH+'/ratto1/a4_1/',(542, 634, 0, 245),29,True)
	#sessione('a41','12May','_color_',DATA_PATH+'/ratto1/a4_1/',(542, 635, 0, 245),28,True)
	#sessione('c11','12May','_NONcolor_',DATA_PATH+'/ratto1/c1_1/',(186, 630, 0, 245),33,True)
	#sessione('c11','12May','_color_',DATA_PATH+'/ratto1/c1_1/',(186, 632, 0, 150),33,True)
	#sessione('c21','12May','_NONcolor_',DATA_PATH+'/ratto1/c2_1/',(242, 617, 0, 245),32,True)
	#sessione('c21','12May','_color_',DATA_PATH+'/ratto1/c2_1/',(242, 614, 0, 120),32,True)  		# tracking molto bello
	#sessione('c31','12May','_NONcolor_',DATA_PATH+'/ratto1/c3_1/',(331, 625, 120, 245),32,True)   	# tracking molto bello
	#sessione('c31','12May','_color_',DATA_PATH+'/ratto1/c3_1/',(331, 625, 150, 245),32,True) 		# l'inquadratura ogni tanto perde la punta del baffo! 
	#sessione('c51','12May','_NONcolor_',DATA_PATH+'/ratto1/c5_1/',(480, 630, 120, 220),30,True)   	# tracking molto bello
	#sessione('c51','12May','_color_',DATA_PATH+'/ratto1/c5_1/',(480, 625, 130, 240),30,True)   	# tracking molto bello
	#sessione('d21','12May','_NONcolor_',DATA_PATH+'/ratto1/d2_1/',(310, 629, 50, 210),29,True)		# tracking molto bello
	#sessione('d21','12May','_color_',DATA_PATH+'/ratto1/d2_1/',(310, 629, 120, 240),30,True)		# tracking molto bello
	#sessione('d31','12May','_NONcolor_',DATA_PATH+'/ratto1/d3_1/',(423, 625, 140, 245),29,True)
	#sessione('d31','12May','_color_',DATA_PATH+'/ratto1/d3_1/',(423, 622, 120, 215),32,True)		# e` comparsa una imperfezione...
	#sessione('b11','12May','_NONcolor_',DATA_PATH+'/ratto1/b1_1/',(430, 620, 120, 245),32,True)   	
	#sessione('b11','12May','_color_',DATA_PATH+'/ratto1/b1_1/',(460, 675, 120, 215),30,True)	   	
	#sessione('c12','12May','_NONcolor_',DATA_PATH+'/ratto1/c1_2/',(310, 640, 70, 235),30,True)    	
	#sessione('c12','12May','_color_',DATA_PATH+'/ratto1/c1_2/',(310, 640, 70, 235),30,True)	   
	#sessione('c22','12May','_NONcolor_',DATA_PATH+'/ratto1/c2_2/',(280, 655, 30, 205),30,True)	
	#sessione('c22','12May','_color_',DATA_PATH+'/ratto1/c2_2/',(260, 635, 20, 235),30,True)	
	#sessione('c41','12May','_NONcolor_',DATA_PATH+'/ratto1/c4_1/',(390, 640, 50, 175),29,True)	
	#sessione('c41','12May','_color_',DATA_PATH+'/ratto1/c4_1/',(390, 640, 90, 245),32,True)	
	#sessione('d11','12May','_NONcolor_',DATA_PATH+'/ratto1/d1_1/',(90, 635, 20, 245),32,True)	
	#sessione('d11','12May','_color_',DATA_PATH+'/ratto1/d1_1/',(110, 615, 20, 245),32,True)   	   
	#sessione('d22','12May','_NONcolor_',DATA_PATH+'/ratto1/d2_2/',(210, 635, 20, 245),30,True)	
	#sessione('d22','12May','_color_',DATA_PATH+'/ratto1/d2_2/',(210, 625, 20, 205),30,True)	

	# TRACKING 6 LUGLIO 
	#sessione('a11','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/a1_1/',(250+105, 605+105, 0, 200),32,True,True,False)
	#sessione('a31','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/a3_1/',(450+105, 618+105, 0, 200),32,True,True,False)
	#sessione('a41','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/a4_1/',(540+105, 635+105, 0, 200),30,True,True,False)
	#sessione('c11','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/c1_1/',(170+100, 635+100, 0, 240),32,True,True,False)
	#sessione('c21','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/c2_1/',(260+100, 615+100, 0, 200),32,True,True,False)
	#sessione('c31','6Jul','_color_',DATA_PATH+'/ratto1/c3_1/5Jul2016/',(230+100, 530+100, 0, 200),29,True,True,False)
	#sessione('c51','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/c5_1/',(480+100, 635+100, 0, 200),32,True,True,False)
	#sessione('d21','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/d2_1/',(300+100, 615+100, 0, 200),29,True,True,False)
	#sessione('d31','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/d3_1/',(410+100, 612+100, 100, 230),29,True,True,False)
	#sessione('b11','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/b1_1/',(430+100, 610+100, 100, 230),29,True,True,False)
	#sessione('c12','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/c1_2/',(270+100, 620+100, 100, 240),29,True,True,False)
	#sessione('c22','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/c2_2/',(220+100, 615+100, 0, 240),32,True,True,False)
	#sessione('c41','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/c4_1/',(390+100, 625+100, 100, 200),32,True,True,False)
	#sessione('d11','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/d1_1/',(90+100, 620+100, 0, 200),32,True,True,False)
	#sessione('d22','6Jul','_color_',DATA_PATH+'/ratto1/6Luglio/d2_2/',(210+100, 610+100, 0, 200),29,True,True,False)
	
	#TRACKING 2 AGOSTO
	#sessione('c31','2Ago_senzaSmaltoTrasparente','_color_',DATA_PATH+'/ratto1/c3_1/senzaSmaltoTrasparente/',(300, 660, 50, 205),35,True,True,False) 
	#sessione('c31','2Ago_conSmaltoTrasparente','_color_',DATA_PATH+'/ratto1/c3_1/conSmaltoTrasparente/',(260, 625, 50, 205),35,True,True,False) 

	#TRACKING ACCIAIO 13 APRILE
	#sessione('filo_acciaio','13Apr','_NONcolor_',DATA_PATH+'/ratto1/0_acciaio_no_rot/',(260, 780, 0, 205),33,True) 

	# ROBA DA DARE AD ALE...
	#PickleAsciiTimeTrendsConversion() #<-- da finire serve ad alessandro per il modlelo

	# CALCOLO TRANSFER FUNCTION POST TRACKING
	#a = sessione('d21','12May','_NONcolor_',DATA_PATH+'/ratto1/0_acciaio_no_rot/',(260, 780, 0, 205),33,True, False)
	#a.calcoloTransferFunction(True)

	# CONTROLLO BASE STIMOLO PER OGNI WHISKER
	#a = confrontoBaffiDiversi('baffi_12May','diversiBaffi',False)    
	#a.checkBaseTracking()
	

	# ---- POST - PROCESSING ---- #
	#sessione('d21','12May','_NONcolor_',DATA_PATH+'/ratto1/d2_1/',(310, 629, 50, 210),29,True,True,False,True)		# tracking molto bello
	if 1:
		confrontoBaffiDiversi('baffi_12May','diversiTempi',True)    
		confrontoBaffiDiversi('baffi_12May','diversiBaffi',True)    
	#confrontoAddestramento()						
	#creoSpettriBaffi()								
	#stampo_lunghezza_whiskers()					
	#mergeComparisonsResults()						
	#simulatedAndSetup() 							
	#creoImageProcessing_Stacked()					
	print 'stampo per far fare qualcosa al main'




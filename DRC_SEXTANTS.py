# from tables import open_file 
import h5py 
import numpy as np
from numpy import trapz,nan_to_num,all,diff,interp,flipud,savetxt,genfromtxt,array,max,argmax,zeros,exp,sin,radians,pi,sqrt,polyfit,polyval,arange
import ccdproc
#from re import match,search
from scipy.optimize import curve_fit,minimize
#from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.pylab import subplots,Figure  #close,
import matplotlib.colors as colors
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk #Agg
from scipy import signal
from scipy.ndimage import shift

print( ' '+'*'*52 + '\n'+
    ' *                DRC_SEXTANTS 1.0.10               *\n' +
    ' *  Python code to visualize and treat RIXS data    *\n' +
    ' *            sets generated at SEXTANTS            *\n' +
    ' *               V. Porée, A. Nicolaou              *\n' +
    ' * ------------------------------------------------ *\n' +
    ' *  L.A.Cosmic correction from:                     *\n' +
    ' *   - van Dokkum, 2001, “Cosmic-Ray Rejection by   *\n' +
    ' *   Laplacian Edge Detection”. doi: 10.1086/323894 *\n' +
    ' *   - McCully, C., 2014, “Astro-SCRAPPY”           *\n' +
    ' ' + '*'*52+'\n') #

# def __init__(self):
#     pass

class rixs_image:
    """Has methods to get data on SEXTANTS/ID20/ETC 
    For SEXTANTS has methods to quickly get and plot XAS and 
    RIXS scans. 2D/3D datasets (ie raster scans, data dependent 
    on external parameters etc can be handled IN THE FUTURE"""

    def __init__(self,fname,Talkative=False):
        self.fname=fname
        self.getdata(Talkative=Talkative)
        pass

    def __findcounterid__(self,ctrname):
        if not hasattr(self,'longnames'):
            print("no counters found!")
            pass
        for ind,name in enumerate(self.longnames):
            tmp = name.split('/')
            if len(tmp)==4:
                if ctrname in tmp[2] or ctrname in tmp[3]:
                    return ind


    def getdata(self,ctrlist=list(("energy","keithley","counter01")),loc="sextants",Talkative=False):
        """Extracts the counters and possible CCD images from a nxs file named "fname". 
        The counters are put to the attribute self.data and the image(s) to self.rawimg
        
        Created: 21.02.2023 Victor
        Modified:
        """
        #if loc == "sextants":
        print("-------------------------")
        print("Handling file ",self.fname)
        
        try:
            file = h5py.File(self.fname,'r')

            tmpfile = file[list(file.keys())[0]]
 
            # file.keys()
            
            tmpdata = tmpfile.get('scan_data')
            
            # tmpfile = h5py.File(self.fname,'r')
            # tmpdata = tmpfile.root.__getattr__(tmpfile.root.__members__[0])
            self.longnames=list()
            self.ctrnames = list()
            self._hasdata = False
            self._hasimg = False
            self._ctrlen = 0
            self._ctrn = 0
            self._hascurvature = False
            try:
                self.Ei_mono = np.array((tmpfile.get('SEXTANTS/mono/energy')))[0]
            except:
                self.Ei_mono = np.array((tmpfile.get('SEXTANTS/Monochromator/energy')))
            print("Incident energy: ",self.Ei_mono,"eV")
            
            try:
                self.Rz = np.array(tmpfile.get('SEXTANTS/i14-m-cx1__ex__sample.2-rz/raw_value'))[0]
                self.Ts = np.array(tmpfile.get('SEXTANTS/i14-m-cx1__ex__sample.2-ts/raw_value'))[0]
                self.Tx = np.array(tmpfile.get('SEXTANTS/i14-m-cx1__ex__sample.2-tx/raw_value'))[0]
                self.Tz = np.array(tmpfile.get('SEXTANTS/i14-m-cx1__ex__sample.2-tz/raw_value'))[0]
                # self.tstart  = tmpdata.start_time.read() # causing some errors sometimes ; to investigate !
            except:
                 if Talkative == True:
                     print('No sample orientation info.')       
            for it in tmpdata:
                if Talkative==True:
                    print("Counter ",it," found.")
                tmpo = tmpdata.get(it)
                self.longnames.append(str(tmpo.attrs['long_name']))
                
                if tmpo.ndim == 1:
                    self._hasdata = True
                    self._ctrn += 1
                    self._ctrlen = tmpo.len()    
                    self.ctrnames.append(it)
                    
                if tmpo.ndim > 1:
                    if Talkative==True:
                        print("Detector image found")
                    self._hasimg = True
                    self._imshape = tmpo.shape
                    self.ctrnames.append(it)
                    
            if self._ctrlen > 0:
                self.data = np.zeros((self._ctrlen,self._ctrn))
                if Talkative==True:
                    print("Created counter array data with shape (",self._ctrlen,",",self._ctrn,")")
                
            if self._hasimg ==True:
                self.rawimg = np.zeros(self._imshape)
                if Talkative==True:
                    print ("Created image array rawimg  with shape ",self._imshape)
                        
            ctrind = 0
                        
            for it in tmpdata:
                tmpo = tmpdata.get(it)
                if tmpo.ndim  <2:
                    self.data[:,ctrind] = np.array(tmpo)
                    ctrind += 1

                elif tmpo.ndim > 1:
                    self.rawimg = np.array(tmpo)
                    self.integrationtimes = np.array(tmpfile.get('scan_data/integration_times/')) # store acquisition time of each image for subsequent normalisation 
            print("The file contains ",self._imshape[0],"image(s)")    
            if Talkative==True:
                print("------------")
                print("Closing file")
            file.close()       
                
        except Exception as e:
            if Talkative == True:
                print('Ignoring Exception: "', e,'"')

    def plotdata(self,idx=list((0,1))):
        """plots columns of the data matrix. 
        give indices of the columns with the keyword variable idx,
        where the first column represents the x axis
        Columns 0 and 1 are plotted by default"""
        if not hasattr(self,'data'):
            print("No counters found!")
            pass
        else: 
            self.fig1,self.ax1 =subplots()
            for i in range(1,len(idx)):
                self.ax1.plot(self.data[:,idx[0]],self.data[:,idx[i]])
            self.fig1.show()             

    def getxas(self,ctrnames=(('keithley.6517b.1','averagechannel0','cpt.1')),Grid = False):
        """
        
        Get xas data.
        ----------
        ctrnames : List, optional
            List of possible counters (data types) in the file.
        Grid : Bool, optional.
            Indicates wether the grid values are used to normalize the XAS spectrum (True condition).
            Default values is False
        
        Returns
        -------
        Energy array
        XAS array
        Grid array if Grid == True
            
        Created: 28.04.2023 Victor
        Modified:
        """
        if not hasattr(self,'data'):
            print("No counters found!")
            pass
          
        else:
            teyid = self.__findcounterid__(ctrnames[0])
            if Grid == True:
                Gridid = self.__findcounterid__(ctrnames[1])
                return (self.data[:,0],self.data[:,teyid],self.data[:,Gridid])
            else:
                return(self.data[:,0],self.data[:,teyid])
                
    def plotxas(self,ctrnames=(('keithley.6517b.1','averagechannel0','cpt.1')),Grid = False,plothandle = True, labels = (('Energy (eV)','TEY (arb.u.)'))):
        """
        
        Plots TEY for the current scan. By default plots energy vs keithley.
        ----------
        ctrnames : List, optional
            List of possible counters (data types) in the file.
        Grid : Bool, optional.
            Indicates wether the grid values are used to normalize the XAS spectrum (True condition).
            Default values is False
        labels : List of str, optional.
            Defines the labels to be used on the x and y axes.
            Default values are 'Energy (eV)' and 'TEY (arb.u.)'
        Returns
        -------
        Plot.
        Energy array
        XAS array
        Grid array if Grid == True


        Alternative counter names :
            i14-m-cx1/ex/keithley.6517b.1/value
            i14-m-c00/ca/sai.1/averagechannel0
            
        Created: 28.04.2023 Victor
        Modified:
        """
        if not hasattr(self,'data'):
            print("No counters found!")
            pass
          
        else:
            teyid = self.__findcounterid__(ctrnames[0])
            if plothandle == False:
                if Grid == True: # will normalize the data to the Grid values
                    Gridid = self.__findcounterid__(ctrnames[1])
                    plt.plot(self.data[:,0],self.data[:,teyid]/self.data[:,Gridid],'b-',lw=2,label=labels[1])
                else:
                    plt.xasax.plot(self.data[:,0],self.data[:,teyid],'b-',lw=2,label=labels[1])
            else:
                self.xasfig,self.xasax =subplots()
                if Grid == True: # will normalize the data to the Grid values
                    Gridid = self.__findcounterid__(ctrnames[1])
                    self.xasax.plot(self.data[:,0],self.data[:,teyid]/self.data[:,Gridid],'b-',lw=2,label=labels[1])
                else:
                    self.xasax.plot(self.data[:,0],self.data[:,teyid],'b-',lw=2,label=labels[1])

                self.xasax.set_xlabel(labels[0])
                han,lab = self.xasax.get_legend_handles_labels()
                self.xasax.legend(han,lab,loc=2)
                self.xasfig.tight_layout()
                self.xasfig.show()
                
            if Grid == True:
                return (self.data[:,0],self.data[:,teyid],self.data[:,Gridid])
            else:
                return(self.data[:,0],self.data[:,teyid])
            
    def cleanimages(self, bgroix=((100,1900)), bgroiy=((200,400)), Noise = 3, SubBkg=True, FlatBkg=True, sigmaclip = 10, showmask = False):
        """
        
        New version by Victor using cosmic ray detection from ccdproc ;
        Parameters
        ----------
        bgroix : List of 2, optional
            Gives the limits of the box used to determine the background, in the dispersive direction. The default is ((100,1900)).
        bgroiy : List of 2, optional
            Gives the limits of the box used to determine the background in the non-dispersive direction. The default is ((200,400)).
        SubBkg : Bool, optional
            Removes the background to have it centered around 0 // will not be flat if FlatBkg is not used !. The default is True.
        FlatBkg : Bool, optional
            Apply a high-pass filter with a threshold set by the Noise parameter. The default is True.
        sigmaclip : Float, optional
            defines the number of sigma above which the intensity of a pixel is defined as a cosmic ray within the ccdproc module.. The default is 5.
		Noise: Float, optional
            High-pass threshold used to flatten the background (i.e. if FlatBkg is true).
            If SubBkg is true then all values below the estimated average bkg + Noise will be set to zero.
            f SubBkg is false then all values below the estimated average bkg + Noise will be set to the average background.
            Default 3. Default 3
        showmask: Bool, optional
            If true, will plot the mask used to correct the data from the cosmics. Default is False.
        
        Returns
        -------
        None.

        Created: 08.02.2023 Victor
        Modified:
        """
        
        if not hasattr(self,'rawimg'):
            print("No raw image found!")
            pass
        
        print("Cleaning image.")
        self.cleanimg = zeros(self.rawimg.shape) # create an empty image(s) to store the cleaned one(s) ;
        self.mask = zeros(self.rawimg.shape)
        
        for i in range(0,self.rawimg.shape[0]): # loop over the images ;
            tmpimg = self.rawimg[i,:,:].copy() # create a temporary copy to avoid tempering with raw data ;
            tmpimg = tmpimg.astype('float32')

            # print('Estimated background: ',bkg)
            tmpimg_cleaned, mask = ccdproc.cosmicray_lacosmic(tmpimg, sigclip=sigmaclip,gain_apply=False) # Faster than the next line !

            bkg = np.median(tmpimg_cleaned[bgroix[0]:bgroix[1],bgroiy[0]:bgroiy[1]]) # Using median instead of mean to limit impact of cosmic rays ;
            #var = np.var(tmpimg_cleaned[bgroix[0]:bgroix[1],bgroiy[0]:bgroiy[1]])
            print('estimated bkg = ',bkg)
            #print('estimated noise = ',var)
            
            if SubBkg == True:
                tmpimg_cleaned -= (bkg)*np.ones(tmpimg_cleaned.shape) # subtract the background everywhere ;
             
            if FlatBkg==True:
                if Noise == None:
                    Noise = 3
                if SubBkg == True:  
                    mask_bkg = tmpimg_cleaned[:,:] < Noise
                    tmpimg_cleaned[mask_bkg]= 0 # apply the mask --> send all points associated with the background to a single background value ;
                else:
                    mask_bkg = tmpimg_cleaned < bkg+Noise
                    tmpimg_cleaned[mask_bkg]= bkg
   
            # tmpimg_cleaned, mask = ccdproc.cosmicray_median(tmpimg,thresh=sigmaclip, mbox=11,rbox=11, gbox=10)
            self.cleanimg[i,:,:] = tmpimg_cleaned
            self.mask[i,:,:] = mask
            m = np.argwhere(self.mask[i,:,:])

            if showmask == True:
                imgfig,imgax = subplots()
                imgax.imshow(mask,cmap="gray",clim=[0,1]) # "binary"
                plt.title(("Mask for Image#"+str(i+1)))
                for sp in m:
                    plt.scatter(x=sp[1],y=sp[0],s=25,edgecolors='r',facecolors='none')
                    
                
    def showimg(self,nimg=0,ColorLim = None, ColorScale= "gist_gray", ImgKind='raw'):
        ''''Plotting the requested image(s) ;
        Parameters
        ----------
        nimg : float, optional
            Selecting the image to plot in case of a file containing several images.
            default is 0 (first image).
        ColorLim : array, optional
            Setting the limits of the colorscale to enhance visualisation ;
            default is none
        ColorScale: String, optional
            Set the colorscale to be used ; default is "gist_gray"
        ImgKind : String, optional
            Determines which image to plot i.e. raw 'raw', 
            cleaned from background/cosmics 'cleaned' 
            or cleaned and corrected for curvature 'corrected'
        ----------
        
        Created: ? Karis
        Modified: 21.02.2023 Victor'''
        
        if not hasattr(self,'rawimg'):
            raise Exception("No image found!")

        else:
            self.imgfig,self.imgax = subplots()
            if ColorLim == None:
                # Sets  reasonable colorscale limits if not given in input ;
                ColorLim = [np.min(self.rawimg[:])*2, np.max(self.rawimg[:]/2)]
                
            if not np.size(ColorLim) == 2:
                raise Exception("Wrong 'Colorlim' dimension. Should be an array of two numbers.")
            
            if ImgKind=='cleaned':
                if not hasattr(self,'cleanimg'):
                    raise Exception("No image found or image not cleaned yet. Aborting!")
                self._imtoshow = self.imgax.imshow(self.cleanimg[nimg,:,:],cmap=ColorScale,clim=ColorLim)
                
            elif ImgKind=='corrected':
                if not hasattr(self,'img'):
                    raise Exception("No image found or image not corrected yet. Aborting!")
                self._imtoshow = self.imgax.imshow(self.img[nimg,:,:],cmap=ColorScale,clim=ColorLim)
                
            elif ImgKind == 'raw':
                self._imtoshow = self.imgax.imshow(self.rawimg[nimg,:,:],cmap=ColorScale,clim=ColorLim) 
            else:
                raise Exception("Image kind not recognized, try raw, cleaned or corrected. Aborting!")
            self.imgax.set_title((self.fname.split('/')[-1].split('.')[0]+" ; "+ImgKind+" ; Image#"+str(nimg+1)))
            self.imgfig.colorbar(self._imtoshow)
            #self.imgfig.show() #commented out for new python version (showing annoying error message) ;
            

    def extractrixs(self,roi_bg_x=((100,1900)), roi_bg_y=((200,400)),SubBkg=True, Noise = None, FlatBkg=True, sigmaclip = 5, curvaturefile=None,roi_int_x=None, roi_int_y=None, Normalization = False,SumScan=True,Scans2sum = None):
        """
        Extract RIXS signal (raw, cleaned, corrected) from an image.

        Parameters
        ----------
        roi_bg_x : list of size 2, optional
            Boundaries of the region for background instimation in the dispersive direction. The default is ((100,1900)).
        roi_bg_y : list of size 2, optional
            Boundaries of the region for background instimation in the non-dispersive direction. The default is ((200,400)).
        SubBkg : Bool, optional
            If True, the estimated background will be subtracted. The default is False.
        FlatBkg : Bool, optional
            If True, the background will be flatened. The default is False.
        sigmaclip : Float, optional
            Number of sigma above which the intensity of a pixel is defined as a cosmic ray within the ccdproc module.. The default is 5.
        curvaturefile : String, optional
            Path to the file containing the curvature. The default is None.
        roi_int_x : List of 2, optional
            Boundaries of the RIXS integration along the dispersive direction. The default is None.
        roi_int_y : List of 2, optional
            Boundaries of the RIXS integration along the non-dispersive direction. The default is None.
        Normalization : Bool, optional
            If True, the function will 'normalize' the final spectrum by dividing
            it by its integral ; The default is False.

        Returns
        -------
        None.

        Created 08.02.2023 Victor
        """
        print("Extracting RIXS signal from file ",self.fname)
        self.rixs = zeros((self.rawimg.shape[0],self.rawimg.shape[1]))
        self.cleanimages(bgroix=roi_bg_x, bgroiy=roi_bg_y, Noise = Noise, SubBkg=SubBkg, FlatBkg=FlatBkg, sigmaclip = sigmaclip)
        
        if curvaturefile == None:
            self.applycurvature()
        else:
            self.setcurvature(curvaturefile)
            self.applycurvature()
            
        self.intx(roi_int_x,roi_int_y,Normalization,SumScan=SumScan,Scans2sum = Scans2sum)


    def intx(self,roi_x=None, roi_y=None, Normalization = False,Talkative=False, SumScan=True,Scans2sum = None):
        '''
        Integrates over the nondispersive direction of the detector images
        Parameters
        ----------
        roi_x : array, optional
            Setting the limits of integration along the dispersive direction of the image ;
            default is none
        roi_y: array, optional
            Setting the limits of integration along the non-dispersive direction of the image ;
        Normalization : Bool, optional
            If True, the function will 'normalize' the spectrum by dividing
            the sum by the integral of the resulting rixs spectrum ; Default is False
        SumScan: Bool, optional
            If True, will sum the rixs spectra from several images contained in the file. default is True
        Scans2sum: List of int. optional
            If SumScan is used, this is the number of the image indices to sum together. 
        ----------
        
        Created: ? Karis
        Modified: 21.02.2023 Victor'''
        
        if roi_x == None:
            roi_x = np.array([0, self.img.shape[1]])
        if roi_y == None:
            roi_y = np.array([0, self.img.shape[2]])
            
        if not np.size(roi_x)==2 or not np.size(roi_y)==2:
            print('Wrong roi format. Should be two arrays of dimension 2. Using full image instead!')
            roi_x = np.array([0, self.img.shape[1]]) ; roi_y = np.array([0, self.img.shape[2]])
        elif roi_x[1]>self.img.shape[1] or roi_y[1]>self.img.shape[2]:
            print('Given roi out of the image boundaries. Using full image instead!')
            roi_x = [0, self.img.shape[1]] ; roi_y = [0, self.img.shape[2]]
        
        if hasattr(self,"rawimg"): 
            self.rawrixs = self.rawimg[:,roi_x[0]:roi_x[1],roi_y[0]:roi_y[1]].sum(axis=2).T 
            # print(self.rawrixs.shape[1])
            self.rawrixs = np.dot(np.reshape(self.rawrixs,(self.rawrixs.shape[0],self.rawrixs.shape[1])), (1/ self.integrationtimes[0]).T) # normalize by the acquisition time for comparisons !
            
            if Talkative == True:
                print("Raw detector images integrated over the non-dispersive direction. See self.rawrixs") 
        if hasattr(self,'cleanimg'):
            self.cleanrixs = self.cleanimg[:,roi_x[0]:roi_x[1],roi_y[0]:roi_y[1]].sum(axis=2).T 
            self.cleanrixs = np.dot(np.reshape(self.cleanrixs,(self.cleanrixs.shape[0],self.rawrixs.shape[1])), (1/ self.integrationtimes[0]).T) # normalize by the acquisition time for comparisons !

            if Talkative == True:
                print ("Cleaned detector images integrated over the non-dispersive direction. See self.cleanrixs")
        if hasattr(self,'img'): 
            self.rixs = self.img[:,roi_x[0]:roi_x[1],roi_y[0]:roi_y[1]].sum(axis=2).T
            self.rixs = np.dot(np.reshape(self.rixs,(self.rixs.shape[0],self.rawrixs.shape[1])), (1/ self.integrationtimes[0]).T) # normalize by the acquisition time for comparisons !
            
            if Talkative == True:
                print("Cleaned and aligned detector images integrated over the non-dispersive direction. See self.rixs")

        # Now aligning spectra if several images are in the same file ;
        if SumScan==True:
            if Scans2sum == None:
                Scans2sum = np.arange(self.rixs.shape[1]) # prepares the indices of columns to align if not given ;
                # print(Scans2sum)
            for specNum in Scans2sum: # determine shift and align based on the first spectrum ;
                # idx_shift = np.argmax(signal.correlate(-np.min(self.rixs[:,Scans2sum[0]]), -np.min(self.rixs[:,specNum]))) - len(self.rixs[:,specNum])+1
                idx_shift = findShift(self.rixs[:,Scans2sum[0]],self.rixs[:,specNum])
                
                tmp_rawrixs = shift(self.rawrixs[:,specNum], idx_shift, cval=np.min(self.rawrixs[:,specNum]))
                self.rawrixs[:,specNum] = tmp_rawrixs
                
                tmp_cleanrixs = shift(self.cleanrixs[:,specNum], idx_shift, cval=np.min(self.cleanrixs[:,specNum]))
                self.cleanrixs[:,specNum] = tmp_cleanrixs
                
                tmp_rixs = shift(self.rixs[:,specNum], idx_shift, cval=np.min(self.rixs[:,specNum]))
                self.rixs[:,specNum] = tmp_rixs
        
            # Finally, averaging over the spectra of the file ;
            self.rawrixs = np.mean(self.rawrixs[:,Scans2sum],axis=1)
            self.cleanrixs = np.mean(self.cleanrixs[:,Scans2sum],axis=1)
            self.rixs = np.mean(self.rixs[:,Scans2sum],axis=1)
            
            # Normalization if requested ;
            if Normalization == True:
                self.rixs /= np.trapz(self.rixs) # normalization by the integral 
                
        self.intx_ax = np.linspace(roi_x[0],roi_x[1],roi_x[1]-roi_x[0]) # pixel array for plotting ;

	       	
 	
        
    def fitcurvature(self,roix=[100,1900],roiy=[100,1900],SaveCurvature=False,CurvatureName=None,returnCurvature=False):
        ''' Fit the curvature of the spectra from the ccd image ;
        Parameters
        ----------
        roix : list, optional
            Boundaries of the region where the fitting is performed in the x direction. The default is [100,1900].
        roiy : list, optional
            Boundaries of the region where the fitting is performed in the y direction. The default is [100,1900].
        SaveCurvature : bool, optional
            If True, the result of the fitting routine will be saved to a file.
            The default is 'False'.
        CurvatureName : str, optional
            If given and if SaveCurvature is True, the curvature will be saved as a file with the name provided (Uses the raw data path).
            If let as none, it will save it with a default name at the location of the raw file if provided. The default value is None.
        returnCurvature : bool, optional
            If true, the method will return the coefficient of the 2nd degree polynomial fit as an array.
            Default is false.
        -------
        
        Created ??
        Modified 11.02.2023 Victor
        '''
        if not hasattr(self,'cleanimg'):
            print("No cleaned image found! Applying image cleaning with default parameters!")
            self.cleanimages()
        if self.cleanimg.shape[0] > 1:
            print("Stack of images found! Aborting curvature fitting.")
        
        print("Fitting curvature from file ",self.fname)
        tmpx = arange(roiy[0],roiy[1]) # Initialize array to store temporary x (corresponding to energies, dispersion not identified yet)
        position = zeros((roix[1]-roix[0])) # Initialize array to store the fitted Gaussian positions ;
        area = zeros((roix[1]-roix[0])) # Initialize array to store fitted Gaussian area/amplitude ;
        width = zeros((roix[1]-roix[0])) # Initialize array to store fitted Gaussian width ;
        
        print("Fitting detector columns")
        for ind,i in enumerate(range(roix[0],roix[1])): 
            tmpy = self.cleanimg[0,roiy[0]:roiy[1],i] 
            tmppars = fitgaussian(tmpx,tmpy,plot=False,printrep=False)       	
            position[ind] = tmppars[1] 
            area[ind] = tmppars[0] 
            width[ind] = tmppars[2] 
            cpolycoeff = polyfit(arange(roix[0],roix[1]),position,2)
        print("Curvature correction polynomial coefficients \nas [a,b,c] in a * x**2 + b * x + c:")
        print(cpolycoeff)
        figx = arange(roix[0],roix[1]) 
        #polyy = polyval(cpolycoeff,arange(0,2048)) 
        fig1,ax1 = subplots(1,3) 
        ax1[0].plot(figx,position,lw=2) 
        ax1[0].plot(figx,polyval(cpolycoeff,figx)) 
        ax1[0].set_xlabel('CCD column',fontsize=18) 
        ax1[0].set_ylabel('Peak position',fontsize=18) 
        
        ax1[1].plot(figx,area,lw=2) 
        ax1[1].set_xlabel('CCD column',fontsize=18) 
        ax1[1].set_ylabel('Peak area',fontsize=18) 
        ax1[2].plot(figx,width,lw=2) 
        ax1[2].set_xlabel('CCD column',fontsize=18) 
        ax1[2].set_ylabel('Peak width',fontsize=18) 
        
        #fig1.show() 
        fig1.tight_layout()
        
        if SaveCurvature==True:
            if CurvatureName==None:
                name = "Curvature_"+self.fname[-13:-4]
                savetxt(self.fname[0:-13]+name,cpolycoeff) 
                print("Curvature saved as: ",name) 
            else:
                savetxt(CurvatureName,cpolycoeff) 
                print("Curvature saved as: ",CurvatureName)
        if returnCurvature==True:
            return cpolycoeff


    def setcurvature(self,fname=None,refcolumn=1023,Curv=None):
        ''' Set the curvature as an attribute of the object ;
        Parameters
        ----------
        fname : String
            fname: name of the file containing the curvature of the data as the coefs of a polynomial of second order.
        refcolumn : int, optional
            refcolumn --> column of reference for the correction. The default is 1023.
        -------
        
        Created ??
        Modified 08.02.2023 Victor
        '''
        try: 
            if not Curv== None:
                self.cpolycoeff(Curv)
            elif not fname==None:
                self.cpolycoeff = genfromtxt(fname) # store the curvature data in the attributes
                self.cpolyfile = fname # keep the name of the file ;
            else:
                print("No curvature given, aborting!")
                pass
            self._cx = range(0,2048) 
            self._cy = polyval(self.cpolycoeff,self._cx)
            self._cshifty = self._cy-self._cy[refcolumn] 
            self._hascurvature = True 
        except(IOError): 
                print("File ",fname," not found!")
                
                
    def applycurvature(self,refcolumn=1023):
        ''' Apply the curvature to the data
        Parameters
        ----------
        refcolumn : int, optional
            refcolumn --> column of reference for the correction. The default is 1023.
        ----------
        
        Created ??
        Modified 08.02.2023 Victor
        '''
        
        if self._hascurvature == True:
            print("Using curvature correction from the file ",self.cpolyfile)
        if self._hascurvature == False: # No set curvature so we imporovise by using a default curvature ;
            print("No curvature set! Will use defaultcurvature.txt (if found)") 
            try: 
                tmp = genfromtxt('defaultcurvature.txt') 
                self.cpolycoeff = tmp 
                self.cpolyfile = "defaultcurvature.txt" 
                self._cx = range(0,2048) 
                self._cy = polyval(self.cpolycoeff,range(0,2048)) 
                self._cshifty = self._cy-self._cy[refcolumn]
                print("defaultcurvature.txt found")
            except(IOError):
                print("defaultcurvature.txt not found!") 
                pass

        if not hasattr(self,'cleanimg'):
            print("This scan has no cleaned image! Aborting")
            pass
            
        self.img = zeros(self.cleanimg.shape) # Initialize the array to store the image ;
        for j in range(0,self.cleanimg.shape[0]): # loop over the x (non dispersive) direction ;
            for i in range(0,self.cleanimg.shape[1]): # loop over the y (dispersive) direction ;
                self.img[j,:,i] = kinterp(self._cx,self.cleanimg[j,:,i],self._cshifty[i]) # apply the curvature correction;
            # print("Image corrected and saved to self.img")		
	
    def savecurvature(self,fname): 
        """Saves the curvature polynomial to file fname""" 
        if not hasattr(self,'cpolycoeff'): 
            print("ERROR! No curvature polynomial found") 
            pass 
        else: 
            savetxt(fname,self.cpolycoeff)
    
    def setDispersion(self,fname=None,Dispersion=None):
        ''' Set the dispertion as an attribute of the object ;
        Parameters
        ----------
        fname : String
            fname: name of the file containing the dispersion of the data as the coefs of a polynomial of second order.
        Dispersion : list of float, optional
            Coefficients obtained from fitting the dispersion with a polynomial ;
        -------
        
        Created 12.02.2023 Victor
        '''
        try: 
            if not fname==None:
                self.Dispersion = genfromtxt(fname) # store the curvature data in the attributes
                self.Dispersionfile = fname # keep the name of the file ;
                self._hasdispersion = True 
            elif not Dispersion.any == None:
                self.Dispersion=Dispersion
                self._hasdispersion = True 
            else:
                print("No dispersion given, aborting!")
                pass
        except(IOError): 
                print("File ",fname," not found!")
        
        
    def applyDispersion(self,prominence=200,width=5,E_shift=True):
        """
        Create the list of energies corresponding to the RIXS signal 
        based on the determined dispersion on the detector.
        
        Parameters
        ----------
        E_shift : Bool, optional
            If True, the fucntion will try to correct for possible energy shift.
        prominence : float, array of floats, optional
            minimum or range of prominence to attribute a peak. The default is 20000.
        width : float or array of two floats, optional
            Minimum or range of peak width to be accepted. The default is 5.

        Created 12.02.2023 Victor
        Modified 28.02.2023 Victor
        """
        
        if self._hasdispersion == True:
            print("Using disperion from the file ",self.cpolyfile)
        if self._hasdispersion == False: # No set curvature so we imporovise by using a default curvature ;
            print("No Dispersion set! Aborting!") 
            pass
        
        self.E_out = polyval(self.Dispersion,self.intx_ax)
        tmp_E_out = self.E_out
        
        if E_shift==True:
            # E0 = tmp_E_out[np.argmax(self.rixs)]
            # E0 = self.__getEi_fit() # Best result bu fitting the RIXS and setting the gaussian center as the E_0 or E_incident ;
            E0 = self.__getEi_prominence(prominence=prominence,width=width)
            # E0 = self.Ei_mono # More save as signal can shift significantly in between frames ...
            
            self.__shiftSpectrum(ObsElastiEnergy=E0)
        else:
            E0 = self.Ei_mono
            
        print("Nominal energy: ",self.Ei_mono,"eV")
        print("Estimated energy: ",E0,"eV")
        self.E_loss = (self.Ei_mono*np.ones(np.size(tmp_E_out)) - tmp_E_out) # subtract energy of the spectrum's max instead of the nominal incident energy ;
        self.Ei = E0 # sets a new attribute, the estimated elastic energy, which might be different than the one given by the mono ;

    def __shiftSpectrum(self,ObsElastiEnergy):
        """
        The function is used to manually shift the spectrum to align the 
        observed elastic line to its nominal value, therefor assuring an 
        elastic line at zero energy loss.

        Parameters
        ----------
        ObsElastiEnergy : float
            Observed energy of the elastic line.

        Returns
        -------
        None.

        Created 23.02.2023 Victor
        """
        tmp_shift_rixs = shiftSpectrum(ObsElastiEnergy, self.Ei_mono, self.E_out, self.rixs)
        self.rixs = tmp_shift_rixs # updating the rixs attribute 
        
        
    def __getEi_fit(self):
        """
        Fits the maximum of the RIXS signal to estimate the center of the Gaussian peak.
        The value is then used as the energy of the elastic line.

        Returns
        -------
        Float
            E_0 or E-incident i.e. energy at zero energy transfer.

        Created 13.02.2023 Victor
        """
        E0 = self.E_out[np.argmax(self.rixs)]
        # print(E0)
        parameters, covariance = curve_fit(gausswbg, self.E_out, self.rixs, p0=[np.max(self.rixs),E0,1.0,np.min(self.rixs)])#, bounds=((-np.inf, E0-0.2, 0, 0),(np.inf, E0+0.2, 3, np.inf)))
        # print(parameters[1])
        return parameters[1]
    
    def __getEi_prominence(self,prominence=20000,width=5):
        """
        Detect the elastic line. For details, check out the signal.find_peaks function ;

        Parameters
        ----------
        prominence : float, array of floats, optional
            minimum or range of prominence to attribute a peak. The default is 20000.
        width : float or array of two floats, optional
            Minimum or range of peak width to be accepted. The default is 5.

        Returns
        -------
        Float
            Estimated eneergy of the elastic line.
        
        Created 23.02.2023 Victor
        """
        return getEi_prominence(self.E_out, (self.rixs-np.nanmean(self.rixs)), prominence=prominence,width=width)

        
    def genfigure(self,i,thll,thlh):
        fig = Figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        immean = self.rawimg[i,:,:].mean()
        imtoshow = ax.imshow(self.rawimg[i,:,:],vmin=thll*immean,vmax=thlh*immean)
        fig.colorbar(imtoshow)
        return fig,ax,imtoshow


    def saveRIXS(self,filename,EKind=None,rixsKind='rixs'):
        """
        Save rixs spectrum into a text file.

        Parameters
        ----------
        filename : str
            Name and full path where the data should be written.
        EKind : str, optional
            Gives the energy reference to be used as x. Can be energy loss of outgoing energy. The default is None and leads to the writting of the pixel numbers.
        rixsKind : str, optional
            Gives the rixs signal to be written in the file. Can be the 'rawrixs'', 'cleanrixs' and fully corrected 'rixs'. The default is 'rixs'.

        Returns
        -------
        None.

        """
        if rixsKind=='rawrixs':
            spectrum2save = self.rawrixs
        elif rixsKind=='cleanrixs':
            spectrum2save = self.cleanrixs
        elif rixsKind=='rixs':
            spectrum2save = self.rixs
        else:
            print("The rixs object does not have the type of rixs spectra asked. Aborting!") 
            pass
        if EKind == 'E_loss' and hasattr(self,'E_loss'):
            f = open(filename,"w+")
            f.write("E_loss (eV)   rixs_signal (arb.u.)\n")
            for idx,E in enumerate(self.E_loss):
                f.write("%.2f  %.6f\n" % (E, spectrum2save[idx]))
            f.close()
        elif EKind == 'E_out' and hasattr(self,'E_out'):
            f = open(filename,"w+")
            f.write("E_out (eV)   rixs_signal (arb.u.)\n") 
            for idx,E in enumerate(self.E_out):
                f.write("%.2f  %.6f\n" % (E, spectrum2save[idx]))
            f.close()
        else:
            print("No energy type given. The pixel associated with the spectrum will be written instead.")
            f = open(filename,"w+")
            f.write("pixel number   rixs_signal (arb.u.)\n")
            for idx,E in enumerate(self.intx_ax):
                f.write("%i  %.6f\n" % (E, spectrum2save[idx]))
            f.close()
            
def fixglitches(data):
	pass

def align(ymat,aliroi=None): 
    ymat = nan_to_num(ymat) 
    # "print NaNs set to zero" 
    if aliroi is None: 
        aliroi = (0,ymat.shape[0]) 
    x = arange(aliroi[0],aliroi[1]) 
    alix = arange(0,ymat.shape[0]) 
    newymat = zeros(ymat.shape) 
    newymat[:,0]=ymat[:,0] 
    
    refyn = ymat[aliroi[0]:aliroi[1],0]/trapz(ymat[aliroi[0]:aliroi[1],0]) 
    print("Spectrum:	Shift:") 
    for i in range(1,ymat.shape[1]): 
        tmpyn = ymat[aliroi[0]:aliroi[1],i]/trapz(ymat[aliroi[0]:aliroi[1],i]) 
        shiftable = lambda s: sum( (refyn-kinterp(x,tmpyn,s)**2)) 
        guess = x[argmax(tmpyn)]-x[argmax(refyn)] 
        out = minimize(shiftable,guess,tol=1e-12) 
        newymat[:,i]=kinterp(alix,ymat[:,i],out.x) 
        print(i,"		",out.x)
        return newymat
	
	
	
def gausswbg(x,a,x0,sig,bg):
    return a/(sig*1.0)*sqrt(2*pi)*exp(-1/(2.0)*(x-x0)**2/(sig**2))+bg

def fitgaussian(x,y,roi=None,plot=True,printrep=True): # guess=None,
    if roi == None:
        roi = (0,len(x))
    # if guess.any() == None: 
    guess =  array([max(y[roi[0]:roi[1]]),x[argmax(y[roi[0]:roi[1]])],1,0])
    popt,pcov = curve_fit(gausswbg,x[roi[0]:roi[1]],y[roi[0]:roi[1]],p0=guess)
    if plot ==True: 
        fig1,ax1 = subplots() 
        ax1.plot(x,y,label='Data') 
        ax1.plot(x,gausswbg(x,popt[0],popt[1],popt[2],popt[3]),label="Fit") 
        han,lab=ax1.get_legend_handles_labels()	
        ax1.legend(han,lab) 
        fig1.show()
    if printrep==True:
        print("Area: ",popt[0])
        print("Position: ",popt[1]) 
        print("Sigma: ",popt[2])
        print("FWHM: ",popt[2]*2.355) 
        print("Baseline:",popt[3])
    return popt


def momtrans(ein,tth):
    """For ein (keV) gives the absolute value of the momentum transfer
    at scattering angle tth (deg) in units of inverse angstrom"""
    kevtoang = 12.389
    q = [4*pi/(kevtoang/ein)*sin(radians(ang/2)) for ang in tth]
    return q

def kinterp(x,y,delta):
    """kinterp is a wrapper for numpys interp for interpolating from one axis x to another 
     axis x\' that is shifted by the number delta with respect to x. kinterp is super 
     duper smart and is able to interpolate even if the x vector is monotonously 
     decreasing."""

    if all(diff(x)>0):
        return interp(x+delta,x,y)
    elif all(diff(x)<0):
        return flipud(interp(flipud(x+delta),flipud(x),flipud(y)))
    else:
        print("Something is wrong with the x vector") 
        pass     

           
def getDispersion(fnames,curvaturefile,roi_bg_x=((100,1900)), roi_bg_y=((0,200)),SubBkg=True, FlatBkg=True,Noise=3, sigclip = 5, saveDispersion=False,DispersionName=None):
    """
    Function to quickly apply cleaning and curvature to data file, 
    followed by RIXS ectraction of several files, which are then used to 
    determine the dispersion of the experiment ;

    Parameters
    ----------
    fnames : List of string
        List of file names to be cleaned, reduced and used for dispersion.
    Eis : array of float
        List of incident energies associated with the files ;
    curvaturefile : string
        File to be used as curvature parameters.
    roi_bg_x : list of int, optional
        Set the x range (dispersive direction) for background. The default is ((100,1900)).
    roi_bg_y : list of int, optional
        Set the x range (non-dispersive direction) for background. The default is ((0,200)).
    FlatBkg : Bool, optional
        If True, makes the background flat. The default is True.
    lower : TYPE, optional
        lower boudary of the signal set to the background value if FlatBkg is used (True --> produces a flat background). The default is 1.
    upper : TYPE, optional
        upper boudary of the signal set to the background value if FlatBkg is used (True --> produces a flat background). The default is 2.
    sigclip : float, optional
        sigmaclip defines the number of sigma above which the intensity of a pixel is defined as a cosmic ray within the ccdproc module.. The default is 5.

    Returns 
    -------
        The coeff used to fit the dispersion with a polynomial of second degree.
        The Figure handle
        The plot handle
        The list of created RIXS objects.
        
    Created 11.02.2023  Victor
    """
    Scans_list=[]
    Pos_list=[]
    Eis = []
    for n in fnames:
        Scans_list.append(rixs_image(n))
        Scans_list[-1].extractrixs(roi_bg_x=roi_bg_x, roi_bg_y=roi_bg_y, FlatBkg = FlatBkg, SubBkg=SubBkg , Noise = Noise , sigmaclip = sigclip, curvaturefile=curvaturefile)
        TmpScan = Scans_list[-1].rixs # save the rixs signal in a simple array ;
        parameters, covariance = curve_fit(gausswbg, np.linspace(1,2048,2048), TmpScan,p0=[np.max(TmpScan),np.argmax(TmpScan),1.0,np.min(TmpScan)])
        Pos_list.append(parameters[1])
        Eis.append(Scans_list[-1].Ei_mono)
    cpolycoeff = polyfit(Pos_list,Eis,2)
        
    Fig, ax = plt.subplots(1,1,figsize=[10,4])
    ax.scatter(Pos_list,Eis)
    ax.plot(Pos_list,polyval(cpolycoeff,Pos_list),color='orange',label='fit')
    
    if saveDispersion==True:
        if DispersionName==None:
            name = "Disperion_"+Scans_list[-1].fname[-13:-4]
            savetxt(Scans_list.fname[0:-13]+name,cpolycoeff) 
            print("Curvature saved as: ",name) 
        else:
            savetxt(DispersionName,cpolycoeff) 
            print("Dispersion saved as: ",DispersionName)    
    return(cpolycoeff,Fig,ax,Scans_list)


def getCurvature(fname,bgroix=((100,1900)),bgroiy=((200,400)),SubBkg=True,FlatBkg=True,Noise= 3 ,sigmaclip = 5,roix=[300,1700],roiy=[200,1800],SaveCurvature=False,CurvatureName=None,refcolumn=1023):
    """
    Function to try and quickly determine the curvature of the signal on the detector ;

    Parameters
    ----------
    fname : String
        Name of the file with which we determine the curvature.
    roi_bg_x : list of int, optional
        Set the x range (dispersive direction) for background. The default is ((100,1900)).
    roi_bg_y : list of int, optional
        Set the x range (non-dispersive direction) for background. The default is ((0,200)).
    FlatBkg : Bool, optional
        If True, makes the background flat. The default is True.
    lower : TYPE, optional
        lower boudary of the signal set to the background value if FlatBkg is used (True --> produces a flat background). The default is 1.
    upper : TYPE, optional
        upper boudary of the signal set to the background value if FlatBkg is used (True --> produces a flat background). The default is 2.
    sigclip : float, optional
        sigmaclip defines the number of sigma above which the intensity of a pixel is defined as a cosmic ray within the ccdproc module.. The default is 5.
    SubBkg : Bool, optional
        SubBkg removes the background to have it centered around 0 // will not be flat if FlatBkg is not used !. The default is False.
    roix : List of int, optional
        Boundaries of the region where the fitting is performed in the x direction. The default is [100,1900].
    roiy : List of int, optional
        Boundaries of the region where the fitting is performed in the y direction. The default is [100,1900].
        
    SaveCurvature : Bool, optional
        If true, will save the curvature into a file. The default is False.
    CurvatureName : string, optional
        If SaveCurvature=True, this will be the name of the file. The default is None.
    
    refcolumn : TYPE, optional Ask Kari
        DESCRIPTION. The default is 1023.

    Returns 
    -------
        The Curvature as coeff of the fitted polynomial of degree two.
        The created RIXS object.
        
    Created 11.02.2023  Victor
    """
    Scan = rixs_image(fname)
    Scan.cleanimages(bgroix=bgroix,bgroiy=bgroiy,SubBkg=SubBkg,FlatBkg=FlatBkg,Noise=Noise,sigmaclip = sigmaclip)
    Curv = Scan.fitcurvature(roix=roix,roiy=roiy,SaveCurvature=SaveCurvature,CurvatureName=CurvatureName,returnCurvature=True)
    return Curv, Scan

    
def plotRIXSMap(RIXSobjectList, colormap = 'plasma',binsize=0.1,col_scale='normalized',LowLimFactor=50,HighLimFactor=50):
    """
    Function to create a RIXS map from a list of RIXS objects.

    Parameters
    ----------
    RIXSobjectList : List of RIXS objects
        List of RIXS object from which the spectra will be plotted as a map.
    colormap : string, optional
        Name of the colormap to use for plot. The default is 'plasma'.
    binsize : float, optional
        Step size in energy to be used for interpolation and plotting. The default is 0.1.
    col_scale : string, optional
        Type of color scale normalization (at the moment acceptes 'normalized' & 'logNorm'). The default is 'normalized'.

    Returns
    -------
        Handle of the figure and plot.
    
    Created 14/02/2023 Victor
    """
    if binsize < 0.1:
        print("Warning: the energy step size is smaller than that of the actual data. Artefacts could appear.")
    # initialize arrays storing incident energies, RIXS signal and energy losses ;    
    Eis_list = np.empty((np.size(RIXSobjectList),))
    RIXS_list = np.empty((np.size(RIXSobjectList),np.size(RIXSobjectList[0].rixs)))
    Eloss_list = np.empty((np.size(RIXSobjectList),np.size(RIXSobjectList[0].E_loss)))
    for idx, Scan in enumerate(RIXSobjectList):
        Eis_list[idx] = Scan.Ei_mono
        RIXS_list[idx] = Scan.rixs
        Eloss_list[idx] = Scan.E_loss


    M = int(np.max(Eloss_list)) # max energy loss
    m = -5 # Min energy low (gain here) to be plotted
    n_bins = int((M-m)/binsize)+1 # Number of bins to interpolate the data ;
    bins = np.linspace(m,M,n_bins) # new energy loss bins ;

    Map = np.zeros((np.size(Eis_list),n_bins)) # initialize RIXS map
    Map[:] = np.nan

    for idx, S in enumerate(RIXS_list): # loop on spectra 
        tmp_E_loss = Eloss_list[idx]
        tmp_max = int(np.max(tmp_E_loss)) # get max energy loss
        tmp_n_bins = int((tmp_max-m)/binsize)+1 # associated number of bins
        tmp_bins = np.linspace(m,tmp_max,tmp_n_bins) # and associated new bins
        Map[idx,0:tmp_n_bins] = np.interp(tmp_bins,tmp_E_loss,S) # intepolate the RIXS for new bins and store it in the map;

    Map -= np.nanmin(Map) # subtract 'background' of the Map 

    X, Y = np.meshgrid(Eis_list, bins) # prepare mesh for plot ;

    # Plot the surface.
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    
    if col_scale == 'normalized':
        im = axs.pcolormesh(Y,X, Map.T,antialiased=False,norm=colors.Normalize(vmin=(np.nanmin(Map)*LowLimFactor), vmax=np.nanmax(Map)/HighLimFactor),linewidth=0,cmap='plasma', shading='auto',edgecolors=None)
    elif col_scale == 'logNorm':
        im = axs.pcolormesh(Y,X, Map.T,antialiased=False,norm=colors.LogNorm(vmin=(np.nanmin(Map)+1e-0), vmax=np.nanmax(Map)),linewidth=0,cmap='plasma', shading='auto',edgecolors=None)
    else:
        print("Warning: Wrong color scale normalization, use normalized of logNorm.")
        pass
    
    axs.set_xlabel('$E_{low}$ (eV)')
    axs.set_ylabel('$E_{incident}$ (eV)')
    fig.colorbar(im,ax = axs)
    return fig,axs, im, Map, Eis_list, bins

def getEi_prominence(EnergyOut, Spectrum, prominence=20,width=3,height=None,direction='L2R'):
    """
    Detect the elastic line. For details, check out the signal.find_peaks function ;

    Parameters
    ----------
    EnergyOut: array of floats.
        List of energy values in the spectrum.
    Spectrum: array of floats.
        List of intensities correponding to the energies in EnergyOut.
    prominence : float, array of floats, optional
        Minimum or range of prominence to attribute a peak. The default is 20000.
    width : float or array of two floats, optional
        Minimum or range of peak width to be accepted. The default is 5.
    direction : str, optional.
        Direction of the spectrum L2R means with increasing index goes increasing E_out / R2L increasing index means decreasing E_out
    Returns
    -------
    Float
    Estimated eneergy of the elastic line.
        
    Created 23.02.2023 Victor
    """
    peaks_loc = signal.find_peaks(Spectrum, prominence=prominence,width=width, height=height)
    if direction=='L2R':
        return EnergyOut[peaks_loc[0][0]]
    elif direction=='R2L':
        return EnergyOut[peaks_loc[0][-1]]

def shiftSpectrum(ObsElastiEnergy, EnergyMono, EnergyOut, Spectrum):
    """
    The function is used to shift the spectrum to align the 
    observed elastic line to its nominal value, therefor assuring an 
    elastic line at zero energy loss.

    Parameters
    ----------
    ObsElastiEnergy : float
        Observed energy of the elastic line.
    EnergyMono : Float.
        Nominal incident energy given by the monochromator.
    EnergyOut : array of floats.
        List of outgoing energies of the spectrum to shift.
    Spectrum : array of floats.
        List of intensities of the spectrum.
        
    Returns
    -------
    shifted_rixs : array of floats.
        List of spectrum intensities after shifting.

    Created 23.02.2023 Victor
    """
    idx_ElasE = np.nanargmin(np.abs(EnergyOut - ObsElastiEnergy)) # find location/index of the observed elastic peak 
    idx_Emono = np.nanargmin(np.abs(EnergyOut - EnergyMono)) # same for the energy of the mono ;
    idx_shift = idx_Emono - idx_ElasE  # number of points to shift
    # print(idx_shift)
    shifted_rixs = shift(Spectrum, idx_shift, cval=np.min(Spectrum)) # shifting
    return shifted_rixs 
    
    
def findShift(Spectrum1,Spectrum2):
    """
    Function to find a small shift in between to very similar spectra.

    Parameters
    ----------
    Spectrum1 : Array of floats.
        First spectrum.
    Spectrum2 : Array of floats.
        Second spectum.

    Returns
    -------
    idx_shift : Integer
        Number of bins to shift to have maximum overlap of both spectra.

    """
    idx_shift = np.argmax(signal.correlate(Spectrum1-np.min(Spectrum1), Spectrum2-np.min(Spectrum2))) - len(Spectrum2)+1
    return idx_shift
# %% Old and unused stuffs:
    
   # def __findcounterid__(self,ctrname):
   #     if not hasattr(self,'longnames'):
   #         print('no counters found!')
   #         pass
   #     for ind,name in enumerate(self.longnames):
   #         tmp = name.split('/')
   #         if len(tmp)==4:
   #             if ctrname in tmp[2]:
   #                 return ind


   # def getdata(self,ctrlist=list(("energy","keithley","counter01")),loc="sextants",Talkative=False):
       # Old version using Tables !!
   #     """Extracts the counters and possible CCD images from a nxs file named "fname". 
   #     The counters are put to the attribute self.data and the image(s) to self.rawimg 
   #     """
   #     #if loc == "sextants":
   #     print("Handling file ",self.fname)
       
   #     try:
   #         self.longnames=list()
   #         # tmpfile = open_file(self.fname,'r')
   #         tmpfile = h5py.File(self.fname,'r')
   #         tmpdata = tmpfile.root.__getattr__(tmpfile.root.__members__[0])
   #         self.ctrnames = list()
   #         self._hasdata = False
   #         self._hasimg = False
   #         self._ctrlen = 0
   #         self._ctrn = 0
   #         self._hascurvature = False
   #         self.Ei_mono = tmpdata.SEXTANTS.mono.energy[0]
   #         print("Incident energy: ",tmpdata.SEXTANTS.mono.energy[0],"eV")
   #             # self.tstart  = tmpdata.start_time.read() # causing some errors sometimes ; to investigate !
               
   #         for it in tmpdata.scan_data.__iter__():
   #             if Talkative==True:
   #                 print("Counter ",it.attrs.long_name," found.")
   #             if it.ndim == 1:
   #                 self._hasdata = True
   #                 self._ctrn += 1
   #                 self._ctrlen = len(it.read())
   #                 self.longnames.append(it.attrs.long_name)      
   #                 self.ctrnames.append(it.name)

   #             if it.ndim > 1:
   #                 if Talkative==True:
   #                     print("Detector image found")
   #                 self._hasimg = True
   #                 self._imshape = it.shape
   #                 self.longnames.append(it.attrs.long_name)      
   #                 self.ctrnames.append(it.name)

   #             if self._ctrlen > 0:
   #                 self.data = zeros((self._ctrlen,self._ctrn))
   #                 if Talkative==True:
   #                     print("Created counter array data with shape (",self._ctrlen,",",self._ctrn,")")
           
   #             if self._hasimg ==True:
   #                 self.rawimg = zeros(self._imshape)
   #                 if Talkative==True:
   #                     print ("Created image array rawimg  with shape ",self._imshape)
           
   #         ctrind = 0
           
   #         for it in tmpdata.scan_data.__iter__():
   #             if it.ndim  <2:
   #                 self.data[:,ctrind] = it.read()
   #                 ctrind += 1
                           
   #             elif it.ndim > 1:
   #                 self.rawimg = it.read()
   #                 if Talkative==True:
   #                     print("------------")
   #                     print("Closing file")
   #                 tmpfile.close()       
   #     except Exception as e:
   #         if Talkative == True:
   #             print('Ignoring Exception: "', e,'"')
   
   

    # def alignrixs(self,aliroi=None):
    #     """Aligns the spectra derived from CCD images. Currently only operates on otherwise fully processed RIXS"""
    #     self.alirixs = align(self.rixs) 
    #     fig1,ax1 = subplots() 
    #     px = arange(0,self.rixs.shape[0]) 
    #     ax1.plot(px,self.rixs[:,0],label="Reference") 
    #     ax1.plot(px,self.alirixs[:,1:]) 
    #     han,lab = ax1.get_legend_handles_labels() 
    #     ax1.legend(han,lab) 
    #     fig1.show() 
    #     print("RIXS spectra aligned")

           
    # def plotxas(self,ctrnames=(('keithley.6517b.1','cpt.1')),labels = (('Energy (eV)','TEY (arb.u.)','TFY (arb.u)'))):
    #     """Plots TEY and TFY for the current scan.
    #     By default plots energy vs keithley/MCP. You can change this using 
    #     the keyword variable ctrnames"""
    #     if not hasattr(self,'data'):
    #         print("No counters found!")
    #         pass
          
    #     else:
    #         teyid = self.__findcounterid__(ctrnames[0])
    #         tfyid = self.__findcounterid__(ctrnames[1])
    #         self.xasfig1,self.xasax1 =subplots()
    #         self.xasax1.plot(self.data[:,0],self.data[:,teyid],'b-',lw=2,label=labels[1])
    #         self.xasax2 = self.xasax1.twinx()
    #         self.xasax2.plot(self.data[:,0],self.data[:,tfyid],'ko-',lw=2,label=labels[2])
    #         self.xasax1.set_xlabel(labels[0])
    #         han1,lab1 = self.xasax1.get_legend_handles_labels()
    #         han2,lab2 = self.xasax2.get_legend_handles_labels()
    #         self.xasax1.legend(han1+han2,lab1+lab2,loc=2)
    #         self.xasfig1.tight_layout()
    #         self.xasfig1.show()

    # def sumrixs(self,ax=1,tosum="rixs"): 
    #     """Sums the raw and processed RIXS spectra associated with this scan""" 
    #     if tosum == "rawrixs": 
    #         if hasattr(self,"rawrixs"): 
    #             self.rawrixssum = self.rawrixs.sum(axis=ax) 
    #             print ("Raw data summed. See attribute self.rawrixssum") 
    #     if tosum == "cleanrixs": 
    #         if hasattr(self,"cleanrixs"): 
    #             self.cleanrixssum = self.cleanrixs.sum(axis=ax) 
    #             print ("Cleaned data summed. See attribute: self.cleanrixssum") 
    #     if tosum == "rixs": 
    #         if hasattr(self,"rixs"): 
    #             self.rixssum = self.rixs.sum(axis=ax) 
    #             print("Cleaned and curvature corrected data summed. See attribute: self.rixssum") 
    #     if tosum == "alirixs": 
    #         if hasattr(self,"alirixs"): 
    #             self.alirixssum = self.alirixs.sum(axis=ax)
    #             print("Cleaned, curvature corrected and aligned RIXS scans summed. See attribute self.alirixssum")

import numpy as np

import warnings
warnings.simplefilter("always")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')
from matplotlib import colors
import scienceplots
plt.style.use(["science", "notebook"])

import time

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

from WDMWaveletTransforms.wavelet_transforms import (
    transform_wavelet_freq,
    transform_wavelet_freq_time,
    transform_wavelet_time,
)

from phentax.waveform import IMRPhenomTHM

from lisaconstants import ASTRONOMICAL_YEAR, C, GRAVITATIONAL_CONSTANT, SOLAR_MASS
from lisatools.detector import EqualArmlengthOrbits
from fastlisaresponse import ResponseWrapper  # also can do from fastlisaresponse.response import ResponseWrapper; ResponseWrapper is located in fastlisaresponse.response

from scipy.signal import welch

from multiprocessing import Pool, Manager

import yaml
import pickle

import math
import gc

import argparse

import traceback


def getWDMMatrix(waveform, Nt = 128//1, Nf = None, noiseLevel = 10**(-22.5), noiseLevel2 = 10**(-27), whitenPSD = None, dt = None, fmin = 1e-8, fmax = np.inf):  # orig: noiseLevel = 1e-24
    if Nf is None:
        Nf = len(data) // Nt
    ND = Nf * Nt
    
    dataFFT = np.fft.fft(waveform[:ND])
    
    waveletAmplitudeMatrix = transform_wavelet_freq(dataFFT, Nf, Nt).T

    fCoords = np.arange(0, Nf + 1) / (2 * dt * Nf)

    # Take abs value. taking the abs value to remove negative values instead of just adding everything by the min value works better for some reason. haven't thought about why this is
    waveletAmplitudeMatrix = np.abs(waveletAmplitudeMatrix)

    # Whiten with a PSD
    if whitenPSD is not None and dt is not None:
        fCoords = np.arange(0, Nf + 1) / (2 * dt * Nf)
        waveletAmplitudeMatrix /= np.sqrt(np.interp(fCoords[1:], whitenPSD[0], whitenPSD[1])[:, np.newaxis])
    elif whitenPSD is not None:
        print('When passing a whitenPSD, you need to specify dt in getWDMMatrix(). Whitening is being skipped!')

    # White noise
    #if noiseLevel is not None and noiseLevel > 0:
    #    rng = np.random.default_rng()
    #    noise = np.abs(rng.normal(loc = 0, scale = 1, size = waveletAmplitudeMatrix.shape)) * noiseLevel  # 1e-22 for h (actual signal), 1e-24 for TDI (response from LISA)
    #    noise2 = np.cumsum(np.cumsum(np.abs(rng.normal(loc = 0, scale = 0.7, size = waveletAmplitudeMatrix.shape)), axis = 0)[::-1, :], axis = 0)[::-1, :] * noiseLevel2  # sum by the columns bc we want different frequency bands to have different noise levels
    
    #    waveletAmplitudeMatrix += noise + noise2
    
    # Find min and max freq indices
    if fmax == np.inf:
        fMaxIdx = Nf
        fmax = fCoords[-1]
    else:
        fMaxIdx = np.abs(fCoords - fmax).argmin()
    
    if fmin == 0:
        fMinIdx = 0
    else:
        fMinIdx = np.abs(fCoords - fmin).argmin()

    # Only take relevant values in the matrix
    waveletAmplitudeMatrix = waveletAmplitudeMatrix[fMinIdx:fMaxIdx, :]  # :fMaxIdx is deliberately w/o the +1 bc fCoords represent the boundaries so there'll be one extra for fCoords but not for the WDM matrix
    
    
    # Some form of normalization before cleaning/LogNorm (i think this helps somehow?)
    waveletAmplitudeMatrix /= waveletAmplitudeMatrix.max()

    # Clean the matrix from bad values by replacing every bad value with the minimum value in the matrix and taking out negative vals. This may be a necessary step because plt.pcolormesh wasn't giving good results when I plotted before cleaning. Though that could just be pcolormesh messing up or something, because the whole graph was just white and white areas are an issue with pcolormesh sometimes. According to Gemini, pcolormesh might plot nan, inf, and -inf vals as white.
    minAmp = waveletAmplitudeMatrix.min()
    waveletAmplitudeMatrixClean = np.nan_to_num(waveletAmplitudeMatrix, nan=minAmp, posinf=minAmp, neginf=minAmp)  # take out NaN, ∞, and -∞ vals

    return waveletAmplitudeMatrixClean, fMinIdx, fMaxIdx



def getSpectrogram(WDMMatrix, Nt, Nf, dt, cmap = 'inferno', vmin = None, vmax = None, logColor = True, logFreq = True, fmin = 1e-8, fmax = np.inf, filename = None, imageFolder = None, showPlot = True, plotFeatures = False):
    # Coords for pcolormesh
    Nd = Nf * Nt
    TObs = Nd * dt

    tCoords = np.linspace(0, TObs, Nt + 1)
    fCoords = np.arange(0, Nf + 1) / (2 * dt * Nf)


    # Find min and max freq indices
    if fmax == np.inf:
        fMaxIdx = Nf
        fmax = fCoords[-1]
    else:
        fMaxIdx = np.abs(fCoords - fmax).argmin()

    if fmin == 0:
        fMinIdx = 0
    else:
        fMinIdx = np.abs(fCoords - fmin).argmin()


    # Figure parameters
    if vmin is None:
        vmin = WDMMatrix.min() + 1e-20  # we can't have vmin = 0 because we might want to use LogNorm. So add a very small value to the vmin.
    if vmax is None:
        vmax = WDMMatrix.max()

    if logColor:
        norm = colors.LogNorm(vmin = vmin, vmax = vmax, clip = True)  # if you don't set clip = True, it'll plot every row of zeros as white for some reason, even though matplotlib.colors.Colormap's set_under defaults to 'k'. I have no idea what is happening but sure.  # Edit 6/30: now changing it back to False does nothing? i dont know if this is good or bad
    else:
        norm = colors.Normalize(vmin = vmin, vmax = vmax)


    # Plot with plt.pcolormesh
    fig, ax = plt.subplots(figsize=(10, 7))

    plt.pcolormesh(tCoords, fCoords[fMinIdx:fMaxIdx + 1], WDMMatrix, cmap = cmap, shading = 'auto', snap = False, rasterized = True, norm = norm)

    plt.grid(False)

    if plotFeatures:
        plt.xlabel('t (s)')
        plt.ylabel('f (Hz)')
        plt.title('WDM Spectrogram')
    else:
        # Courtesy of Gemini
        # 1. Turn off the axes
        ax.set_axis_off()

        # 2. Strip padding from the internal Axes object
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())


    if logFreq:
        ax.set_yscale('asinh', linear_width = 0.0005)  # 0.001 is good, 0.0001 is alright, 0.0005 is also good

    ax.set_ylim(fmin, fmax)
    plt.tight_layout()

    if showPlot:
        plt.show()

    if filename is not None:
        plt.savefig(Path(imageFolder) / (filename + '.jpg'), dpi = 300, pad_inches = 0.0)

    # If you don't do this, Jupyter Notebook will show you the plot either way (which ig is nice bc you don't need to always do plt.show() for it to show?)
    plt.close()


    return fig, ax, tCoords, fCoords[fMinIdx:fMaxIdx + 1]






def instantiateResponseFunc(dt, Tobs, allModes = False):
    # Step 1: Instantiate the IMRPhenomTHM object from phentax ---------------------------------------------------------------

    tlowfit = True # use a fit to set the starting time of the root finder used in t(f)
    tol = 1e-12 # root finding tolerance
    higher_modes = "all" if allModes else None

    imr = IMRPhenomTHM(
            higher_modes=higher_modes,  # higher_modes="all" for all modes
            include_negative_modes=True, # negative m modes will be produced by symmetry
            t_low_fit=tlowfit,
            coarse_grain=False, # if false it will generate the waveform on a dense time grid with the specified timestep
            atol=tol,
            rtol=tol,
            T=Tobs,
        )


    # Step 2: Initialize the ResponseWrapper object (it's basically a function, main attribute is __call__) ------------------

    # Some fastlisaresponse ResponseWrapper params
    index_lambda, index_beta = 0, 1
    force_backend = "cpu"
    orbits = EqualArmlengthOrbits()

    def compute_polarizations_at_once_new(m1, m2, chi1, chi2, distance, phi_ref, inclination, psi, dt, f_min, f_ref, T):
        times, mask, hplus, hcross = imr.compute_polarizations_at_once(m1 = m1, m2 = m2, chi1z = chi1, chi2z = chi2, distance = distance, phi_ref = phi_ref, inclination = inclination, psi = psi, f_min = f_min, f_ref = f_ref, delta_t = dt)
        return hplus[mask] + 1j * hcross[mask]

    lisaResponseFunc = ResponseWrapper(
        waveform_gen = compute_polarizations_at_once_new,
        Tobs = Tobs / ASTRONOMICAL_YEAR,  # Tobs is in years (can't we just always use SI?)
        dt = dt,
        index_lambda = index_lambda,
        index_beta = index_beta,
        flip_hx = False,
        remove_sky_coords = True,
        remove_garbage = True,
        force_backend = force_backend,
        orbits = orbits
        )


    # Step 3: Return the response function -----------------------------------------------------------------------------------

    return lisaResponseFunc


def getResponse(logmT, q, chi1, chi2, distance, cosinc, phi_ref, psi, f_min, f_ref, sinbeta, lam, dt, Tobs, allModes = False, responseFunc = None):
    # Step 1: Instantiate the response function as needed --------------------------------------------------------------------

    if responseFunc is None:
        responseFunc = instantiateResponseFunc(dt, Tobs, allModes = allModes)


    # Step 2: Get the response -----------------------------------------------------------------------------------------------

    mT = 10**logmT
    m1 = mT/(q + 1)
    m2 = mT - m1

    inclination = np.arccos(cosinc)
    beta = np.arcsin(sinbeta)

    waveA, waveE, waveT = responseFunc(
        beta,
        lam,
        m1 = m1,
        m2 = m2,
        chi1 = chi1,
        chi2 = chi2,
        distance = distance,
        phi_ref = phi_ref,
        inclination = inclination,
        psi = psi,
        dt = dt,
        f_min = f_min,
        f_ref = f_ref
    )


    # Step 3: Return the waveforms -------------------------------------------------------------------------------------------
    return waveA, waveE, waveT



def placeXIntoY(x, y, xIdx = 0, yIdx = 0, add = True, windowFunc = None, windowFuncArgs = [], fadeLengthStart = 0):  # TODO: add functionality for not specifying xIdx and/or yIdx, handle add vs. set, write docstring. i thought of adding a robust window function capability where you can pass in a window function (say scipy.signal.windows.tukey) and arguments and it'll apply the function, but if you want it windowed you can just pass x pre-windowed. fun learning idea though...
    """
    Places an array x into another array y at a given location yIdx that corresponds to xIdx.
    """

    # Old code ---------------------------------------------
    # # TODO: maybe rework the logic so that both of the 1st 2 cases are handled simultaneously (i.e. the first and last indices of totTimeSeries are determined, and the truncation of the original time series is determined, and then the addition is done). we could have variables like totTimeSeriesStartIdx and timeSeriesStartIdx, etc. to handle truncation and placement dynamically. Though, that might make it a bit more difficult to understand? idk. also we can make this into a function
    # if (tCIdx < zeroIdx):  # Handle cases where the coalescence time is too close to t = 0 and we need to truncate the first bit of the time series data.
    #     tIdxDiff = zeroIdx - tCIdx
    #     totTimeSeries[:tCIdx + (numMaskedTimes - zeroIdx)] += totalSignalMasked[tIdxDiff:] * windows.tukey(len(totalSignalMasked[tIdxDiff:]), 0.1)  # for the first one, can do numMaskedTime - tIdxDiff or tCIdx + (numMaskedTimes - zeroIdx).
    #     """
    #     Concrete example of how this works:
    #     Say my time series indices looks like this: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #     And let's say zeroIdx = 6 (that's where the coalescence happens)
    #     And let's say we need index 4 in totTimeSeries to be the coalescence time (tCIdx)
    #     Then tIdxDiff will be 6 - 4 = 2, and the next line will add to the first 8 (= tCIdx + (numMaskedTimes - zeroIdx) = 4 + 10 - 6) values in totTimesSeries by totalSignalMasked[2:] (which gives indices 2 to 9, which is 8 values).
    #     Let's see if index 4 in the new array (totTimeSeries) is the coalescence time (index 6 in the original array): 2, 3, 4, 5, 6 --> 0, 1, 2, 3, 4, so yes, the original index 6 corresponds to index 4 in the new array!
    #     Slap on a window function and we're good to go!
    #     Note that this logic doesn't take into account the edge case in which tCIdx is less than zeroIdx but the number of points after zeroIdx in the time series is greater than the number of points in totTimeSeries after tCIdx, but that won't happen since the total observation time will be long enough and there aren't that many points in the original time series after the coalescence occurs.
    #     """
    # elif (numTotTimes - tCIdx < numMaskedTimes - zeroIdx):  # Handle cases where the coalescence time is too close to t = TObsTot (2 yrs in our case) and we need to truncate the last bit of the time series data. Also, I don't like "elif" >:(
    #     tIdxDiff = (numMaskedTimes - zeroIdx) - (numTotTimes - tCIdx)
    #     totTimeSeries[tCIdx - zeroIdx:] += totalSignalMasked[:numMaskedTimes - tIdxDiff] * windows.tukey(len(totalSignalMasked[:numMaskedTimes - tIdxDiff]), 0.1)
    # else:  # If we're here, that means the coalescence time is nicely in the middle of the observation window and not too close to either of the edges.
    #     totTimeSeries[tCIdx - zeroIdx: tCIdx - zeroIdx + numMaskedTimes] += totalSignalMasked * windows.tukey(numMaskedTimes)
    # End of old code --------------------------------------

    def halfCosWindow(length):
        """Generates a half-cosine window of a given length."""
        return 0.5 * (1 - np.cos(np.linspace(0, np.pi, length)))


    xLen = len(x)
    yLen = len(y)

    idxDiffStart = yIdx - xIdx
    idxDiffEnd = (yLen - yIdx) - (xLen - xIdx)

    yStart = 0 if idxDiffStart < 0 else idxDiffStart
    yEnd = yLen if idxDiffEnd < 0 else idxDiffStart + xLen
    xStart = -idxDiffStart if idxDiffStart < 0 else 0
    xEnd = xLen + idxDiffEnd if idxDiffEnd < 0 else xLen

    #windowFunction = windowFunc(len(x[xStart:xEnd]), *windowFuncArgs) if windowFunc else 1

    # Windowing the start of the merger and the part after the merger
    xFaded = np.copy(x[xStart:xEnd])
    xFadedLen = len(xFaded)

    # Fade the start of the waveform
    newFadeLengthStart = min(fadeLengthStart, xFadedLen)
    fullWindow = halfCosWindow(fadeLengthStart)
    xFaded[:newFadeLengthStart] *= fullWindow[:newFadeLengthStart]

    # Fade the end of the waveform, from the coalescence up to the point at which the previous 10 points in time were less than 1e-25 in magnitude
    # valSum = 0
    # signalStopIdx = -1
    # if idxDiffStart >= 0:
    #     xIdxNew = xIdx
    # else:
    #     xIdxNew = xIdx + idxDiffStart
    # for idx, val in enumerate(xFaded[xIdxNew + 1:]):
    #     valSum += np.abs(val)
    #     if idx >= 10:
    #         valSum -= np.abs(xFaded[xIdxNew + 1:][idx - 10])
    #         if valSum / 10 <= 1e-25:
    #             signalStopIdx = idx
    #             break
    # if signalStopIdx != -1:
    #     xFaded[xIdxNew + 1:(xIdxNew + 1) + (signalStopIdx + 1)] *= halfCosWindow(signalStopIdx + 1)[::-1]
    # print(f'signalStopIdx: {signalStopIdx}')

    #print((xFaded == x[xStart:xEnd])[:200])

    # Fade the end of the waveform
    xIdxNew = xIdx if idxDiffStart >= 0 else xIdx + idxDiffStart
    xFaded[xIdxNew + 1:] *= halfCosWindow(xFadedLen - (xIdxNew + 1))[::-1]


    if add:
        y[yStart:yEnd] += xFaded
    else:
        y[yStart:yEnd] = xFaded

    return y




def TDIPlacement(totTimeSeries, tC, TDIData, dt, plot = True):
    """
    totTimeSeries: total time series that you want to place the binary waveform in
    tC: coalescence time in total observation time array (s), starting from t = 0 in the whole observation array
    TDIData: TDI time-domain waveform data (amplitude list)
    dt: amount of time between observations (s)
    plot: whether or not to plot the unpadded and padded (placed) waveforms; True by default
    """

    # Unpadded Time Series ---------------------------------------------------
    dataLen = len(TDIData)

    # Plot
    if plot:
        # Make time array
        dataTimes = np.arange(0, dataLen * dt, dt)

        plt.figure(figsize=FIGSIZE)
        plt.plot(dataTimes, TDIData)
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.title('Time Domain Waveform (unpadded)')
        plt.tight_layout()


    # Padded Time Series -----------------------------------------------------
    # Get coalescence time index
    tCIdx = int(tC / dt)

    # Get the coalescence time from the TDI data
    tCData = np.argmax(np.abs(TDIData))


    # Place waveA into the total time series
    fadeLengthStart = dataLen // 20
    placeXIntoY(TDIData, totTimeSeries, tCData, tCIdx, fadeLengthStart = fadeLengthStart)

    # Plot
    if plot:
        # Get total time series time array
        totTimeSeriesTimes = np.arange(0, numTotTimes * dt, dt)

        plt.figure(figsize = FIGSIZE)
        plt.plot(totTimeSeriesTimes, totTimeSeries)
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.title('Time Domain Waveform (padded)')
        plt.tight_layout()





def getBoundingBox(matrix, tCIdx, leftThreshold, tObsIdxSize, fMinIdx, lineExtension = -1, manualTopBoundaryY = None, manualLeftBoundaryX = None, manualBottomBoundaryY = None, topThreshold = None, logNormalize = True):
    """
    Get a bounding box for a merger. (Made by me and slightly fixed by Gemini.)

    matrix (array-like): A 2d matrix containing a spectrogram with a single BHB.
    tCIdx (int): the index of the coalescence time. This is also the right boundary of the box.
    threshold: the brightness threshold for determining the other 3 boundaries
    lineExtension (int): the width of the line (in pixels) added to each side of the center of the line, used for finding the other 3 boundaries.
    tObsIdxSize (int): TObs, converted to pixels. Used to determine the starting column for the left boundary.
    manualTopBoundaryY (int, optional): manually set the top boundary index.
    """

    print('bbox marker 1')
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    newMatrix = matrix.copy()
    print('bbox marker 2')
    if logNormalize:
        #vmin = newMatrix.min()
        #if vmin <= 0:
        #    vmin = 1e-20
        print('bbox marker 3')
        vmin = 10**(-4)  # orig: 1e-4  # 3.3  # 3.25 for imageData2
        vmax = newMatrix.max()
        #vmax = 10**(-1)
        print('bbox marker 4')

        print('bbox marker 5')
        logVmin = np.log10(vmin)
        print('bbox marker 6')
        logVmax = np.log10(vmax)
        print('bbox marker 7')
        logData = np.log10(newMatrix)
        print('bbox marker 8')

        print('bbox marker 9')
        newMatrix = (logData - logVmin) / (logVmax - logVmin)
        print('bbox marker 10')
        newMatrix = np.clip(newMatrix, 0, 1)
        print('bbox marker 11')

    print('bbox marker 2')
    nRows, nCols = newMatrix.shape
    rightBoundary = tCIdx if tCIdx < nCols else nCols - 1

    zeroIdx = int(False)
    posOne = int(True)
    negOne = -1

    leftExtend = lineExtension if tCIdx >= lineExtension else tCIdx
    rightExtend = lineExtension if (nCols - 1) - tCIdx >= lineExtension else (nCols - 1) - tCIdx
    leftStart = tCIdx - tObsIdxSize if tCIdx >= tObsIdxSize else zeroIdx

    topBoundary = bottomBoundary = leftBoundary = negOne

    print('bbox marker 3')

    # find top boundary
    if manualTopBoundaryY is not None:
        topBoundary = manualTopBoundaryY if manualTopBoundaryY < nRows else nRows - 1
    else:
        for row in reversed(range(nRows)):
            if lineExtension != negOne:
                leftSlice = tCIdx - leftExtend
                rightSlice = tCIdx + rightExtend + posOne
                vals = newMatrix[row, leftSlice:rightSlice]
            else:
                vals = newMatrix[row, :]

            if (vals >= topThreshold).any():
                topBoundary = row
                break
    print('bbox marker 4')

    # find left and bottom boundaries
    if manualLeftBoundaryX is not None and manualBottomBoundaryY is None:
        leftBoundary = manualLeftBoundaryX
        colData = newMatrix[fMinIdx:, leftBoundary]
        bottomBoundary = fMinIdx + colData.argmax()
    elif manualLeftBoundaryX is not None:
        leftBoundary = manualLeftBoundaryX
        bottomBoundary = manualBottomBoundaryY
    else:
        for col in range(leftStart, tCIdx):
            colData = newMatrix[fMinIdx:, col]
            validPixels = np.where(colData >= leftThreshold)[zeroIdx]

            if validPixels.size > zeroIdx:
                leftBoundary = col
                # grab the topmost pixel of the signal in this specific column
                bottomBoundary = fMinIdx + validPixels.max()
                break
    print('bbox marker 5')

    warningsTextArr = np.array(["Top", "Bottom", "Left"])
    boundariesArr = np.array([topBoundary, bottomBoundary, leftBoundary])

    if (boundariesArr == negOne).any():
        missingMask = boundariesArr == negOne
        missingBoundaries = ", ".join(warningsTextArr[missingMask])
        warnings.warn(f"The following boundaries were not found: {missingBoundaries}", UserWarning, stacklevel = 2)

    print('bbox marker 6')

    return leftBoundary, rightBoundary, bottomBoundary, topBoundary





def convertToYoloCoords(bboxIndices, tCoords, fCoords, yLimMin, yLimMax, asinhLinearWidth = 0.0005):
    """
    Converts matrix index boundaries into YOLO normalized coordinates,
    accounting for an asinh scaled y-axis. Made by Gemini.

    bboxIndices: tuple of (leftBoundary, rightBoundary, bottomBoundary, topBoundary)
    tCoords: 1D array of time coordinates
    fCoords: 1D array of frequency coordinates
    yLimMin: The lower limit of the matplotlib y-axis (e.g., 1e-8)
    yLimMax: The upper limit of the matplotlib y-axis (e.g., fmax)
    asinhLinearWidth: The linear_width parameter used in ax.set_yscale
    """

    leftIdx, rightIdx, bottomIdx, topIdx = bboxIndices

    # 1. get the physical data values (seconds and hz)
    tLeft = tCoords[leftIdx]
    tRight = tCoords[rightIdx]
    fBottom = fCoords[bottomIdx]
    fTop = fCoords[topIdx]

    # 2. handle the linear x-axis (time)
    tMin = tCoords[int(False)]
    tMax = tCoords[-1]

    xLeftNorm = (tLeft - tMin) / (tMax - tMin)
    xRightNorm = (tRight - tMin) / (tMax - tMin)

    # 3. handle the non-linear asinh y-axis (frequency)
    def asinhTransform(val):
        return np.arcsinh(val / asinhLinearWidth)

    sMin = asinhTransform(val = yLimMin)
    sMax = asinhTransform(val = yLimMax)

    yBottomNorm = (asinhTransform(val = fBottom) - sMin) / (sMax - sMin)
    yTopNorm = (asinhTransform(val = fTop) - sMin) / (sMax - sMin)

    # 4. invert the y-axis for yolo (yolo origin is top-left, matplotlib is bottom-left)
    yoloTop = 1.0 - yTopNorm
    yoloBottom = 1.0 - yBottomNorm

    # 5. calculate final yolo parameters
    xCenter = (xLeftNorm + xRightNorm) / 2.0
    yCenter = (yoloTop + yoloBottom) / 2.0
    width = xRightNorm - xLeftNorm
    height = yoloBottom - yoloTop

    # safely clip values between 0.0 and 1.0 to prevent out-of-bounds yolo errors
    xCenter = np.clip(a = xCenter, a_min = 0.0, a_max = 1.0)
    yCenter = np.clip(a = yCenter, a_min = 0.0, a_max = 1.0)
    width = np.clip(a = width, a_min = 0.0, a_max = 1.0)
    height = np.clip(a = height, a_min = 0.0, a_max = 1.0)

    return f'0 {xCenter:.6f} {yCenter:.6f} {width:.6f} {height:.6f}'



# Example usage in your loop:
# yLimMin = 1e-8  # from your ax.set_ylim(1e-8, fmax)
# yLimMax = fmax
#
# for idx, bbox in enumerate(BBoxes):
#     yoloString = convertToYoloCoords(bbox, tCoords, fCoords, yLimMin, yLimMax)
#     txtFileStr += f'{yoloString}\n'


def saveWaveformsToFile(filepath, indiv_waveforms, tot_waveform, params):
    """
    Saves generated waveforms and their random parameters to a file in append mode.
    The list order is individual unpadded waveforms, followed by the total padded waveform.
    Made by Gemini.
    """
    with open(filepath, 'ab') as file:
        waveform_list = indiv_waveforms + [tot_waveform]
        data_obj = {'waveforms': np.array(waveform_list, dtype=object), 'params': params}
        pickle.dump(data_obj, file)


def loadWaveformsFromFile(filepath):
    """
    Loads all saved waveform data from the file into a numpy array of objects.
    Made by Gemini.
    """
    data = []
    path = Path(filepath)
    if path.exists():
        with open(path, 'rb') as file:
            while True:
                try:
                    data.append(pickle.load(file))
                except EOFError:
                    break
    return np.array(data, dtype=object)


# TODO: Make checks for the 1st list item being < the 2nd item and provide useful messages if not; also check that tC is within the correct limits considering TObsTot and also TObs is in the correct limits, and all other variables (for example, beta and lamda (kinda; they repeat so idk) have min and max possible values; also, too high mass ratio might be bad). Edit: too high mass ratio was too bad.. doing 10:1 max lol; also, we should make it so that we clip the noise in case it gets too high, bc its possible it gives us a very large value even if it's improbable
def makeTrainingImage(
    numBHBsRange = [5, 12],
    # m1Range = [1e3, 1e7],
    # m2Range = [1e3, 1e7],
    logmTRange = [4,8],
    qRange = [0.1, 0.99999],
    chi1Range = [-0.9999999, 0.9999999],
    chi2Range = [-0.9999999, 0.9999999],
    distanceRange = [1e2, 1e5],
    cosincRange = [-1.0, 1.0],
    phiRefRange = [0, 2*np.pi],
    psiRange = [0, 2 * np.pi],
    fMinRange = [1e-8, 1e-8],
    fRefRange = [1e-8, 1e-8],
    sinbetaRange = [-1.0,1.0],
    lambdaRange = [0.0, 2 * np.pi],
    tCRange = [0, 2 * ASTRONOMICAL_YEAR],
    dt = 2.5,
    waveletDuration = 3600 * 3,
    filename = 'WDMImage',
    imageFolder = Path('imageData') / 'images' / 'train',
    showBBoxSpectrogram = False,
    vmin = 1e-8,
    vmax = None,
    fmin = 1e-8,
    fmax = 1e-2,
    noiseLevel = 10**(-22.5),
    TObs = 5 * ASTRONOMICAL_YEAR / 12,
    TObsTot = 2 * ASTRONOMICAL_YEAR,
    allModes = False,
    responseFunc = None,
    multiprocessing = False,
    topThresh = 0.7,
    leftThresh = 0.6,
    precomputedData = None,
    saveWaveformsFile = None):


    # Step 1a: Load precomputed time series ---------------------------------------------------------------------------------------------------------------

    rng = np.random.default_rng()
    if precomputedData is not None:
        randomArgsDict = precomputedData['params']
        waveform_list = precomputedData['waveforms']
        
        tCs = randomArgsDict['tC']
        numBHBs = len(tCs)
        
        indiv_waveforms_unpadded = list(waveform_list[:-1])
        totTimeSeries = waveform_list[-1].copy()
        cleanTotTimeSeries = totTimeSeries.copy()

    
    # Step 2a: Place the loaded individual time series into their own padded arrays -----------------------------------------------------------------------
    
        numTotTimes = int(TObsTot / dt)
        indivTimeSeries = np.zeros((numBHBs, numTotTimes))
        for idx, wf_unpadded in enumerate(indiv_waveforms_unpadded):
            TDIPlacement(indivTimeSeries[idx], tCs[idx], wf_unpadded, dt, plot = False)

    
    # Step 1b: Generate the random parameters -------------------------------------------------------------------------------------------------------------

    else:

        numBHBs = rng.integers(numBHBsRange[0], numBHBsRange[1], endpoint = True)
        tCs = rng.uniform(tCRange[0], tCRange[1], numBHBs)

        randomArgsDict = {}
        rangesList = [logmTRange, qRange, chi1Range, chi2Range, distanceRange, cosincRange, phiRefRange, psiRange, fMinRange, fRefRange, sinbetaRange, lambdaRange]
        randomParamNameList = ['logmT', 'q', 'chi1', 'chi2', 'distance', 'cosinc', 'phi_ref', 'psi', 'f_min', 'f_ref', 'sinbeta', 'lam'] 
        for idx, paramName in enumerate(randomParamNameList):
            randomArgsDict[paramName] = rng.uniform(rangesList[idx][0], rangesList[idx][1], numBHBs)
        
        randomArgsDict['tC'] = tCs


    # Step 2b: Generate numBHBs waveforms -----------------------------------------------------------------------------------------------------------------
        
        waveforms = []
        for idx in range(numBHBs):
            argSetDict = {paramName: paramVal[idx] for paramName, paramVal in randomArgsDict.items() if paramName != 'tC'}
            waveforms.append(getResponse(
                **argSetDict,
                dt = dt,
                Tobs = TObs,
                allModes = allModes,
                responseFunc = responseFunc
            ))

        indiv_waveforms_unpadded = [wf[0] for wf in waveforms]


    # Step 3b: Place the numBHBs waveforms into a total padded array and into their own padded arrays -----------------------------------------------------

        numTotTimes = int(TObsTot / dt)
        totTimeSeries = np.zeros(numTotTimes)
        for idx, wf in enumerate(waveforms): 
            TDIPlacement(totTimeSeries, tCs[idx], wf[0], dt, plot = False) 

        indivTimeSeries = np.zeros((numBHBs, numTotTimes))
        for idx, wf in enumerate(waveforms): 
            TDIPlacement(indivTimeSeries[idx], tCs[idx], wf[0], dt, plot = False)

        cleanTotTimeSeries = totTimeSeries.copy()


    # Step 3a/4b: Add noise to all the time arrays --------------------------------------------------------------------------------------------------------

    maxAmp = max(totTimeSeries)
    print(f'{maxAmp:.3e}')

    # Total time series
    noise1 = np.clip(rng.normal(loc = 0.0, scale = 1.0, size = numTotTimes), -2.5, 2.5)
    noise2 = np.diff(rng.normal(loc = 0.0, scale = 1.0, size = numTotTimes + 1))
    A1 = 10**(-20.2)  # 10**(-20.2) when we restricted the mT/distance priors  # 10**(-20.2 for imageData4)
    A2 = 0  # A2 = 10 makes it so that noise1 and noise2 intersect at 3 mHz
    #A1 = maxAmp * 0.008  # .01 for imageData2
    noiseTot = A1 * (noise1 + noise2 * A2)
    
    totTimeSeries += noiseTot

    # Individual time series
    noise3 = np.clip(rng.normal(loc = 0.0, scale = 1.0, size = numTotTimes), -2.5, 2.5)* 10**(-7)  # 10**(-2.5) for imageData2
    for indivTimeSer in indivTimeSeries:
        A3 = 0.0001 * max(indivTimeSer)
        newNoise = noiseTot * 1e-5
        indivTimeSer += newNoise


    # Step 4a/5b: Make a spectrum from the noise time series ----------------------------------------------------------------------------------------------

    #nSegs = 100
    #noiseTotSpectrum = welch(noiseTot, fs = 1/dt, nperseg = numTotTimes / nSegs)

    #plt.figure()
    #plt.plot(noiseTotSpectrum[0], noiseTotSpectrum[1], color = 'blue', label = 'A1(n1 + A2*n2)')
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('TDI')
    #plt.legend()
    #plt.loglog()
    #plt.tight_layout()
    #plt.savefig(f'imageDataTest/images/train/{filename}_noisePlot.jpg', dpi = 300)
    #plt.close()


    # Step 4a/5b: Make all the WDM matrices ---------------------------------------------------------------------------------------------------------------

    # Get nT, nF, nD
    nT = int(TObsTot / waveletDuration)
    nD = len(totTimeSeries)
    nF = int(nD / nT)

    # Force nF and nT to be even because WDMWaveletTransforms says it assumes they're even and if they're not then results can be inaccurate
    if nF % 2 != 0:
        nF -= 1
    if nT % 2 != 0:
        nT -= 1

    nD = nF * nT

    print('b4 wdm total')
    # Total time series WDM map
    masterWDM, absFMinIdx, _ = getWDMMatrix(totTimeSeries, nT, nF, noiseLevel = noiseLevel, fmin = fmin, fmax = fmax, dt = dt)
    print('after wdm total')

    # Individuals
    WDMs = [getWDMMatrix(timeSeries, nT, nF, noiseLevel = 1e-32, noiseLevel2 = 10**(-33), fmin = fmin, fmax = fmax, dt = dt)[0] for timeSeries in indivTimeSeries]
    print('after wdm indiv')


    # Step 5a/6b: Make the spectrogram image of the total time series -------------------------------------------------------------------------------------

    print('b4 specgram')
    _, _, tCoords, fCoords = getSpectrogram(masterWDM, nT, nF, dt, vmin = vmin, vmax = vmax, fmin = fmin, fmax = fmax, showPlot = False, filename = filename, imageFolder = imageFolder, cmap = 'inferno')
    print('after specgram')


    # Step 6a/7b: Get the bounding boxes for each BHB (iff we're saving the spectrogram image) ------------------------------------------------------------

    if filename is not None:
        BBoxes = []

        fCoordSpacing = 1 / waveletDuration / 2  # 1 / WAVELET_DURATION for frequency of wavelet; / 2 for Nyquist. Can also do fCoords[-1] / len(fCoords).
        for idx, WDM in enumerate(WDMs):
            mT = 10**(randomArgsDict['logmT'][idx])
            m1 = mT/(randomArgsDict['q'][idx] + 1)
            m2 = mT - m1
            fISCO = 4400 / (m1 + m2)

            aboveNoise = np.where(np.log10(np.abs(indivTimeSeries[idx])) >= -22.12 + np.log10(2))[0]  # log(ab) = log(a) + log(b), so, noise = 1e-20.2 and we want noisyValues >= noise * 2 --> log10(noisyValues) >= -20.2 + log10(2)  # -22.12 + log10(2) for imageData4
            print(aboveNoise)
            if len(aboveNoise) == 0:
                print(f'BHB of idx {idx} didn\'t have any location >= noise so it likely wont be visible and will have a trash bbox')
                leftIdx = leftIdxWDM = None
            else:
                leftIdx = aboveNoise.min()
                actualTime = leftIdx * dt
                leftIdxWDM = int(np.ceil(actualTime / waveletDuration))  # using ceil() bc we still need to find the bottom boundary so we can't have it be too far to the left but we can have it be too far to the right

                betaParam = 64 * GRAVITATIONAL_CONSTANT**3 * (m1 * SOLAR_MASS) * (m2 * SOLAR_MASS) * SOLAR_MASS * (m1 + m2) / (5 * C**5)
                print(betaParam)
                T_c = tCs[idx] - actualTime
                print(T_c)
                a0 = (4 * betaParam * T_c) ** (1/4)
                print(a0)
                #a = (a0**4 - 4 * betaParam * actualTime)**(1/4)
                #print(a)
                fBottomOrb = 1/np.sqrt(4 * np.pi**2 * a0**3 / (GRAVITATIONAL_CONSTANT * mT * SOLAR_MASS))
                print(fBottomOrb)
                fBottomGW = 2 * fBottomOrb
                print(fBottomGW)
                fBottomGWIdx_absolute = int(np.ceil(fBottomGW / fCoordSpacing))
                fBottomGWIdx = max(0, fBottomGWIdx_absolute - absFMinIdx)
                fBottomGWIdx = min(fBottomGWIdx, WDM.shape[0] - 1)  # safely cap it at the top of the matrix just in case  # edit by me: actually my bbox function probably accounts for this anyway but whatever
                print(fBottomGWIdx)

            #BBoxes.append(getBoundingBox(WDM, round(tCs[idx] / waveletDuration), threshold = 0.2, tObsIdxSize = int(TObs / waveletDuration), fMinIdx = 0, manualTopBoundaryY = round(fISCO * 2 / fCoordSpacing), logNormalize = True))
            print('before bboxes')
            BBoxes.append(getBoundingBox(WDM, round(tCs[idx] / waveletDuration) + 5, topThreshold = topThresh, leftThreshold = leftThresh, tObsIdxSize = int(TObs / waveletDuration), fMinIdx = 0, logNormalize = True, manualLeftBoundaryX = leftIdxWDM, manualBottomBoundaryY = fBottomGWIdx))  # 0.63, 0.565  # 0.62, 0.5  # 0.63, 0.47 for imageData2/imageData3
            print('after bboxes get')


        print('before bbox write to file')
        testTrainVal = Path(imageFolder).name
        with (Path(imageFolder).parent.parent / "labels" / testTrainVal / (filename + ".txt")).open('w') as file:  # TODO: maybe make this a bit nicer with relpath or something rather than parent.parent and whatever
            txtFileStr = ""
            for bbox in BBoxes:
                fmaxConversionFunc = fmax if fmax != np.inf else fCoords[-1]
                yoloString = convertToYoloCoords(bbox, tCoords, fCoords, fmin, fmaxConversionFunc)
                txtFileStr += f'{yoloString}\n'
            txtFileStr = txtFileStr.removesuffix('\n')

            file.write(txtFileStr)
        print('after bbox write to file')

    # Step 7a/8b: Save the params for each BHB (iff we're saving the spectrogram image) -------------------------------------------------------------------
    # Made by me and modified by Gemini

    if filename is not None:
        randomArgsDict['tC'] = tCs

    # convert numpy arrays to lists for safe yaml serialization
    yamlDict = {}
    for key, val in randomArgsDict.items():
        if isinstance(val, np.ndarray):
            yamlDict[key] = val.tolist()
        else:
            yamlDict[key] = val

    testTrainVal = Path(imageFolder).name
    outPath = Path(imageFolder).parent.parent / "params" / testTrainVal / (filename + ".yaml")

    with outPath.open(mode = 'w') as file:
        yaml.dump(data = yamlDict, stream = file, default_flow_style = False)


    # Step 9b: Save the un-noised waveforms if required ---------------------------------------------------------------------------------------------------
    
    if saveWaveformsFile is not None and precomputedData is None:
        global fileLock
        if 'fileLock' in globals() and fileLock is not None:
            with fileLock:
                saveWaveformsToFile(saveWaveformsFile, indiv_waveforms_unpadded, cleanTotTimeSeries, randomArgsDict)
        else:
            saveWaveformsToFile(saveWaveformsFile, indiv_waveforms_unpadded, cleanTotTimeSeries, randomArgsDict)



    # Step 8a/10b: Return WDMs/Time Series ----------------------------------------------------------------------------------------------------------------

    return masterWDM, totTimeSeries, WDMs, indivTimeSeries






# global variable for the worker processes to hold the c-backend wrapper
workerResponseFunc = None
fileLock = None

def initWorker(dt, tObs, lock):
    """
    initializes the response function once per cpu core.
    """
    global workerResponseFunc, fileLock
    workerResponseFunc = instantiateResponseFunc(dt = dt, Tobs = tObs, allModes = False)
    fileLock = lock

def generateSingleTask(args):
    """
    wrapper to execute the image generation with robust retry logic and aggressive cleanup.
    """
    fileName, folderPath, topThresh, leftThresh, vmin, precomputedData, saveWaveformsFile = args
    
    dtConst = 2.5
    tObsConst = 2 * ASTRONOMICAL_YEAR / 12
    
    # retry loop -----------------------------------------------------------------
    success = False
    while not success:
        try:
            makeTrainingImage(
                filename = fileName,
                imageFolder = folderPath,
                dt = dtConst,
                TObs = tObsConst,
                responseFunc = workerResponseFunc,
                precomputedData = precomputedData,
                saveWaveformsFile = saveWaveformsFile,
                vmin = 10**(-4),  # 1e-3.3 is decent  # 1e-4 for imageData2  # 1e-6 prev test
                noiseLevel = 10**(-20.8),
                vmax = 2e-1,  # 2e-1
                fmin = 1e-4,
                fmax = 1e-2,
                logmTRange = [5.5, 7],
                distanceRange = [5e2, 1e3]
                #distanceRange = [4e3, 1.2e5]
                #distanceRange = [1e2, 1e3]
                
                #logmTRange = [6.8, 7],  # 6.5, 7  # imageData4
                #distanceRange = [10**(3.45), 10**(3.5)]  # imageData4
            )
            success = True
        except Exception as e:
            print(f'makeTrainingImage failed: {e}')
            print('Printing traceback:')
            traceback.print_exc()
        finally:
            # strictly required to prevent matplotlib from hoarding ram across loops
            plt.close(fig = 'all')
            gc.collect()

def getOptimalCoreCount(ramPerTaskGb = 8.0):
    """
    dynamically calculates the maximum number of cores to use based on physical ram.
    prevents ssd swap thrashing which severely degrades performance.
    """
    posOne = int(True)
    
    try:
        # read total physical ram in bytes (works natively on macos/linux)
        bytesRam = os.sysconf(name = 'SC_PAGE_SIZE') * os.sysconf(name = 'SC_PHYS_PAGES')
        gbRam = bytesRam / (1024 ** 3)
        
        # calculate how many tasks fit in memory, leaving a few gb for the os
        availableGb = max(0, gbRam - 4.0)
        maxCoresByRam = max(posOne, int(availableGb / ramPerTaskGb))
        
        # use the maximum safe cores, but don't exceed actual cpu cores
        return min(os.cpu_count(), maxCoresByRam)
    except Exception:
        # fallback if sysconf fails
        return 2
    
def generateDataset(totalImages, trainValTest, baseDir = "imageData", topThresh = 0.5, leftThresh = 0.1, vmin = 1e-6, waveformFile = None, useSavedWaveforms = False):
    """
    calculates splits, sets up directories, and launches the optimized pool.
    """
    loaded_waveforms = []
    if useSavedWaveforms and waveformFile is not None:
        loaded_waveforms = loadWaveformsFromFile(waveformFile)
        print(f"Loaded {len(loaded_waveforms)} previously generated waveforms.")

    # split calculations ---------------------------------------------------------
    numTrain = math.floor(totalImages * trainValTest[0] / 100)
    numVal = math.floor(totalImages * trainValTest[1] / 100)
    numTest = totalImages - numTrain - numVal
    
    basePath = Path(baseDir)
    splits = ['train', 'val', 'test']
    counts = [numTrain, numVal, numTest]
    
    tasks = []
    task_idx = 0
    
    # directory setup and task queueing ------------------------------------------
    for splitIdx in range(len(splits)):
        splitName = splits[splitIdx]
        splitCount = counts[splitIdx]
        
        imagePath = basePath / 'images' / splitName
        labelPath = basePath / 'labels' / splitName
        paramPath = basePath / 'params' / splitName
        
        imagePath.mkdir(parents = True, exist_ok = True)
        labelPath.mkdir(parents = True, exist_ok = True)
        paramPath.mkdir(parents = True, exist_ok = True)
        
        for imgIdx in range(splitCount):
            fName = f"{splitName}_img_{imgIdx:04d}"
            
            precomputed = loaded_waveforms[task_idx] if task_idx < len(loaded_waveforms) else None
            save_file = waveformFile if precomputed is None and waveformFile is not None else None
            
            tasks.append((fName, imagePath, topThresh, leftThresh, vmin, precomputed, save_file))
            task_idx += 1
            
    # launch the dynamically optimized pool --------------------------------------
    optimalCores = getOptimalCoreCount(ramPerTaskGb = 8.0)
    tasksPerChildLimit = 1
    
    print(f"system optimizing: running {totalImages} images across {optimalCores} cores to maximize speed without swap thrashing...")
    
    dtConst = 2.5
    tObsConst = 5 * ASTRONOMICAL_YEAR / 12

    manager = Manager()
    lock = manager.Lock()
    
    with Pool(processes = optimalCores, initializer = initWorker, initargs = (dtConst, tObsConst, lock), maxtasksperchild = tasksPerChildLimit) as pool:
        pool.map(func = generateSingleTask, iterable = tasks)
        
    print("dataset generation complete.")






# TODO: implement the args other than image count
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numImages', type = int, default = 100, help = 'How many images you want to generate.')
    parser.add_argument('--topThresh', type = float, default = 0.5, help = 'Top threshold (after log-norming) for top boundary of BHBs in WDM matrix.')
    parser.add_argument('--leftThresh', type = float, default = 0.1, help = 'Left threshold (after log-norming) for left and bottom boundaries of BHBs in WDM matrix.')
    parser.add_argument('--vmin', type = float, default = 1e-6, help = 'Minimum value for pcolormesh spectrogram. Higher will show you less of the noise.')
    parser.add_argument('--trainValTest', type = str, default = '[70, 20, 10]', help = 'Train/Val/Test split in the form of a Python list; default is [70, 20, 10].')
    parser.add_argument('--baseDir', type = str, default = 'imageData', help = 'Directory containing "images", "labels", and "params" folders for storing files.')
    parser.add_argument('--waveformFile', type = str, default = None, help = 'File to save/load waveforms. Defaults to baseDir/saved_waveforms.pkl')
    parser.add_argument('--useSavedWaveforms', action = 'store_true', help = 'Use previously generated waveforms if available.')
    
    args = parser.parse_args()

    trainValTestList = [float(x) for x in args.trainValTest.strip('[]').replace(',', '').split()]
    print(type(trainValTestList), trainValTestList)
    print(type(args.trainValTest), args.trainValTest)

    # resolve the waveform file path to default inside baseDir
    waveformPath = args.waveformFile
    if waveformPath is None:
        waveformPath = str(Path(args.baseDir) / 'saved_waveforms.pkl')

    generateDataset(
        totalImages = args.numImages, 
        baseDir = args.baseDir, 
        trainValTest = trainValTestList,
        topThresh = args.topThresh,
        leftThresh = args.leftThresh,
        vmin = args.vmin,
        waveformFile = waveformPath,
        useSavedWaveforms = args.useSavedWaveforms
    )
    
    
    #masterWDM, totTimeSeries, WDMs, indivTimeSeries = makeTrainingImage(filename = 'largeBBoxImage', imageFolder = 'imageDataTest/images/train', dt = 2.5, TObs = 2 * ASTRONOMICAL_YEAR, vmin = 1e-4, numBHBsRange = [1, 1], chi1Range = [-0.39516464010775015, -0.39516464010775015], chi2Range = [-0.7407309689028186, -0.7407309689028186], cosincRange = [0.6939273133128703, 0.6939273133128703], distanceRange = [2540.9108632486586, 2540.9108632486586], fMinRange = [1e-8, 1e-8], fRefRange = [1e-8, 1e-8], lambdaRange = [2.2239381290901217, 2.2239381290901217], logmTRange = [6.9696011111446134, 6.9696011111446134], phiRefRange = [1.1913378991974717, 1.1913378991974717], psiRange = [3.9071082290317456, 3.9071082290317456], qRange = [0.21028293636056916, 0.21028293636056916], sinbetaRange = [0.4168029896124097, 0.4168029896124097], tCRange = [25863595.73272265, 25863595.73272265])

    #newMatrix = masterWDM.copy()
    #vmin2 = 10**(-3.25)  # orig: 1e-4  # 3.3
    #vmax = newMatrix.max()

    #logVmin = np.log10(vmin2)
    #logVmax = np.log10(vmax)
    #logData = np.log10(newMatrix)

    #newMatrix = (logData - logVmin) / (logVmax - logVmin)
    #newMatrix = np.clip(newMatrix, 0, 1)

    #getSpectrogram(newMatrix, dt = 2.5, cmap = 'Dark2', plotFeatures = True, vmin = 1e-4)


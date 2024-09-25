""" 
Carter code for illumination calculations

https://zenodo.org/records/10534584

Definitions required to create figures 
Last update by JLCarter on 2024, 01, 18

License:

MIT License

Copyright (c) 2024 carterphysicslabs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np

def getDiff(fun1, fun2):
    # calculate the difference between two arrays
    return fun1 - fun2

def getPerDiff(fun1, fun2):
    # calculate the percent difference between two arrays
    # reshape to 1D so for loop can be run through single dimension
    res = np.sqrt(fun1.size)
    res = res.astype(np.int64)
    fun1 = np.reshape(fun1, res ** 2)
    fun2 = np.reshape(fun2, res ** 2)
    # do the operations
    top = (fun1 - fun2)
    bottom = 0.5 * (fun1 + fun2)
    # if bottom is zero and both functions are always positive or zero,
    # then in an area where both fun1 and fun2 are zero
    # and the difference is zero, handle this
    thePerDiff = np.zeros(np.shape(top))   
    for i in range(0,len(thePerDiff)):
        if bottom[i] != 0:
            thePerDiff[i] = top[i] / bottom[i]
        
    thePerDiff = thePerDiff * 100
    # reshape before return
    return np.reshape(thePerDiff, [res, res])


def getScaled(a, Rp, Rs):
    # determine the scaled semi-major axis a and planet radius Rp
    # inputs:
    # a = semi-major axis in AU
    # Rp = planetary radius in earth radii
    # Rs = stellar radius in solar radii
    # outputs:
    # aRs = scaled semi-major axis in stellar radii
    # RpRs = scaled planetary radius in stellar radii
    
    # do the conversion
    a = a * 1.495978707e8 # in km
    Rp = Rp * 6378.1 # in km
    Rs = Rs * 695700 # in km

    aRs = a / Rs
    RpRs = Rp / Rs
    
    return aRs, RpRs

def latlon_to_xyz(lat, lon):
    # get the points on the planet as xyz positions, lat = latitude in deg, lon = longitude in deg
    latrad = lat * (np.pi / 180)
    lonrad = lon * (np.pi / 180)
    # see equation 4 in Paper 1 for coordinate system. Set so rotation to LOS has +z is pointed toward observer.
    x = np.sin(lonrad) * np.cos(latrad)
    y = np.sin(latrad)
    z = np.cos(lonrad) * np.cos(latrad)
    return x, y, z

def getReff(Rs, rs):
    # get effective radius of star
    # inputs:
    # Rs = stelllar radius 
    # rs = star-planet seperation
    # NOTE: Rs and rs must be in the same units.
    
    return Rs * np.sqrt(1 - ((Rs - 1) / rs) ** 2)

def equiHemi(r,num):
    # generates about num points equally over a hemisphere, 
    # outputs: 
    # theta = polar angle in radians
    # phi = azimuthal angle in radians.
    
    # Based on 2004 paper by Markus Deserno, Max-Planck-Institut:
    # "How to generate equidistributed points on the surface of a sphere"
    
    # initialize vectors, have a couple of extra just in case
    thetavec = np.zeros(num+2) 
    phivec = np.zeros(num+2)
    #Break out if zero points
    if num==0:
        return thetavec, phivec
    
    # approximate area of each tile, reduced to hemisphere
    a = (2.0 * np.pi * (r ** 2.0)) / num 
    # distance for tile "sides"
    d = np.sqrt(a) 
    # increment for theta, reduced to pi/2 for hemisphere
    m_theta = int(round((np.pi / 2) / d)) 
    # goal is d_theta * d_phi ~ a for a unit sphere
    # remember to only go to pi/2!
    d_theta = (np.pi/ 2) / m_theta 
    # increment for phi
    d_phi = a / d_theta 
    
    # code cannot distrubute all vales of num equally, count actual 
    # number of tiles generated
    newNUM = 0

    for m in range(0,m_theta):
        # remember to only go to pi/2
        theta = (np.pi / 2) * (m + 0.5) / m_theta 
        m_phi = int(round(2.0 * np.pi * np.sin(theta) / d_phi))
        for n in range(0,m_phi):
            phi = 2.0 * np.pi * n / m_phi
            thetavec[newNUM] = theta
            phivec[newNUM]= phi
            newNUM = newNUM + 1
        
    if newNUM < num+2:
        thetavec = thetavec[0:newNUM]
        phivec = phivec[0:newNUM]
    
    return  thetavec, phivec, a, newNUM

def getSourcePoints(source_npts, Rs, rs):
    # generate displacement of points onto sphere of star
    # drQ = 0 if a point source, place at center of host star
    # this is based on starry's code (ver 1.2.0), found here:
    # https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/_core/core.py#L916
    # here we use numpy instead of theano tensor variables
    if source_npts <= 1: 
        source_dx = np.array([0.0])
        source_dy = np.array([0.0])
        source_dz = np.array([0.0])
    else: 
        # determine the number of increments, N, by calculating radius 
        # r = sqrt(source_npts * 4/ np.pi) of circle with area A = source_npts * unit area
        # 4/pi converts from radians to square units. 
        # adds 2 and then takes the integer value to round down.
        # area of each cell with be about 1/source_pts
        N = int(2+np.sqrt(source_npts * 4/ np.pi))
        
        # get effective radius of star
        Reff = getReff(Rs, rs)
    
        dx = np.linspace(-1,1,N)
        dx, dy = np.meshgrid(dx, dx)
        dz2 = 1-dx**2-dy**2 
        # keep only real solutions for dz
        source_dx = dx[dz2 > 0].flatten()
        source_dy = dy[dz2 > 0].flatten()
        
        # scale by Reff
        source_dx = source_dx * Reff
        source_dy = source_dy * Reff
        # calculate dz, negative because CLOSER to planet
        source_dz = -np.sqrt(Rs ** 2 - source_dx ** 2 - source_dy ** 2)
        
        # set new number of source points
        source_npts = len(source_dx)
    return source_dx, source_dy, source_dz, source_npts

def getEquiSourcePoints(source_npts, Rs, rs):
    # calculates the cartesian coordinates of the source points equally 
    # distributed over a hemisphere. IN PROGRESS
    if source_npts <= 1: # no offsets
        source_dx = np.array([0.0])
        source_dy = np.array([0.0])
        source_dz = np.array([0.0])
        dA = 1
        newNUM = source_npts
    else: 
        # get effective radius of star
        Reff = getReff(Rs, rs)
        # get theta and phi using equiHemi
        theta, phi, a, newNUM = equiHemi(1, source_npts)
#         dA = 1 / newNUM
        dA = a
        # get the cartesian coordinates on a unit sphere
        source_dx = np.sin(theta) * np.cos(phi)
        source_dy = np.sin(theta) * np.sin(phi)
#         source_dz = np.cos(theta)
        # scale to Reff
        source_dx *= Reff
        source_dy *= Reff
        # get dz, negative because CLOSER to planet
        # make sure dx^2 + dy^2 + dz^2 = Rs^2
        source_dz = -np.sqrt(Rs ** 2 - source_dx ** 2 - source_dy ** 2)
#         source_dz *= -Rs
        
    return source_dx, source_dy, source_dz, newNUM, dA

def getPtLatLong(RpRs, aRs, num):
    # get extent of illumination, generates longitudes and latitudes at which terminator is located due to 
    # point soruce illumination.
    
    # point source terminates to form spherical cap at
    cappt = RpRs / aRs
    taupt = np.arccos(cappt)
    
    # determine vectors defining the spherical cap in latitude and logitude, 
    # this is not a circle in rectangular projection space!
    # start with point source vectors, always will be in hemisphere pointed toward star
    latptvec = np.linspace(-taupt, taupt, num)
    # cappt^2 + y^2 = Rp^2, so
    longptvec = np.pi /2 -  np.arcsin(cappt / np.cos(latptvec)) 
    # get left and right of substellar point
    latPT = np.concatenate((latptvec, np.flip(latptvec)))
    longPT = np.concatenate((longptvec, -np.flip(longptvec)))
    # convert to degrees
    xtermpt = longPT * (180 / np.pi)
    ytermpt = latPT * (180 / np.pi)
    
    return xtermpt, ytermpt

def getFullLatLong(RpRs, aRs, num):
    # get extent of the fully illuminated zone for extended source illumination.
    # zone froms a sphereical cap with apex angle tau
    cap = (1 + RpRs) / aRs
    tau = np.arccos(cap)
    
    # determine vectors defining the spherical cap in latitude and logitude
    # here we assume the substellar point is at 0, 0.
    # start with point source vectors, always will be in hemisphere pointed toward star
    latFullvec = np.linspace(-tau, tau, num)
    # because capFull^2 + y^2 = Rp^2, so
    longFullvec = np.pi /2 -  np.arcsin(cap / np.cos(latFullvec)) 
    # get left and right
    latFull = np.concatenate((latFullvec, np.flip(latFullvec)))
    longFull = np.concatenate((longFullvec, -np.flip(longFullvec)))
    # convert to degrees
    xFull = longFull * (180 / np.pi)
    yFull = latFull * (180 / np.pi)
    
    return xFull, yFull

def getExtLatLong(RpRs, aRs, num):
    # determine extent of illumination for extended source
    # extended source terminates tauext beyond pi/2, see figure 2
    capext = (1 - RpRs) / aRs
    tauext = (np.pi / 2) - np.arcsin(capext)

    # repeat for extedend case, which lines beyond hemisphere facing illumination source
    latextvec = np.linspace(-tauext, tauext, num) 
    longextvec = np.arcsin(capext / np.cos(latextvec)) 
    # won't use concatenation here because of discontinuity over hemispheres
    xtermextPOS = longextvec * (180 / np.pi)+90
    xtermextNEG = -90 - longextvec * (180 / np.pi)
    ytermext = latextvec * (180 / np.pi)
    
    return xtermextPOS, xtermextNEG, ytermext

def getIpt(lat, lon, xs, ys, zs, res):
    """ determine illumination as function of latitude and longitude on planet
        assuming point source illumination. """
    # lat and lon are in degrees and location on planet
    # xs, ys, and zs, are the location of the point source (center of star) from center of planet
    
    # get the star-planet seperation squared
    rs2 = xs ** 2 + ys ** 2 + zs ** 2

    # # get x, y and z coordinates on unit sphere from lat and lon
    x, y, z = latlon_to_xyz(lat, lon)

    # # get the location of each point on planet
    p = np.vstack([x, y, z])
    s = np.array([xs, ys, zs])

    # # Ipt is the dot product of the normal of p and the normal of s divided by (pi rs**2)
    cos_theta_i = np.dot(s, p) / (1 * np.sqrt(rs2))
    cos_theta_i = np.maximum(cos_theta_i, 0)
    Ipt = cos_theta_i / (np.pi * rs2)
    Ipt = np.reshape(Ipt, res ** 2)
    
    return Ipt

def getIptxi(lat, lon, xs, ys, zs, res):
    # lat and lon are in degrees and location on planet
    # xs, ys, and zs, are the location of the point source (center of star) from center of planet
    
    # get the star-planet seperation squared
    rs2 = xs ** 2 + ys ** 2 + zs ** 2

    # get x, y and z coordinates on unit sphere from lat and lon
    x, y, z = latlon_to_xyz(lat, lon)

    # get the location of each point on planet
    p = np.vstack([x, y, z])
    s = np.array([xs, ys, zs])

    # Calculate rho, the difference between s and each point on surface of illuminated body
    # v is a holder array each column is elements of s
    v = s[:, np.newaxis] 
    rhovec = v-p


    # calculate magnitudes for future division as part of calculating cosine of angles
    # rho squared is dot prodcut with self
    rho2 = np.sum(rhovec*rhovec, axis = 0) 

    # take dot product of columns
    rhodotp = np.sum(rhovec*p, axis=0) 
    rhodotp = np.maximum(0,rhodotp)
    
    cos_xi = rhodotp/(1 * np.sqrt(rho2))
    Illum_xi = cos_xi / (np.pi * rho2)
    
#     Illum_xi = np.reshape(Illum_xi, res ** 2)

    return Illum_xi

def getCosGammaPrime(lat, lon, xs, ys, zs):
    # updated to use xs, ys, and zs position
    # expects lat and lon to be in degrees and for them to be 1D
    a = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    Rp = 1
    den = a * Rp
    
    x, y, z = latlon_to_xyz(lat, lon)
    
    cosGammaPrime = (xs * x + ys * y + zs * z) / den
    return cosGammaPrime

def getIplane(lat, lon, xs, ys, zs):
    # determine illumination as function of lat and lon on planet using 
    # plane parallel ray approximation. See equation (8)
    a2 = xs ** 2 + ys ** 2 + zs ** 2
    cosGammaPrime = getCosGammaPrime(lat, lon, xs, ys, zs)
    Iplane = (1 / (np.pi * a2)) * cosGammaPrime
    Iplane = np.maximum(0, Iplane)
    return Iplane

def getIfull(lat, lon, xs, ys, zs, Rs):
    """ Calculate illumination in fully illuminated zone of exoplanet for 
        the extended source case. 
        """
    
    # a is star-planet seperation in units of Rp
    a2 = xs ** 2 + ys ** 2 + zs ** 2
    a = np.sqrt(a2)
    
    # set cosgammaPRIME to nan if less than sinETA1, edge of fully illuminated zone
    cosgammaPRIMEnan = getCosGammaPrime(lat, lon, xs, ys, zs)
    sinETA1 = (Rs + 1) / a
    cosgammaPRIMEnan[cosgammaPRIMEnan < sinETA1] = np.nan
    
    # see equation 26
    Ifull = (a * cosgammaPRIMEnan - 1) / ((a2+1-2*a*cosgammaPRIMEnan) ** (3/2))
    Ifull /= np.pi

    # get parts that are negative and set to zero, should not be necessary?
    Ifull = np.maximum(0, Ifull)
    return Ifull


def getIfinite(lat, lon, xs, ys, zs, Rs, source_npts, point_type = "disk"):
    """ Written by J. L. Carter, unknown date in Summer 2023
        determine intensity using extended source code from Ben Placek with my corrections,
        Makes use of source point code of starry code and Reff
        instead of generating all on a sphere
        instead of testing if cosxi > 0 and costhetaprime > 0, just set negatives to zero.
        See Equation (11)
        INPUTS:
            lat = latitude(s) of point(s) on planet
            lon = longitude(s) of point(s) on planet
            xs, ys, and zs = location of center of host star (illumination source)
            Rs = stellar radius in units of the planetary radius
            source_npts = number of source points to put on the star
            point_type = optional parameter, string. Default argument is "disk" and uses STARRY's
                         generation of points. Other argument option is "equi" which uses equiHemi
        OUTPUTS:
            Ifinite = np.array of intensity for each point on planet in units of stellar intensity.
    
        UPDATES:
            2023/09/18 by JLC = added optional parameter point_type to determine which generation
                                of points on star to use and set dA appropriately. Note: equi is 
                                in progress!
    """
    
    # get the star-planet seperation of centers squared
    rs2 = xs ** 2 + ys ** 2 + zs ** 2
    sep = np.sqrt(rs2)
    # generate points on star
    if point_type == "disk":
        dx, dy, dz, numpts = getSourcePoints(source_npts, Rs, sep)

        # compute area of each cell on stellar surface
        dA = (1 / numpts)
    elif point_type == "equi":
        dx, dy, dz, numpts, dA = getEquiSourcePoints(source_npts, Rs, sep)
#         dA *= (Rs ** 2)
    else:
        print("point_type not recognized, using default of disk")
        dx, dy, dz, numpts = getSourcePoints(source_npts, Rs, sep)

        # compute area of each cell on stellar surface
        dA = (1 / numpts)

    
    # make dr a matrix and get magnitude
    dr = np.vstack([dx, dy, dz])
    magdr = np.sqrt(np.sum(dr*dr, axis = 0))
    drhat = dr / magdr

    # get x, y and z coordinates on unit sphere from lat and lon
    xp, yp, zp = latlon_to_xyz(lat, lon)

    # get the location of each point on planet
    p = np.vstack([xp, yp, zp])
    # location of center of star
    s = np.array([xs, ys, zs])
    
    # Calculate rq positions
    # svec is holder array each column is elements of s
    svec = s[:, np.newaxis] 
    rq = svec + dr
    
    # initialize Ifinite, should be shape of p[1] (res ** 2, )
    Ifinite = np.zeros(np.shape(p[1]))
    # don't plot if not analyzed
    Ifinite[:] = np.nan 
    
    for pp in range(0,np.size(p[1])):
        # pull current point
        curP = p[:,pp]
        # get rhoprime for all rq
        curP = curP[:,np.newaxis]
        curRhoprime = rq - curP
        # get magnitude of rhoprime squared for division to determine cos_xi and cos_thetaprime
        curRhoprime2 = np.sum(curRhoprime * curRhoprime, axis = 0)
        
        # get cos_xi = dot(p, rhoprime)/(|p|*|rhoprime|) size should be that of rq
        pdotrhoprime = np.sum(curP * curRhoprime, axis = 0)
        # drop the negatives
        pdotrhoprime = np.maximum(0, pdotrhoprime) 
        cos_xi = pdotrhoprime / (1 * np.sqrt(curRhoprime2))
        # get cos_thetaprime = - dot(dr, rhoprime) / (|dr| * |rhoprime|)
        drdotrhoprime = np.sum(dr * curRhoprime, axis = 0)
        cos_thetaprime = -drdotrhoprime / (magdr * np.sqrt(curRhoprime2))
        # drop the negatives
        cos_thetaprime = np.maximum(0, cos_thetaprime) 
        
        # Ifinite(pp) is the sum of all (cos_xi * cos_thetaprime) / (|rhoprime| ^ 2)
        Ifinite[pp] = np.sum( cos_xi * cos_thetaprime / curRhoprime2, axis = 0)
    
    
    Ifinite *= (dA / np.pi)
#     Ifinite = np.reshape(Ifinite, res ** 2)
    
    return Ifinite


def getIfinite2(lat, lon, xs, ys, zs, Rs, source_npts, point_type = "disk"):
    """ written by J. L. Carter summer of 2023
        determine intensity using same ideas as getIfinite, but neglecting cos_thetaprime which
        accounts for the foreshortening of the star (like STARRY).
        See equation (28)
        INPUTS:
            lat = latitude(s) of point(s) on planet
            lon = longitude(s) of point(s) on planet
            xs, ys, and zs = location of center of host star (illumination source)
            Rs = stellar radius in units of the planetary radius
            source_npts = number of source points to put on the star
            point_type = optional parameter, string. Default argument is "disk" and uses STARRY's
                         generation of points. Other argument option is "hemi" which uses equiHemi
        OUTPUTS:
            Ifinite = np.array of intensity for each point on planet in units of stellar intensity.
    
        UPDATES:
            2023/09/18 by JLC = added optional parameter point_type to determine which generation
                                of points on star to use and set dA appropriately. Note that equi 
                                is in progress.
    """
    
    # get the star-planet seperation squared
    rs2 = xs ** 2 + ys ** 2 + zs ** 2
    sep = np.sqrt(rs2)
    # generate points on star# generate points on star
    if point_type == "disk":
        dx, dy, dz, numpts = getSourcePoints(source_npts, Rs, sep)

        # compute area of each cell on stellar surface
        dA = (1 / numpts)
    elif point_type == "equi":
        dx, dy, dz, numpts, dA = getEquiSourcePoints(source_npts, Rs, sep)
    else:
        print("point_type not recognized, using default of disk")
        dx, dy, dz, numpts = getSourcePoints(source_npts, Rs, sep)

        # compute area of each cell on stellar surface
        dA = (1 / numpts)
    
    # make vector and get magnitude
    dr = np.vstack([dx, dy, dz])
    magdr = np.sqrt(np.sum(dr*dr, axis = 0))
    drhat = dr / magdr

    # get x, y and z coordinates on unit sphere from lat and lon
    xp, yp, zp = latlon_to_xyz(lat, lon)

    # get the location of each point on planet
    p = np.vstack([xp, yp, zp])
    # location of center of star
    s = np.array([xs, ys, zs])
    
    # Calculate rq positions
    # svec holder array each column is elements of s
    svec = s[:, np.newaxis] 
    rq = svec + dr
    
    # initialize Ifinite, should be shape of p[1] (res ** 2, )
    Ifinite = np.zeros(np.shape(p[1]))
    # don't plot if not analyzed, debugging
    Ifinite[:] = np.nan 
    
    for pp in range(0,np.size(p[1])):
        # pull current point
        curP = p[:,pp]
        # get rhoprime for all rq
        curP = curP[:,np.newaxis]
        curRhoprime = rq - curP
        # get magnitude of rhoprime squared for division to determine cos_xi and cos_thetaprime
        curRhoprime2 = np.sum(curRhoprime * curRhoprime, axis = 0)
        
        # get cos_xi = dot(p, rhoprime)/(|p|*|rhoprime|) size should be that of rq
        pdotrhoprime = np.sum(curP * curRhoprime, axis = 0)
        pdotrhoprime = np.maximum(0, pdotrhoprime) # drop the negatives
        cos_xi = pdotrhoprime / (1 * np.sqrt(curRhoprime2))
        
        # Ifinite(pp) is the sum of all (cos_xi * cos_thetaprime) / (|rhoprime| ^ 2)
        Ifinite[pp] = np.sum( cos_xi / curRhoprime2, axis = 0)
    
    
    Ifinite *= (dA / np.pi)
#     Ifinite = np.reshape(Ifinite, res ** 2)
    
    return Ifinite


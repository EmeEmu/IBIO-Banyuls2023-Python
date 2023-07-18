"""Functions and routines to get ZBrain, brain and regions, contours."""
import os
import sys

import h5py
import numpy as np


def get_vals_from_ref(file, ref):
    """Handle h5 file referencing system.

    :file: <HDF5 file> .h5 file opened with h5py.
    :ref: <HDF5 object reference> hdf5 reference.

    return
        :dat,dats: <array> data contained at reference pointer.
    """
    name = h5py.h5r.get_name(ref, file.id)
    dat = file[name][:]

    if type(dat[0, 0]) == h5py.h5r.Reference:
        dats = []
        for d in dat[0]:
            name = h5py.h5r.get_name(d, file.id)
            dats.append(file[name][:])
        return dats
    else:
        return dat


def get_ZBrain_brain_countours(
    path=os.path.dirname(__file__) + "/../content/banyuls_data/ContourZBrain.mat",
    factx=0.8*1.e-3,
    facty=0.8*1.e-3,
    factz=1*1.e-3,
):
    """Get the xy and yz contours of the ZBrain reference brain.

    :path: <string> path to .mat file containing contours.
    :fact x,y,z: <float> conversion factor for coordinates.

    return
        (contourxy, contouryz) <arrays of shape (2,nb_points)> points along the contours.
    """
    file = h5py.File(path, "r")

    dname_xy = "CountourBrainXY"
    dname_yz = "CountourBrainYZ"

    contourxy = get_vals_from_ref(file, file[dname_xy][0, 0])
    contouryz = get_vals_from_ref(file, file[dname_yz][0, 0])

    contourxy[0] *= factx
    contourxy[1] *= facty
    contouryz[0] *= facty
    contouryz[1] *= factz

    return contourxy, contouryz


def get_ZBrain_regions_countours(
    regions_inds=[0, 182, 93, 113, 259, 274],
    path=os.path.dirname(__file__) + "/../content/banyuls_data/ContourZBrain.mat",
    factx=0.8*1.e-3,
    facty=0.8*1.e-3,
    factz=1*1.e-3,
):
    """Get the xy and yz contours of regions in the ZBrain reference brain.

    :regions_inds: <int or list or 1d array> indice(s) of regions to get.
    :path: <string> path to .mat file containing contours.
    :fact x,y,z: <float> conversion factor for coordinates.

    return
        :contours: <list of lenght=len(regions_inds)>
                    data organisation is the following (complex because one
                    region can be split into multiple contours):
                    contours[regions][xy or yz][contour][dim1 or dim2][points]
                    points along the contours.
    """

    if type(regions_inds) == int:
        regions_inds = np.array([regions_inds])
    elif type(regions_inds) == list:
        regions_inds = np.array(regions_inds)
    elif type(regions_inds) == np.ndarray:
        pass
    else:
        raise TypeError(
            "Wrong type for :regions_inds: ." " Should be <int> or <list> or <ndarray>."
        )
    regions_inds = np.sort(regions_inds)

    file = h5py.File(path, "r")

    data_xy = file["CountourBrainRegionsXY"]
    data_yz = file["CountourBrainRegionsYZ"]
    data_xy = data_xy[regions_inds]
    data_yz = data_yz[regions_inds]

    contours = []
    for d_xy, d_yz in zip(data_xy, data_yz):
        XYs = get_vals_from_ref(file, d_xy[0])
        YZs = get_vals_from_ref(file, d_yz[0])
        for i in range(len(XYs)):
            XYs[i][0] *= factx
            XYs[i][1] *= facty
        for i in range(len(YZs)):
            YZs[i][0] *= facty
            YZs[i][1] *= factz
        contours.append([XYs, YZs])

    return contours

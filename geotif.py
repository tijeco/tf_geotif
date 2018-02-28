from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from osgeo import osr, gdal
import rasterio
from affine import Affine
from pyproj import Proj, transform
import glob

SAMPLES = glob.glob('*.tif')
print(len(SAMPLES))



def gtif_toGPS(gtiffFile):
    with rasterio.open(gtiffFile) as r:
        T0 = r.affine  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read(1)  # pixel values

    cols, rows = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[0]))
    T1 = T0 * Affine.translation(0.5, 0.5)
    rc2en = lambda r, c: (c, r) * T1
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)
    p2 = Proj(proj='latlong',datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)
    return (longs,lats)


def gtif2np(gtiffFile):
    im = Image.open(gtiffFile)
    pix = np.array(im)
    return pix

fname = 'wc2.0_10m_tmin_01.tif'
file2 = 'wc2.0_10m_tmin_02.tif'
file_dict = {}
file3 = "bio19.tif"
im = Image.open(file3)
# for tif_file in SAMPLES:
#     print(tif_file)
#     try:
#         file_dict[tif_file]=gtif2np(tif_file)
#     except:
#         print(tif_file,"failed")
n1, n2 = np.arange(280,400), np.arange(450,650)
# print(n1)
longs,lats = gtif_toGPS(fname)
pix = gtif2np(fname)
pix2 = gtif2np(file2)
new_longs = longs[n1[:,None], n2[None,:]]
new_lats = lats[n1[:,None], n2[None,:]]
new_pix = pix[n1[:,None], n2[None,:]]
new_pix2 = pix2[n1[:,None], n2[None,:]]
new_combined = np.dstack((new_pix,new_pix2,new_longs,new_lats))
print(new_combined[0][0])

# combinedPix = np.dstack((pix,pix2,longs,lats))
#
# print(combinedPix[0][0])

print(pix.shape)
print(pix2.shape)
# print(combinedPix.shape)


# print(n1,n2)
# neww = combinedPix[n1[:,None], n2[None,:]]
# print(neww)
# plt.imshow(new_pix)
# plt.show()

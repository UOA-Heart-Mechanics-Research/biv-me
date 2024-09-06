from argparse import ArgumentParser
import logging

import numpy as np
import SimpleITK as sitk

from .thickness import compute_thickness
from vtscan_utils.misc import keep_biggest_cc

logging.basicConfig(level=logging.DEBUG)

parser = ArgumentParser()
parser.add_argument("endo")
parser.add_argument("epi")
parser.add_argument("output")
args = parser.parse_args()

print("Reading images")
endo_sitk = sitk.ReadImage(args.endo)
epi_sitk = sitk.ReadImage(args.epi)

endo = sitk.GetArrayFromImage(endo_sitk).astype(bool)
epi = sitk.GetArrayFromImage(epi_sitk).astype(bool)

print("Preparing labeled image")
endo = keep_biggest_cc(endo, connectivity=1)
labeled_image = np.zeros_like(epi, np.uint8)
labeled_image[epi] = 2
labeled_image[endo] = 1

print("Computing thickness")
thickness = compute_thickness(labeled_image,
                              spacing=endo_sitk.GetSpacing()[::-1])

thickness_sitk = sitk.GetImageFromArray(thickness)
thickness_sitk.CopyInformation(endo_sitk)

print("Writing result")
sitk.WriteImage(thickness_sitk, args.output, True)

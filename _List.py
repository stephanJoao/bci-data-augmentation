import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from Models.DataAugmentation.GaussianNoise import GaussianNoise
from Models.DataAugmentation.TimeFrequencyRecombination import TimeFrequencyRecombination
from Models.DataAugmentation.TimeFrequencyRecombinationReloaded import TimeFrequencyRecombinationReloaded
from Models.DataAugmentation.TimeFrequencyGaussianNoise import TimeFrequencyGaussianNoise
from Models.DataAugmentation.EmpiricalModeDecomposition import EmpiricalModeDecomposition

DataAugmentation_list = {
	'GaussianNoise': GaussianNoise,
	'TimeFrequencyRecombination': TimeFrequencyRecombination,
	'TimeFrequencyRecombinationReloaded': TimeFrequencyRecombinationReloaded,
	'TimeFrequencyGaussianNoise': TimeFrequencyGaussianNoise,
	'EmpiricalModeDecomposition': EmpiricalModeDecomposition
}
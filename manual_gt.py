import numpy as np
import os

"""
Amplitude - drop the low ones
presence - drop sections
add random (take from far away, same pairwise as below)
drop random
take from nearby, lose from nearby (do 2 seperate pairwise runs and combine during analysis)
trade with nearby (percentage based on connected based on highest similarity)
full splits (like presence but keep the dropped section)

when we drop, need to drop them from all the ones that have times -0 
pc_features, template_features, spike_times, spike_templates, spike_clusters, amplitudes
^^^ instead of this we are just adding to a 1 massive noise cluster
"""

class SortManipulator(object):
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.spike_clusters = np.load(os.path.join(data_dir, 'spike_clusters.npy'))
		self.amplitudes = np.load(os.path.join(data_dir, 'amplitudes.npy'))


	def manipulate_spike_clusters(self, percentage_decimal, manipulation_mask_generator):


	def amplitude_mask(id_mask):
		id_amplitudes = self.amplitudes[id_mask]
		mean = np.mean(self.amplitudes)
		std = np.std(self.amplitudes)
		cutoff = (mean+3*std) - 6*std*percentage_decimal

	def manipulate_amplitude(self, percentage_decimal):
		lower_half_id = np.max(self.spike_clusters)+1
		amp_spike_clusters = self.spike_clusters.copy()
		for cluster_id in np.unique(self.spike_clusters):
			print('Dropping low amplitude spikes for '+str(cluster_id))
			id_mask = self.spike_clusters==cluster_id

			#for idx, amplitude in enumerate(self.amplitudes):
		#		if id_mask[idx] and (amplitude<cutoff):
	#				amp_spike_clusters[idx] = lower_half_id
			amp_spike_clusters[] = lower_half_id
		file_dict = {
			'spike_clusters.npy': amp_spike_clusters
		}
		save_manipulated_sort('amplitude', str(percentage_level), file_dict)

	def save_manipulated_sort(self, manipulation_type, level, file_dict):
		dest = os.path.join(self.data_dir, manipulation_type+'_'+level)
		os.makedirs(dest, exist_ok=True)
		for file_name, array in file_dict.items():
			file = os.path.join(dest, file_name)
			np.save(file, array)




if __name__ == '__main__':
	data_dir = r"C:\Users\greggh\Documents\Python Scripts\metrics_analysis\412804"
	sort_412804 = SortManipulator(data_dir)
	sort_412804.manipulate_amplitude(.25)
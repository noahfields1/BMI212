#Noah wrote this code

from pandas_plink import read_plink1_bin
import numpy as np

#Choosing the correct plink files
G = read_plink1_bin("dataset_04669194_plink.bed", "dataset_04669194_plink.bim", "dataset_04669194_plink.fam", verbose=True)
#G = read_plink1_bin("dataset_49623708_plink.bed", "dataset_49623708_plink.bim", "dataset_49623708_plink.fam", verbose=True)

# Convert the list to a NumPy array and reshape it to a column vector
list_data = np.array(G.coords['sample'].values.tolist()).reshape(-1, 1)

# Add the list as the first column of the existing matrix
new_matrix = np.column_stack((list_data,G.values)).astype(str)


output_file = 'cases_29SNPs.csv'
#output_file = 'controls_29SNPs.csv'

#Choosing the correct output file
# Save the array to a text file
np.savetxt(output_file, new_matrix,delimiter=',', fmt='%s')

"""
Generate .tex results table
"""

import os
import scipy.io

test_list_file = '../training-code/master-test-list.txt'
template_file = '../documents/test-results-table-template.tex'
output_file = '../documents/test-results-table.tex'
data_file = '../documents/test-results.mat'

mat_data = scipy.io.loadmat(data_file)

# Delete the old output_file if it exists
try:
	os.remove(output_file)
except OSError:
	pass

# Open the output file
with open(output_file, 'a') as out:
	# Open the template and loop through, printing each line
	with open(template_file, 'r') as template:
		for line in template:
			# If the line contains TEST_NAME, write results
			if 'TEST_NAME' not in line:
				out.write(line)
			else:
				# Loop over tests and results
				with open(test_list_file, 'r') as tests:
					for name, error in zip(tests, mat_data['percent_error'][0]):

						out.write(
							line.replace('TEST_NAME', name.rstrip()) \
								.replace('TEST_RESULT', '%2.3f' % error)
							)

"""
Generate .tex files for all tests.
"""

import os

test_list_file = '../training-code/master-test-list.txt'
template_file = '../documents/training-figures-template.tex'
output_file = '../documents/training-figures.tex'

# Delete the old output_file if it exists
try:
	os.remove(output_file)
except OSError:
	pass

with open(output_file, 'a') as out:
	# Open the list of tests and loop over test
	with open(test_list_file, 'r') as tests:
		for test in tests:
			test_name = test.rstrip()
			print('Processing ' + test.rstrip() + '...')
			# Open the template and loop over it
			with open(template_file, 'r') as template:
				for line in template:
					out.write(line.replace('TEST_NAME', test_name))
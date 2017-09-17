#!/usr/bin/python

import sys, math

if len(sys.argv) != 3:
    print 'usage: %s data predictions' % sys.argv[0]
    sys.exit()

data_file = sys.argv[1]
predictions_file = sys.argv[2]

data = open(data_file)
predictions = open(predictions_file)

# Load the real labels.
true_labels = []
for line in data:
    true_labels.append(line.split()[0])

predicted_labels = []
for line in predictions:
    predicted_labels.append((line.strip()).strip(','))

data.close()
predictions.close()

if len(predicted_labels) != len(true_labels):
    print 'Number of lines in two files do not match.'
    sys.exit()

'''    
match = 0
total = len(predicted_labels)

for ii in range(len(predicted_labels)):
	predicted_label = float(predicted_labels[ii]);
	true_label = float(true_labels[ii]);
	if round(predicted_label,0) == round(true_label,0):
        	match += 1

print 'Accuracy: %f (%d/%d)' % ((float(match)/float(total)), match, total)
'''
# Compute the average squared error.
total_error = 0

for ii in range(len(predicted_labels)):
	predicted_label = float(predicted_labels[ii]);
	true_label = float(true_labels[ii]);
	total_error += abs(predicted_label - true_label)

mean_error = total_error / float(len(predicted_labels))
print 'Error: %f' % (mean_error)

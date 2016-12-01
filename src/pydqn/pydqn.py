import numpy as np
import caffe

net = caffe.Net('dqn.prototxt','dqn.caffemodel',caffe.TEST)
q_values = net.blob_by_name('q_values')

	

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from object_detection.utils import metrics

true='true_positives'
false='fasle_positives'

#0-unworking_chimney,1-working_chimney,2-unworking_condensing,3-working_condensing

for n in range(4):
    i=str(n)
    a=np.loadtxt('/home/dq/models/research/object_detection/chimney/evaluation/recall'+i+'.txt')
    b=np.loadtxt('/home/dq/models/research/object_detection/chimney/evaluation/precision'+i+'.txt')
    #k[n]=np.loadtxt('/home/dq/models/research/object_detection/chimney/evaluation/num_gt'+i+'.txt')
    average_precision = metrics.compute_average_precision(b, a)
    print('average_precision',i,':',average_precision)
    plt.plot(a,b)
    plt.xlabel('recall'+i)
    plt.ylabel('precision'+i)
    plt.show()

# ~ statis=[0,0,0,0]
	
# ~ for n in range(41):
	# ~ i=str(n)
    # ~ a=np.loadtxt('/home/dq/models/research/object_detection/chimney/evaluation/classesimage-'+i+'.txt')
    # ~ b=np.loadtxt('/home/dq/models/research/object_detection/chimney/evaluation/scoresimage-'+i+'.txt')
    # ~ c=(b>=0.5)+0
    # ~ d=(a+1)*c
    # ~ for z in range(len(a)):
      # ~ if d[z]<0||d[z]>4:
        # ~ raise NumError
      # ~ else:
        # ~ statis[d[z]-1]++

# ~ ch_det=statis[0]+statis[1]
# ~ cd_det=statis[2]+statis[3]

# ~ ch_real= k[0]+k[1]
# ~ cd_real= k[2]+k[3]

     	  

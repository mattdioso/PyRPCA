import matplotlib
import numpy as np

def rpca_imshow(results, field, imgNum):
	if field not in results:
		print("[RPCA_imshow] %s is not a field name for this struct", field)
	else:
		mat = lambda x : np.reshape(x, results[0], results[1])
		matplotlib.pyplot.set_cmap('Gray')
		im = matplotlib.pyplot.figure()
		matplotlib.pyplot.imshow(mat(['results.' field '(:,imgNum)']), [])

	return im
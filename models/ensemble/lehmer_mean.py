
import numpy as np

def lehmer_mean (x, p, w=None):
	w = np.zeros(len(x))+1/len(x) if w is None else w / np.sum(w)
	return np.sum(w*x**p) / np.sum(w*x**(p-1))

def lehmer_mean_deriv (x, p, w=None):
	w = np.zeros(len(x))+1/len(x) if w is None else w / np.sum(w)
	lnx = np.log(x); wxp = w*x**p; wxp1 = w*x**(p-1)
	return np.sum(wxp*lnx)/np.sum(wxp1) - (np.sum(wxp)*np.sum(wxp1*lnx))/np.sum(wxp1)**2

# predictions is (samples, models, classes), truth is (samples, classes)
def optimise_lehmer_mean (predictions, truth, iterations=20, w=None):
	p = 1 # init with arithmetic mean
	lr = 0.1 # learning rate
	w = np.zeros(len(x))+1/len(x) if w is None else w / np.sum(w)

	for i in range(iterations):

		# optimise weights
		# w += 10 * np.sum([
		# 	[lehmer_mean(sample[:,cid], p, w=w) * (lehmer_mean(sample[:,cid], p, w=w) - truth[i]) for cid in range(sample.shape[1])]
		# 	for i, sample in enumerate(predictions)
		# ])
		w /= np.sum(w)
		# print(w)

		# matrix of (samples, classes) using lehmer mean across predictions for the same class from different models
		output = np.array([
			[lehmer_mean(sample[:,cid], p, w=w) for cid in range(sample.shape[1])]
			for sample in predictions
		])
		# matrix of derivatives of the prediction with respect to base 'p'
		output_wrt_p = np.array([
			[lehmer_mean_deriv(sample[:,cid], p, w=w) for cid in range(sample.shape[1])]
			for sample in predictions
		])
		try:
			c_wrt_p = (output - truth) * output_wrt_p
		except:
			print(f'output.shape is {output.shape}, truth.shape is {truth.shape}, output_wrt_p.shape is {output_wrt_p.shape}')
			quit()
		p -= lr * np.sum(c_wrt_p)

	return p

def run_tests ():

	# 20 'models', 200 binary vector-valued 'samples' (and 20*200=4000 predictions)

	num_models = 20
	num_samples = 200
	num_classes = 2

	mu = 10
	samples = np.clip(np.random.randn(num_samples, num_models, num_classes) + mu, 0, 2 * mu)

	truths_arithmetic = np.average(samples, axis=1)
	print(optimise_lehmer_mean(samples, truths_arithmetic)) # should be 1

	truths_geometric = np.array([
		[sample[:,cid].prod()**(1.0/len(sample[:,cid])) for cid in range(sample.shape[1])]
		for sample in samples
	])
	print(optimise_lehmer_mean(samples, truths_geometric)) # should be 0.5

	truths_min = np.min(samples, axis=1)
	print(optimise_lehmer_mean(samples, truths_min)) # should tend to -infty

	truths_max = np.max(samples, axis=1)
	print(optimise_lehmer_mean(samples, truths_max)) # should tend to infty

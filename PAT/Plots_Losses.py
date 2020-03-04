from IPython import display
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(i,losses, lam, num_epochs):
	display.clear_output(wait=True)
	display.display(plt.gcf())

	ax1 = plt.subplot(311)
	values = np.array(losses["L_f"])
	#plt.plot(range(len(values)), values, label=r"$L_f$", color="blue")
	#plt.legend(loc="upper right")
        plt.plot(range(len(values)), values, color="blue")
	if(i==num_epochs-1):
		plt.plot(range(len(values)), values, label=r"$L_f$", color="blue")
		plt.legend(loc="upper right")

	ax2 = plt.subplot(312, sharex=ax1)
	values = np.array(losses["L_r"]) / lam
	#plt.plot(range(len(values)), values, label=r"$L_r$", color="green")
	#plt.legend(loc="upper right")
        plt.plot(range(len(values)), values, color="green")
	if(i==num_epochs-1):
		plt.plot(range(len(values)), values, label=r"$L_r$", color="green")
		plt.legend(loc="upper right")

	ax3 = plt.subplot(313, sharex=ax1)
	values = np.array(losses["L_f - L_r"])
	#plt.plot(range(len(values)), values, label=r"$L_f - \lambda L_r$", color="red")
	#plt.legend(loc="upper right")
	plt.plot(range(len(values)), values, color="red")
	if(i==num_epochs-1):
		plt.plot(range(len(values)), values, label=r"$L_f - \lambda L_r$", color="red")
		plt.legend(loc="upper right")

        if(i==num_epochs-1):  
 	       plt.show()
	       plt.savefig('Losses.pdf')
	       plt.savefig('Losses.png')
	       plt.close()


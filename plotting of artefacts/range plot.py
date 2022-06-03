import numpy as np
import matplotlib.pyplot as plt

input_data = np.memmap("C:/Users/Marc/Desktop/data_mclass/val_model_data" + "/model_input.dat", dtype='float32', mode='r', shape=(28, 22, 549200))
target = np.memmap("C:/Users/Marc/Desktop/data_mclass/val_model_data" + "/model_target.dat", dtype='float32', mode='r', shape=(28, 22, 549200))

jo = input_data[1, 0, 66000:66000+60000]
tar = target[1, 0, 66000:66000+60000]

fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)

ax1.set_title("Annotations of the artifacts")
ax1.plot(np.abs(jo) > 200-200/1000, color = "r", label = "abs(EEG) >= 199.8")
ax1.plot(tar, label = "annotation")
ax1.set_xticks([0, 60000], ["5:30", "10:30"])
ax1.legend()

ax2.set_title("The EEG recording")
ax2.plot(jo, label = "FP1-F7")
ax2.set_xticks([0, 60000], ["5:30", "10:30"])
ax2.legend()

ax3.set_title("Annotations of the artifacts")
ax3.plot((np.abs(jo) > 200-200/1000)[40600:50000], color = "r", label = "abs(EEG) >= 199.8")
ax3.plot(tar[40600:50000], label = "annotation")
ax3.set_xticks([0, 9400], ["8:53", "9:40"])
ax3.legend(loc = "upper right")

ax4.set_title("The EEG recording")
ax4.plot(jo[40600:50000], label = "FP1-F7")
ax4.set_xticks([0, 9400], ["8:53", "9:40"])
ax4.legend()

plt.show()
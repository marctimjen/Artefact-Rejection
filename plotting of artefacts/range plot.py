import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt

target = np.memmap("C:/Users/Marc/Desktop/data_mclass/val_model_data" + "/model_target.dat", dtype='float32', mode='r', shape=(28, 22, 549200))

file_dir = r"C:\Users\Marc\Desktop\data_series\v2.1.0\edf\01_tcp_ar\002\00000254\s007_2013_03_25\00000254_s007_t000.edf"


file = mne.io.read_raw_edf(file_dir, preload=True)
data = file.filter(0.1, 100)
data = data.notch_filter(60)
data = data.resample(sfreq=200)
df = data.to_data_frame()
file = df["EEG FP1-REF"]-df["EEG F7-REF"]

jo = np.array(file[66000:66000+60000])
tar = target[1, 0, 66000:66000+60000]

fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)

ax1.set_title("Annotations of the artifacts")
ax1.plot(np.abs(jo) > 200-200/1000, color = "r", label = "EEG >= 199.8")
ax1.plot(tar, label = "annotation")
ax1.set_xticks([0, 60000], ["5:30", "10:30"])
ax1.legend(loc = "upper right")

ax2.set_title("The EEG recording")
ax2.plot(jo, label = "FP1-F7")
ax2.set_xticks([0, 60000], ["5:30", "10:30"])
ax2.set_ylim([-300, 490])
ax2.legend(loc = "upper left")

ax3.set_title("Annotations of the artifacts")
ax3.plot((np.abs(jo) > 200-200/1000)[40600:50000], color = "r", label = "EEG >= 199.8")
ax3.plot(tar[40600:50000], label = "annotation")
ax3.set_xticks([0, 9400], ["8:53", "9:40"])
ax3.legend(loc = "upper right")

ax4.set_title("The EEG recording")
ax4.plot(jo[40600:50000], label = "FP1-F7")
ax4.set_xticks([0, 9400], ["8:53", "9:40"])
ax4.set_ylim([-210, 350])
ax4.legend(loc = "upper right")

plt.show()


target = np.memmap("C:/Users/Marc/Desktop/data_mclass/val_model_data" + "/model_target.dat", dtype='float32', mode='r', shape=(28, 22, 549200))

file_dir = r"C:\Users\Marc\Desktop\data_series\v2.1.0\edf\01_tcp_ar\002\00000254\s005_2010_11_15\00000254_s005_t000.edf"


file = mne.io.read_raw_edf(file_dir, preload=True)
data = file.filter(0.1, 100)
data = data.notch_filter(60)
data = data.resample(sfreq=200)
df = data.to_data_frame()
file = df["EEG FP1-REF"]-df["EEG F7-REF"]

jo = np.array(file[66000+60000:66000+2*60000])
tar = target[0, 0, 66000+60000:66000+2*60000]

fig, (ax2, ax4) = plt.subplots(1, 2) # ((ax1, ax3), (ax2, ax4))

# ax1.set_title("Annotations of the artifacts")
# ax1.plot(np.abs(jo) > 200-200/1000, color = "r", label = "EEG >= 199.8")
# ax1.plot(tar, label = "annotation")
# ax1.set_xticks([0, 60000], ["10:30", "15:30"])
# ax1.legend()

ax2.set_title("The EEG recording")
ax2.plot(jo, label = "FP1-F7")
ax2.set_xticks([0, 60000], ["10:30", "15:30"])
ax2.set_ylim([-300, 490])
ax2.legend(loc = "upper right")

# ax3.set_title("Annotations of the artifacts")
# ax3.plot((np.abs(jo) > 200-200/1000)[40600:50000], color = "r", label = "EEG >= 199.8")
# ax3.plot(tar[40600:50000], label = "annotation")
# ax3.set_xticks([0, 9400], ["13:53", "14:40"])
# ax3.legend()

ax4.set_title("The EEG recording")
ax4.plot(jo[40600:50000], label = "FP1-F7")
ax4.set_ylim([-210, 350])
ax4.set_xticks([0, 9400], ["13:53", "14:40"])
ax4.legend()

plt.show()
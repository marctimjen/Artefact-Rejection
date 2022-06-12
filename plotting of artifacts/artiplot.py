import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

ind = np.memmap("C:/Users/Marc/Desktop/data_mclass/val_model_data" + "/model_input.dat", dtype='float32', mode='r', shape=(28, 22, 549200))



delta = 2 # add two seconds before and after the artefact.

# for 1 chew:

df_chew = pd.DataFrame({'eeg':ind[6][0]})

chew_st, chew_en = 958.9474,959.3461

(chew_art_start, chew_art_end) = (math.floor((chew_st)*200), math.ceil((chew_en)*200))
(chew_start, chew_end) = (math.floor((chew_st-delta)*200), math.ceil((chew_en+delta)*200))
chew_one = df_chew["eeg"][chew_start:chew_end]
chew_two = df_chew["eeg"][chew_art_start:chew_art_end]

# for 2 musc:

df_m = pd.DataFrame({'eeg':ind[1][1]}) # using channel F7-T3 data from 00000254_s007_t000

m_delta = 1 # add one seconds before and after the artefact.
musc_st, musc_en = 306.0403,310.8477

(musc_art_start, musc_art_end) = (math.floor((musc_st)*200), math.ceil((musc_en)*200))
(musc_start, musc_end) = (math.floor((musc_st-m_delta)*200), math.ceil((musc_en+m_delta)*200))
musc_one = df_m["eeg"][musc_start:musc_end]
musc_two = df_m["eeg"][musc_art_start:musc_art_end]


# for 2 eyem_musc:

em = pd.DataFrame({'eeg':ind[1][3]}) # using channel FP1-F7 from 00000254_s007_t000

em_st, em_en = 246000/200+2, 1244

(em_art_start, em_art_end) = (math.floor((em_st)*200), math.ceil((em_en)*200))
(em_start, em_end) = (math.floor((em_st-delta)*200), math.ceil((em_en+delta)*200))
em_one = em["eeg"][em_start:em_end]
em_two = em["eeg"][em_art_start:em_art_end]


# for 3 eyem
df_e = pd.DataFrame({'eeg':ind[1][0]}) # using channel FP1-F7 from 00000254_s007_t000
eye_st, eye_en = 230, 233

(eye_art_start, eye_art_end) = ((eye_st)*200, (eye_en)*200)
(eye_start, eye_end) = ((eye_st-delta)*200, (eye_en+delta)*200)
eye_one = df_e["eeg"][eye_start:eye_end]
eye_two = df_e["eeg"][eye_art_start:eye_art_end]

# for 5 elec
df_elec = pd.DataFrame({'eeg':ind[1][1]}) # using data from 00000254_s007_t000

elec_st, elec_en = 126.0154, 132.0083

(elec_art_start, elec_art_end) = (math.floor((elec_st)*200), math.ceil((elec_en)*200))
(elec_start, elec_end) = (math.floor((elec_st-delta)*200), math.ceil((elec_en+delta)*200))
elec_one = df_elec["eeg"][elec_start:elec_end]
elec_two = df_elec["eeg"][elec_art_start:elec_art_end]

# for elpp 6
df = pd.DataFrame({'eeg':ind[1][0]})
elpp_st, elpp_en = 653.9959,655.0095

(elpp_art_start, elpp_art_end) = (math.floor((elpp_st)*200), math.ceil((elpp_en)*200))
(elpp_start, elpp_end) = (math.floor((elpp_st-delta)*200), math.ceil((elpp_en+delta)*200))
elpp_one = df["eeg"][elpp_start:elpp_end]
elpp_two = df["eeg"][elpp_art_start:elpp_art_end]




fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2) #


# chew
ax1.set_title("Example of chew")
ax1.plot(chew_one, label = "FP1-F7")
ax1.plot(chew_two, color = "r", label = "")
ax1.set_xticks([191400, 192200], ["15:57", "16:01"])
ax1.set_ylabel('µV')
ax1.legend()

# musc
ax2.set_title("Example of musc")
ax2.plot(musc_one, label = "F7-T3")
ax2.plot(musc_two, color = "r", label = "")
ax2.set_xticks([61000, 62369], ["5:05", "5:12"])
ax2.set_ylabel('µV')
ax2.legend()


# eyem
ax3.set_title("Example of eyem")
ax3.plot(eye_one, label = "FP1-F7")
ax3.plot(eye_two, color = "r", label = "")
ax3.set_xticks([45600, 47000], ["3:48", "3:55"])
ax3.set_ylabel('µV')
ax3.legend()


# em
ax4.set_title("Example of eyem_musc")
ax4.plot(em_one, label = "FP1-F7")
ax4.plot(em_two, color = "r", label = "")
ax4.set_xticks([246000, 249000], ["20:30", "20:45"])
ax4.set_ylabel('µV')
ax4.legend()

# elec
ax5.set_title("Example of elec")
ax5.plot(elec_one, label = "F7-T3")
ax5.plot(elec_two, color = "r", label = "")
ax5.set_xticks([24803, 26801], ["2:04", "2:14"])
ax5.set_ylabel('µV')
ax5.legend()

# elpp
ax6.set_title("Example of elpp")
ax6.plot(elpp_one, label = "FP1-F7")
ax6.plot(elpp_two, color = "r", label = "")
ax6.set_xticks([130400, 131400], ["10:52", "10:57"])
ax6.set_ylabel('µV')
ax6.legend()

plt.show()
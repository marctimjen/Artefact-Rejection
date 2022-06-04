import matplotlib as plt
import mne

file_dir = "C:/Users/Bruger/Documents/GitHub/edf/01_tcp_ar/002/00000254/s005_2010_11_15/00000254_s005_t000.edf"

file = mne.io.read_raw_edf(file_dir)


print(file)
print(type(file))


# sd < 50
# range +- 50 mikrowolt
# størrelses orden er 10 eller 10^-5

# brug helst mikrowolt



#
#vi vil gerne have l_freq omkring 0.1 Hz
# og h_freq omkring 100 Hz


file = mne.filter.filter_data(file, l_freq = 0.1, h_freq = 100)

file.plot(block=True)



import matplotlib as plt
import mne

file_dir = "C:/Users/Bruger/Documents/GitHub/edf/01_tcp_ar/002/00000254/s005_2010_11_15/00000254_s005_t000.edf"

file = mne.io.read_raw_edf(file_dir, preload=True)


print(file)
print(type(file))


# sd < 50
# range +- 50 mikrowolt
# størrelses orden er 10 eller 10^-5

# brug helst mikrowolt



#
#vi vil gerne have l_freq omkring 0.1 Hz
# og h_freq omkring 100 Hz


file


data = file.filter(0.1, 100)


# %matplotlib
data.plot(block=True)


pd = data.to_data_frame()
pd



pd["EEG FP1-REF"]


plt.pyplot.plot(pd["EEG FP1-REF"][36000:])


pd["EEG T3-REF"][454*250:464*250]


plt.pyplot.plot(pd["EEG T3-REF"][454*250:464*250])

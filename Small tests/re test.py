import re


# Finding the classes for the elec class:

examples = ["musc_elec", "eyem_musc", "musc", "elec", "eyem", "chew",
            "eyem_elec", "eyem_chew", "shiv", "chew_musc", "elpp",
            "chew_elec", "eyem_shiv", "shiv_elec"]

ele = []
mus = []
eye = []
art = []

for i in examples:
    elec_flag = re.search("elec", i) or re.search("elpp", i)

    # test if string in elec class

    if elec_flag:
        ele.append(i)
    elif i == "musc":
        mus.append(i)
    elif i == "eyem":
        eye.append(i)
    else:
        art.append(i)


print(ele)
print()
print(mus)
print()
print(eye)
print()
print(art)

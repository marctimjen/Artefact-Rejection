import neptune.new as neptune
import os
import matplotlib.pyplot as plt

token = os.getenv('Neptune_api')
run = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-267"
) # adam network - AR1-259

# AR1-263

adam_rate = run['network_ADAM/learning_rate'].fetch_values()
adam_tp = run['network_ADAM/matrix/val_tp_pr_file'].fetch_values()
adam_fp = run['network_ADAM/matrix/val_fp_pr_file'].fetch_values()
adam_tn = run['network_ADAM/matrix/val_tn_pr_file'].fetch_values()
adam_fn = run['network_ADAM/matrix/val_fn_pr_file'].fetch_values()

adam_acc = run['network_ADAM/val_acc_pr_file'].fetch_values() #62
adam_loss = run['network_ADAM/validation_loss_pr_file'].fetch_values() #62


run2 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-266"
) # sgd network - AR1-258

# AR1-262

sgd_rate = run2['network_SGD/learning_rate'].fetch_values()
sgd_tp = run2['network_SGD/matrix/val_tp_pr_file'].fetch_values()
sgd_fp = run2['network_SGD/matrix/val_fp_pr_file'].fetch_values()
sgd_tn = run2['network_SGD/matrix/val_tn_pr_file'].fetch_values()
sgd_fn = run2['network_SGD/matrix/val_fn_pr_file'].fetch_values()

sgd_acc = run2['network_SGD/val_acc_pr_file'].fetch_values() #62
sgd_loss = run2['network_SGD/validation_loss_pr_file'].fetch_values() #62

run.stop()
run2.stop()

print(len(adam_rate["value"]))

fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
fig.suptitle('Linear increasing learning_rates')
ax1.set_title('ADAM optimizor with base_lr=0.0001, max_lr=0.5')



ax1.plot(adam_rate["value"], adam_tp["value"], label = "tp", color = "blue")
ax1.plot(adam_rate["value"], adam_fp["value"], label = "fp", color = "red")
ax1.plot(adam_rate["value"], adam_tn["value"], label = "tn", color = "green")
ax1.plot(adam_rate["value"], adam_fn["value"], label = "fn", color = "black")

ax1.plot(adam_rate["value"], adam_acc["value"], label = "acc", color = "orange")

ax1.set_xlabel('learning_rate')
ax1.set_ylabel('accuarcy')
ax1.legend()

ax2.set_title('ADAM optimizor loss duing training')
ax2.plot([0]+ [i for i in adam_rate["value"]], adam_loss["value"])
ax2.set_xlabel('learning_rate')
ax2.set_ylabel('loss')


ax3.set_title('SGD optimizor with base_lr=0.001, max_lr=9')

ax3.plot(sgd_rate["value"], sgd_acc["value"], label = "acc", color = "orange")

ax3.plot(sgd_rate["value"], sgd_tp["value"], label = "tp", color = "blue")
ax3.plot(sgd_rate["value"], sgd_fp["value"], label = "fp", color = "red")
ax3.plot(sgd_rate["value"], sgd_tn["value"], label = "tn", color = "green")
ax3.plot(sgd_rate["value"], sgd_fn["value"], label = "fn", color = "black")

ax3.set_xlabel('learning_rate')
ax3.set_ylabel('accuarcy')
ax3.legend()

ax4.set_title('SGD optimizor loss duing training')
ax4.plot([0]+ [i for i in sgd_rate["value"]], sgd_loss["value"])
ax4.set_xlabel('learning_rate')
ax4.set_ylabel('loss')

fig.tight_layout(pad=2.0)
plt.show()



# token = os.getenv('Neptune_api')
# run = neptune.init(
#     project="NTLAB/artifact-rej-scalp",
#     api_token=token,
#     run="AR1-242"
# ) # 58
#
# rate_58 = run['network_ADAM/learning_rate'].fetch_values()
# accuacy_58 = run['network_ADAM/validation_loss_pr_file'].fetch_values() #58
#
# run2 = neptune.init(
#     project="NTLAB/artifact-rej-scalp",
#     api_token=token,
#     run="AR1-243"
# ) # 62
#
# rate_62 = run2['network_SGD/learning_rate'].fetch_values()
#
# accuacy_62 = run2['network_SGD/validation_loss_pr_file'].fetch_values() #62
#
# run.stop()
# run2.stop()
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('Linear increasing learning_rates')
# ax1.set_title('base_lr=0.0001, max_lr=0.5')
# ax1.plot([0] + [i for i in rate_58["value"]], accuacy_58["value"])
# ax1.set_xlabel('learning_rate')
# ax1.set_ylabel('accuarcy')
# ax2.set_title('base_lr=0.001, max_lr=9')
# ax2.plot([0] + [i for i in rate_62["value"]], accuacy_62["value"])
# ax2.set_xlabel('learning_rate')
# ax2.set_ylabel('accuarcy')
# fig.tight_layout(pad=2.0)
# plt.show()

# https://www.kite.com/python/answers/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python
# https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

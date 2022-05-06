import neptune.new as neptune
import os
import matplotlib.pyplot as plt

token = os.getenv('Neptune_api')
run = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-245"
) # adam network

adam_rate = run['network_ADAM/learning_rate'].fetch_values()
adam_tp = run['network_ADAM/matrix/val_tp_pr_file'].fetch_values()
adam_fp = run['network_ADAM/matrix/val_fp_pr_file'].fetch_values()
adam_tn = run['network_ADAM/matrix/val_tn_pr_file'].fetch_values()
adam_fn = run['network_ADAM/matrix/val_fn_pr_file'].fetch_values()

run2 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-244"
) # sgd network

run2.sync()

sgd_rate = run2['network_SGD/learning_rate'].fetch_values()
# sgd_tp = run['network_SGD/matrix/val_tp_pr_file'].fetch_values()
# sgd_fp = run['network_SGD/matrix/val_fp_pr_file'].fetch_values()
# sgd_tn = run['network_SGD/matrix/val_tn_pr_file'].fetch_values()
# sgd_fn = run['network_SGD/matrix/val_fn_pr_file'].fetch_values()
accuacy_62 = run2['network_SGD/val_acc_pr_file'].fetch_values() #62
run.stop()
run2.stop()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Linear increasing learning_rates')
ax1.set_title('ADAM optimizor with base_lr=0.0001, max_lr=0.5')
ax1.plot(adam_rate["value"], adam_tp["value"])
ax1.plot(adam_rate["value"], adam_fp["value"])
ax1.plot(adam_rate["value"], adam_tn["value"])
ax1.plot(adam_rate["value"], adam_fn["value"])
ax1.set_xlabel('learning_rate')
ax1.set_ylabel('accuarcy')
ax2.set_title('SGD optimizor with base_lr=0.001, max_lr=9')
ax2.plot(sgd_rate["value"], accuacy_62["value"])
# ax2.plot(sgd_rate["value"], sgd_tp["value"])
# ax2.plot(sgd_rate["value"], sgd_fp["value"])
# ax2.plot(sgd_rate["value"], sgd_tn["value"])
# ax2.plot(sgd_rate["value"], sgd_fn["value"])
ax2.set_xlabel('learning_rate')
ax2.set_ylabel('accuarcy')
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

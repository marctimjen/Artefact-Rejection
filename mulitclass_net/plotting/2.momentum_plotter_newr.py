
import neptune.new as neptune
import os
import matplotlib.pyplot as plt

token = os.getenv('Neptune_api')
run1 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-629"
) # SGD 1

sgd1_rate = run1['network_SGD/learning_rate'].fetch_values()
sgd1_mom = run1['network_SGD/momentum'].fetch_values()
sgd1_tp = run1['network_SGD/matrix/val_noart_tp_pr_file'].fetch_values()
sgd1_tn = run1['network_SGD/matrix/val_noart_tn_pr_file'].fetch_values()

sgd1_acc = run1['network_SGD/val_acc_pr_file'].fetch_values()
sgd1_loss = run1['network_SGD/validation_loss_pr_file'].fetch_values()
sgd1_smloss = run1['network_SGD/smooth_val_loss_pr_file'].fetch_values()

token = os.getenv('Neptune_api')
run2 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-630"
) # SGD 2


sgd2_rate = run2['network_SGD/learning_rate'].fetch_values()
sgd2_mom = run2['network_SGD/momentum'].fetch_values()
sgd2_tp = run2['network_SGD/matrix/val_noart_tp_pr_file'].fetch_values()
sgd2_tn = run2['network_SGD/matrix/val_noart_tn_pr_file'].fetch_values()

sgd2_acc = run2['network_SGD/val_acc_pr_file'].fetch_values()
sgd2_loss = run2['network_SGD/validation_loss_pr_file'].fetch_values()
sgd2_smloss = run2['network_SGD/smooth_val_loss_pr_file'].fetch_values()

token = os.getenv('Neptune_api')
run3 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-628"
) # SGD 3


sgd3_rate = run3['network_SGD/learning_rate'].fetch_values()
sgd3_mom = run3['network_SGD/momentum'].fetch_values()
sgd3_tp = run3['network_SGD/matrix/val_noart_tp_pr_file'].fetch_values()
sgd3_tn = run3['network_SGD/matrix/val_noart_tn_pr_file'].fetch_values()

sgd3_acc = run3['network_SGD/val_acc_pr_file'].fetch_values()
sgd3_loss = run3['network_SGD/validation_loss_pr_file'].fetch_values()
sgd3_smloss = run3['network_SGD/smooth_val_loss_pr_file'].fetch_values()

run1.stop()
run2.stop()
run3.stop()


loss_y_range = [1.1, 1.5]
momentum_y_range = [0.55, 1]
acc_y_range = [-0.1, 1.1]
lr_range = [0.01, 1.5]

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
fig.suptitle("Different momentum tests")

l1, = ax1.plot(sgd1_rate["value"])
ax1.set_ylim(lr_range)
t1 = ax1.twinx()
t1.set_ylim(momentum_y_range)
l2, = t1.plot(sgd1_mom["value"], color = "orange", label = "mom.")
ax1.set_ylabel('learning rate')
ax1.yaxis.label.set_color('blue')

ax4.plot(sgd1_tp["value"], label = "tp", color = "blue")
ax4.plot(sgd1_tn["value"], label = "tn", color = "green")
ax4.plot(sgd1_acc["value"], label = "acc", color = "orange")
ax4.set_ylim(acc_y_range)
ax4.set_ylabel('accuarcy')
ax4.legend(loc = "upper left")

ax7.plot(sgd1_loss["value"], label = "loss")
ax7.plot(sgd1_smloss["value"], label = "sm_loss")
ax7.set_ylim(loss_y_range)
ax7.set_ylabel('loss')
ax7.legend(loc = "upper left")


l3, = ax2.plot(sgd2_rate["value"])
ax2.set_ylim(lr_range)
t2 = ax2.twinx()
t2.set_ylim(momentum_y_range)
l4, = t2.plot(sgd2_mom["value"], color = "orange", label = "mom.")
t2.set_ylabel('momentum')
ax2.set_ylabel('Learning rate')
ax2.yaxis.label.set_color('blue')
t2.yaxis.label.set_color('orange')

ax5.plot(sgd2_tp["value"], label = "tp", color = "blue")
ax5.plot(sgd2_tn["value"], label = "tn", color = "green")
ax5.plot(sgd2_acc["value"], label = "acc", color = "orange")
ax5.set_ylim(acc_y_range)
ax5.set_ylabel('accuarcy')
ax5.legend(loc = "upper left")

ax8.plot(sgd2_loss["value"], label = "loss")
ax8.plot(sgd2_smloss["value"], label = "sm_loss")
ax8.set_ylim(loss_y_range)
ax8.set_ylabel('loss')
ax8.legend(loc = "upper left")


l5, = ax3.plot(sgd3_rate["value"])
ax3.set_ylim(lr_range)
t3 = ax3.twinx()
t3.set_ylim(momentum_y_range)
l6, = t3.plot(sgd3_mom["value"], color = "orange", label = "mom.")
t3.set_ylabel('momentum')
ax3.set_ylabel('learning rate')
ax3.yaxis.label.set_color('blue')
t3.yaxis.label.set_color('orange')

ax6.plot(sgd3_tp["value"], label = "tp", color = "blue")
ax6.plot(sgd3_tn["value"], label = "tn", color = "green")
ax6.plot(sgd3_acc["value"], label = "acc", color = "orange")
ax6.set_ylim(acc_y_range)
ax6.set_ylabel('accuarcy')
ax6.legend(loc = "upper left")

ax9.plot(sgd3_loss["value"], label = "loss")
ax9.plot(sgd3_smloss["value"], label = "sm_loss")
ax9.set_ylim(loss_y_range)
ax9.set_ylabel('loss')
ax9.legend(loc = "upper left")


fig.tight_layout(pad=2.0)

plt.show()

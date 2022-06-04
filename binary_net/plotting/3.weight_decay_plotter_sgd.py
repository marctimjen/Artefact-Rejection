
import neptune.new as neptune
import os
import matplotlib.pyplot as plt

# sgd 411, 412, 413
# adam 414, 415, 416

token = os.getenv('Neptune_api')
run1 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-518" # 428 522
) # SGD 1

sgd1_rate = run1['network_SGD/learning_rate'].fetch_values()
sgd1_mom = run1['network_SGD/momentum'].fetch_values()
sgd1_tp = run1['network_SGD/matrix/val_tp_pr_file'].fetch_values()
sgd1_fp = run1['network_SGD/matrix/val_fp_pr_file'].fetch_values()
sgd1_tn = run1['network_SGD/matrix/val_tn_pr_file'].fetch_values()
sgd1_fn = run1['network_SGD/matrix/val_fn_pr_file'].fetch_values()

sgd1_acc = run1['network_SGD/val_acc_pr_file'].fetch_values()
sgd1_loss = run1['network_SGD/validation_loss_pr_file'].fetch_values()
sgd1_smloss = run1['network_SGD/smooth_val_loss_pr_file'].fetch_values()

token = os.getenv('Neptune_api')
run2 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-523" # 437
) # SGD 2


sgd2_rate = run2['network_SGD/learning_rate'].fetch_values()
sgd2_mom = run2['network_SGD/momentum'].fetch_values()
sgd2_tp = run2['network_SGD/matrix/val_tp_pr_file'].fetch_values()
sgd2_fp = run2['network_SGD/matrix/val_fp_pr_file'].fetch_values()
sgd2_tn = run2['network_SGD/matrix/val_tn_pr_file'].fetch_values()
sgd2_fn = run2['network_SGD/matrix/val_fn_pr_file'].fetch_values()

sgd2_acc = run2['network_SGD/val_acc_pr_file'].fetch_values()
sgd2_loss = run2['network_SGD/validation_loss_pr_file'].fetch_values()
sgd2_smloss = run2['network_SGD/smooth_val_loss_pr_file'].fetch_values()

token = os.getenv('Neptune_api')
run3 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-524" # 438
) # SGD 3




sgd3_rate = run3['network_SGD/learning_rate'].fetch_values()
sgd3_mom = run3['network_SGD/momentum'].fetch_values()
sgd3_tp = run3['network_SGD/matrix/val_tp_pr_file'].fetch_values()
sgd3_fp = run3['network_SGD/matrix/val_fp_pr_file'].fetch_values()
sgd3_tn = run3['network_SGD/matrix/val_tn_pr_file'].fetch_values()
sgd3_fn = run3['network_SGD/matrix/val_fn_pr_file'].fetch_values()

sgd3_acc = run3['network_SGD/val_acc_pr_file'].fetch_values()
sgd3_loss = run3['network_SGD/validation_loss_pr_file'].fetch_values()
sgd3_smloss = run3['network_SGD/smooth_val_loss_pr_file'].fetch_values()

run1.stop()
run2.stop()
run3.stop()


loss_y_range = [0.55, 0.85]
momentum_y_range = [0.55, 1]
acc_y_range = [-0.1, 1.1]
lr_range = [0, 0.2]

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)


l1, = ax1.plot(sgd1_rate["value"])
ax1.set_ylim(lr_range)
t1 = ax1.twinx()
t1.set_ylim(momentum_y_range)
l2, = t1.plot(sgd1_mom["value"], color = "orange")

ax4.plot(sgd1_tp["value"], label = "tp", color = "blue")
ax4.plot(sgd1_fp["value"], label = "fp", color = "gray")
ax4.plot(sgd1_tn["value"], label = "tn", color = "green")
ax4.plot(sgd1_fn["value"], label = "fn", color = "black")
ax4.plot(sgd1_acc["value"], label = "acc", color = "orange")
ax4.set_ylim(acc_y_range)

ax7.plot(sgd1_loss["value"])
ax7.plot(sgd1_smloss["value"])
ax7.set_ylim(loss_y_range)


l3, = ax2.plot(sgd2_rate["value"])
ax2.set_ylim(lr_range)
t2 = ax2.twinx()
t2.set_ylim(momentum_y_range)
l4, = t2.plot(sgd2_mom["value"], color = "orange")


ax5.plot(sgd2_tp["value"], label = "tp", color = "blue")
ax5.plot(sgd2_fp["value"], label = "fp", color = "gray")
ax5.plot(sgd2_tn["value"], label = "tn", color = "green")
ax5.plot(sgd2_fn["value"], label = "fn", color = "black")
ax5.plot(sgd2_acc["value"], label = "acc", color = "orange")
ax5.set_ylim(acc_y_range)
ax5.legend()

ax8.plot(sgd2_loss["value"])
ax8.plot(sgd2_smloss["value"])
ax8.set_ylim(loss_y_range)

l5, = ax3.plot(sgd3_rate["value"])
ax3.set_ylim(lr_range)
t3 = ax3.twinx()
t3.set_ylim(momentum_y_range)
l6, = t3.plot(sgd3_mom["value"], color = "orange")

ax6.plot(sgd3_tp["value"], label = "ax6tp", color = "blue")
ax6.plot(sgd3_fp["value"], label = "fp", color = "gray")
ax6.plot(sgd3_tn["value"], label = "tn", color = "green")
ax6.plot(sgd3_fn["value"], label = "fn", color = "black")
ax6.plot(sgd3_acc["value"], label = "acc", color = "orange")
ax6.set_ylim(acc_y_range)
ax6.legend()

ax9.plot(sgd3_loss["value"])
ax9.plot(sgd3_smloss["value"])
ax9.set_ylim(loss_y_range)

fig.tight_layout(pad=2.0)
plt.show()
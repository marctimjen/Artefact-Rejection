
import neptune.new as neptune
import os
import matplotlib.pyplot as plt


token = os.getenv('Neptune_api')
run1 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-633" # 596
)

sgd1_rate = run1['network_SGD/learning_rate'].fetch_values()
sgd1_weight = run1['network_SGD/parameters/optimizer_weight_decay'].fetch()
sgd1_tp = run1['network_SGD/matrix/val_noart_tp_pr_file'].fetch_values()
sgd1_tn = run1['network_SGD/matrix/val_noart_tn_pr_file'].fetch_values()

sgd1_acc = run1['network_SGD/val_acc_pr_file'].fetch_values()
sgd1_loss = run1['network_SGD/validation_loss_pr_file'].fetch_values()
sgd1_smloss = run1['network_SGD/smooth_val_loss_pr_file'].fetch_values()

token = os.getenv('Neptune_api')
run2 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-632" # 595
)


sgd2_rate = run2['network_SGD/learning_rate'].fetch_values()
sgd2_weight = run2['network_SGD/parameters/optimizer_weight_decay'].fetch()
sgd2_tp = run2['network_SGD/matrix/val_noart_tp_pr_file'].fetch_values()
sgd2_tn = run2['network_SGD/matrix/val_noart_tn_pr_file'].fetch_values()

sgd2_acc = run2['network_SGD/val_acc_pr_file'].fetch_values()
sgd2_loss = run2['network_SGD/validation_loss_pr_file'].fetch_values()
sgd2_smloss = run2['network_SGD/smooth_val_loss_pr_file'].fetch_values()

token = os.getenv('Neptune_api')
run3 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-629" # 592
)




sgd3_rate = run3['network_SGD/learning_rate'].fetch_values()
try:
    sgd3_weight = run3['network_SGD/parameters/optimizer_weight_decay'].fetch()
except:
    sgd3_weight = 0
sgd3_tp = run3['network_SGD/matrix/val_noart_tp_pr_file'].fetch_values()
sgd3_tn = run3['network_SGD/matrix/val_noart_tn_pr_file'].fetch_values()

sgd3_acc = run3['network_SGD/val_acc_pr_file'].fetch_values()
sgd3_loss = run3['network_SGD/validation_loss_pr_file'].fetch_values()
sgd3_smloss = run3['network_SGD/smooth_val_loss_pr_file'].fetch_values()

run1.stop()
run2.stop()
run3.stop()


loss_y_range = [1, 1.5]
momentum_y_range = [0.55, 1]
acc_y_range = [-0.1, 1.1]
lr_range = [1, 1.5]

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)

ax1.set_title(f'SGD optimizor with weight_decay of {sgd1_weight}')
ax1.plot(sgd1_rate["value"])
ax1.set_ylabel('Learning rate')
ax1.set_ylim(lr_range)


ax4.plot(sgd1_tp["value"], label = "tp", color = "blue")
ax4.plot(sgd1_tn["value"], label = "tn", color = "green")
ax4.plot(sgd1_acc["value"], label = "acc", color = "orange")
ax4.set_ylabel('Pct.')
ax4.set_ylim(acc_y_range)
ax4.legend()

ax7.plot(sgd1_loss["value"], label = "loss")
ax7.plot(sgd1_smloss["value"], label = "sm_loss")
ax7.set_ylabel('Loss')
ax7.set_ylim(loss_y_range)
ax7.legend()

ax2.set_title(f'SGD optimizor with weight_decay of {sgd2_weight}')
ax2.plot(sgd2_rate["value"])
ax2.set_ylabel('Learning rate')
ax2.set_ylim(lr_range)


ax5.plot(sgd2_tp["value"], label = "tp", color = "blue")
ax5.plot(sgd2_tn["value"], label = "tn", color = "green")
ax5.plot(sgd2_acc["value"], label = "acc", color = "orange")
ax5.set_ylabel('Pct.')
ax5.set_ylim(acc_y_range)
ax5.legend()


ax8.plot(sgd2_loss["value"], label = "loss")
ax8.plot(sgd2_smloss["value"], label = "sm_loss")
ax8.set_ylabel('Loss')
ax8.set_ylim(loss_y_range)
ax8.legend()

ax3.set_title(f'SGD optimizor with weight_decay of {sgd3_weight}')
ax3.plot(sgd3_rate["value"])
ax3.set_ylabel('Learning rate')
ax3.set_ylim(lr_range)

ax6.plot(sgd3_tp["value"], label = "tp", color = "blue")
ax6.plot(sgd3_tn["value"], label = "tn", color = "green")
ax6.plot(sgd3_acc["value"], label = "acc", color = "orange")
ax6.set_ylabel('Pct.')
ax6.set_ylim(acc_y_range)
ax6.legend()

ax9.plot(sgd3_loss["value"], label = "loss")
ax9.plot(sgd3_smloss["value"], label = "sm_loss")
ax9.set_ylabel('Loss')
ax9.set_ylim(loss_y_range)
ax9.legend()

fig.tight_layout(pad=2.0)
plt.show()

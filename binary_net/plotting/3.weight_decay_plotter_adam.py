
import neptune.new as neptune
import os
import matplotlib.pyplot as plt

# sgd 411, 412, 413
# adam 414, 415, 416

token = os.getenv('Neptune_api')
run1 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-439"
) # adam 1

adam1_rate = run1['network_ADAM/learning_rate'].fetch_values()
adam1_weight = run1['network_ADAM/parameters/optimizer_weight_decay'].fetch()
adam1_tp = run1['network_ADAM/matrix/val_tp_pr_file'].fetch_values()
adam1_fp = run1['network_ADAM/matrix/val_fp_pr_file'].fetch_values()
adam1_tn = run1['network_ADAM/matrix/val_tn_pr_file'].fetch_values()
adam1_fn = run1['network_ADAM/matrix/val_fn_pr_file'].fetch_values()

adam1_acc = run1['network_ADAM/val_acc_pr_file'].fetch_values()
adam1_loss = run1['network_ADAM/validation_loss_pr_file'].fetch_values()
adam1_smloss = run1['network_ADAM/smooth_val_loss_pr_file'].fetch_values()

token = os.getenv('Neptune_api')
run2 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-440"
) # adam 2


adam2_rate = run2['network_ADAM/learning_rate'].fetch_values()
adam2_weight = run2['network_ADAM/parameters/optimizer_weight_decay'].fetch()
adam2_tp = run2['network_ADAM/matrix/val_tp_pr_file'].fetch_values()
adam2_fp = run2['network_ADAM/matrix/val_fp_pr_file'].fetch_values()
adam2_tn = run2['network_ADAM/matrix/val_tn_pr_file'].fetch_values()
adam2_fn = run2['network_ADAM/matrix/val_fn_pr_file'].fetch_values()

adam2_acc = run2['network_ADAM/val_acc_pr_file'].fetch_values()
adam2_loss = run2['network_ADAM/validation_loss_pr_file'].fetch_values()
adam2_smloss = run2['network_ADAM/smooth_val_loss_pr_file'].fetch_values()

token = os.getenv('Neptune_api')
run3 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run="AR1-441"
) # adam 3




adam3_rate = run3['network_ADAM/learning_rate'].fetch_values()
adam3_weight = run3['network_ADAM/parameters/optimizer_weight_decay'].fetch()
adam3_tp = run3['network_ADAM/matrix/val_tp_pr_file'].fetch_values()
adam3_fp = run3['network_ADAM/matrix/val_fp_pr_file'].fetch_values()
adam3_tn = run3['network_ADAM/matrix/val_tn_pr_file'].fetch_values()
adam3_fn = run3['network_ADAM/matrix/val_fn_pr_file'].fetch_values()

adam3_acc = run3['network_ADAM/val_acc_pr_file'].fetch_values()
adam3_loss = run3['network_ADAM/validation_loss_pr_file'].fetch_values()
adam3_smloss = run3['network_ADAM/smooth_val_loss_pr_file'].fetch_values()

run1.stop()
run2.stop()
run3.stop()


loss_y_range = [0.55, 0.85]
momentum_y_range = [0.55, 1]
acc_y_range = [-0.1, 1.1]
lr_range = [-0.0001, 0.02]

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)

ax1.set_title(f'ADAM optimizor with weight_decay of {adam1_weight}')
ax1.plot(adam1_rate["value"])
ax1.set_ylabel('Learning rate')
ax1.set_ylim(lr_range)


ax4.plot(adam1_tp["value"], label = "tp", color = "blue")
ax4.plot(adam1_fp["value"], label = "fp", color = "gray")
ax4.plot(adam1_tn["value"], label = "tn", color = "green")
ax4.plot(adam1_fn["value"], label = "fn", color = "black")
ax4.plot(adam1_acc["value"], label = "acc", color = "orange")
ax4.set_ylabel('Pct.')
ax4.set_ylim(acc_y_range)

ax7.plot(adam1_loss["value"])
ax7.plot(adam1_smloss["value"])
ax7.set_ylabel('Loss')
ax7.set_ylim(loss_y_range)

ax2.set_title(f'ADAM optimizor with weight_decay of {adam2_weight}')
ax2.plot(adam2_rate["value"])
ax2.set_ylabel('Learning rate')
ax2.set_ylim(lr_range)


ax5.plot(adam2_tp["value"], label = "tp", color = "blue")
ax5.plot(adam2_fp["value"], label = "fp", color = "gray")
ax5.plot(adam2_tn["value"], label = "tn", color = "green")
ax5.plot(adam2_fn["value"], label = "fn", color = "black")
ax5.plot(adam2_acc["value"], label = "acc", color = "orange")
ax5.set_ylabel('Pct.')
ax5.set_ylim(acc_y_range)

ax8.plot(adam2_loss["value"])
ax8.plot(adam2_smloss["value"])
ax8.set_ylabel('Loss')
ax8.set_ylim(loss_y_range)

ax3.set_title(f'ADAM optimizor with weight_decay of {adam3_weight}')
ax3.plot(adam3_rate["value"])
ax3.set_ylabel('Learning rate')
ax3.set_ylim(lr_range)



ax6.plot(adam3_tp["value"], label = "tp", color = "blue")
ax6.plot(adam3_fp["value"], label = "fp", color = "gray")
ax6.plot(adam3_tn["value"], label = "tn", color = "green")
ax6.plot(adam3_fn["value"], label = "fn", color = "black")
ax6.plot(adam3_acc["value"], label = "acc", color = "orange")
ax6.set_ylabel('Pct.')
ax6.set_ylim(acc_y_range)

ax9.plot(adam3_loss["value"])
ax9.plot(adam3_smloss["value"])
ax9.set_ylabel('Loss')
ax9.set_ylim(loss_y_range)

fig.tight_layout(pad=2.0)
plt.show()

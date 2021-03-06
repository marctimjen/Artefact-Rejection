import neptune.new as neptune
import os
import matplotlib.pyplot as plt



exp_mode = True

if exp_mode:

    adam_model_run = "AR1-578"
    sgd_model_run = "AR1-588" # 577


    adam_low = 0.05
    adam_max = adam_low/10
    adam_base = round(adam_max/6, 5)

    # adam_base = 0.0038
    # adam_max = 0.0048


    # SGD_base = 0.2
    # SGD_max = 0.5
    #
    # magenta = True
    # SGD_base_m = 0.8  # magenta values
    # SGD_max_m = 1.12  # magenta values

    sgd_low = 8

    SGD_max = sgd_low/10
    SGD_base = round(SGD_max/6, 5)



    magenta = False
    SGD_base_m = 0.07  # magenta values
    SGD_max_m = 0.103  # magenta values

    magenta_adam = False
    adam_max_m = 0.003
    adam_base_m = 0.0048

    log_x_scale = True
    pos_pf_label_cont = 'upper right'

    title = 'Exponential increasing learning rates'
    sgd_title = 'SGD optimizer with lr range: 0.1 to 9'
    adam_title = 'ADAM optimizer with lr range: 0.0001 to 0.7'
else:

    adam_model_run = "AR1-576"
    sgd_model_run = "AR1-585" # 575

    adam_base = 0.007
    adam_max = 0.0138


    SGD_base = 1.2
    SGD_max = 1.3

    magenta = False
    SGD_base_m = 0.07  # magenta values
    SGD_max_m = 0.103  # magenta values

    magenta_adam = False

    log_x_scale = False
    pos_pf_label_cont = 'upper right'
    title = 'Linear increasing learning rates'
    sgd_title = 'SGD optimizer with lr range: 0.2 to 5'
    adam_title = 'ADAM optimizer with lr range: 0.01e-8 to 0.1'

token = os.getenv('Neptune_api')
run = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run=adam_model_run
) # adam network


adam_rate = run['network_ADAM/learning_rate'].fetch_values()
adam_tp = run['network_ADAM/matrix/val_noart_tp_pr_file'].fetch_values()
# adam_fp = run['network_ADAM/matrix/val_fp_pr_file'].fetch_values()
adam_tn = run['network_ADAM/matrix/val_noart_tn_pr_file'].fetch_values()
# adam_fn = run['network_ADAM/matrix/val_fn_pr_file'].fetch_values()

adam_acc = run['network_ADAM/val_acc_pr_file'].fetch_values() #62
adam_loss = run['network_ADAM/validation_loss_pr_file'].fetch_values() #62
adam_smloss = run['network_ADAM/smooth_val_loss_pr_file'].fetch_values() #62

run2 = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
    run=sgd_model_run
) # sgd network

sgd_rate = run2['network_SGD/learning_rate'].fetch_values()
sgd_tp = run2['network_SGD/matrix/val_noart_tp_pr_file'].fetch_values()
# sgd_fp = run2['network_SGD/matrix/val_fp_pr_file'].fetch_values()
sgd_tn = run2['network_SGD/matrix/val_noart_tn_pr_file'].fetch_values()
# sgd_fn = run2['network_SGD/matrix/val_fn_pr_file'].fetch_values()


sgd_acc = run2['network_SGD/val_acc_pr_file'].fetch_values() #62
sgd_loss = run2['network_SGD/validation_loss_pr_file'].fetch_values() #62
sgd_smloss = run2['network_SGD/smooth_val_loss_pr_file'].fetch_values() #62



run.stop()
run2.stop()

print(len(adam_rate["value"]))


fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
fig.suptitle(title)
ax1.set_title(adam_title)

ax1.axvline(x = adam_base, color = 'r', linestyle = "--", label = f'base_rl = {adam_base}')
ax1.axvline(x = adam_max, color = 'r', linestyle = "--", label = f'max_lr = {adam_max}')

ax1.plot(adam_rate["value"], adam_tp["value"], label = "tp", color = "blue")
# ax1.plot(adam_rate["value"], adam_fp["value"], label = "fp", color = "gray")
ax1.plot(adam_rate["value"], adam_tn["value"], label = "tn", color = "green")
# ax1.plot(adam_rate["value"], adam_fn["value"], label = "fn", color = "black")

ax1.plot(adam_rate["value"], adam_acc["value"], label = "acc", color = "orange")

if magenta_adam:
    ax1.axvline(x = adam_base_m, color = 'm', linestyle = "--", label = f'base_rl = {adam_base_m}')
    ax1.axvline(x = adam_max_m, color = 'm', linestyle = "--", label = f'max_lr = {adam_max_m}')


if log_x_scale:
    ax1.set_xscale('log')

ax1.set_xlabel('learning rate')
ax1.set_ylabel('accuarcy')
ax1.legend(loc = pos_pf_label_cont)



ax2.set_title('ADAM optimizer loss duing training')
ax2.axvline(x = adam_base, color = 'r', linestyle = "--", label = f'base_rl = {adam_base}')
ax2.axvline(x = adam_max, color = 'r', linestyle = "--", label = f'max_lr = {adam_max}')

ax2.plot([0]+ [i for i in adam_rate["value"]], adam_loss["value"], label = "loss")
ax2.plot([0]+ [i for i in adam_rate["value"]], adam_smloss["value"], label = "sm_loss")
ax2.set_xlabel('learning rate')
ax2.set_ylabel('loss')

if log_x_scale:
    ax2.set_xscale('log')

if magenta_adam:
    ax2.axvline(x = adam_base_m, color = 'm', linestyle = "--", label = f'base_rl = {adam_base_m}')
    ax2.axvline(x = adam_max_m, color = 'm', linestyle = "--", label = f'max_lr = {adam_max_m}')


ax2.legend(loc = pos_pf_label_cont)



ax3.set_title(sgd_title)

ax3.axvline(x = SGD_base, color = 'r', linestyle = "--", label = f'base_rl = {SGD_base}')
ax3.axvline(x = SGD_max, color = 'r', linestyle = "--", label = f'max_lr = {SGD_max}')

if magenta:
    ax3.axvline(x = SGD_base_m, color = 'm', linestyle = "--", label = f'base_rl = {SGD_base_m}')
    ax3.axvline(x = SGD_max_m, color = 'm', linestyle = "--", label = f'max_lr = {SGD_max_m}')

ax3.plot(sgd_rate["value"], sgd_acc["value"], label = "acc", color = "orange")

ax3.plot(sgd_rate["value"], sgd_tp["value"], label = "tp", color = "blue")
# ax3.plot(sgd_rate["value"], sgd_fp["value"], label = "fp", color = "gray")
ax3.plot(sgd_rate["value"], sgd_tn["value"], label = "tn", color = "green")
# ax3.plot(sgd_rate["value"], sgd_fn["value"], label = "fn", color = "black")

if log_x_scale:
    ax3.set_xscale('log')

ax3.set_xlabel('learning rate')
ax3.set_ylabel('accuarcy')
ax3.legend(loc = pos_pf_label_cont)



ax4.set_title('SGD optimizer loss duing training')

ax4.axvline(x = SGD_base, color = 'r', linestyle = "--", label = f'base_rl = {SGD_base}')
ax4.axvline(x = SGD_max, color = 'r', linestyle = "--", label = f'max_lr = {SGD_max}')

if magenta:
    ax4.axvline(x = SGD_base_m, color = 'm', linestyle = "--", label = f'base_rl = {SGD_base_m}')
    ax4.axvline(x = SGD_max_m, color = 'm', linestyle = "--", label = f'max_lr = {SGD_max_m}')


ax4.plot([0]+ [i for i in sgd_rate["value"]], sgd_loss["value"], label = "loss")
ax4.plot([0]+ [i for i in sgd_rate["value"]], sgd_smloss["value"], label = "sm_loss")
ax4.set_xlabel('learning rate')
ax4.set_ylabel('loss')

if log_x_scale:
    ax4.set_xscale('log')

ax4.legend(loc = pos_pf_label_cont)

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

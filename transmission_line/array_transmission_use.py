import array_transmission_lorentzian
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

n_jct = 80
lj = 15 / n_jct
cj = 1 / lj / (15 * 2 * np.pi) ** 2
cg = 1e-6
ct = 2 / (lj * n_jct * (5 * 2 * np.pi) ** 2)

# n_jct = 80
# lj = 1.9
# cj = 4.06e-5
# cg = 7e-8
# ct = 6e-7


model_base = "ground"
model_type = "c"
model = array_transmission_lorentzian.ArrayLorentzian(model_base + "_" + model_type,
                                                      show_progression=True, show_fit_window=False)

rjp = np.inf
rjs = 0
rg = np.inf
rt = np.inf

#### Slider plot for article fit
# model.init_params(n_jct, lj, cj, cg, ct)
# model.article_fit_plot()

#### Basic plot for debugging
model.init_params(n_jct, lj, cj, cg, ct)
model.init_losses(rjp, rjs, rg, rt)
model.compute_model()
model.compute_v_junc_ground()
model.compute_phi_zpf_ground()
for i, (el, w) in enumerate(zip(model._phi_zpf[:3], model._wps)):
    plt.plot(el, "+--", label=f"Mode {i+1} at {w/2/np.pi:.3f} GHz")
plt.legend()
plt.grid()
plt.xlabel("Junction number")
plt.ylabel(r"$\phi_{ZPF}$")
plt.title(r"$\phi_{ZPF}$ of each junction in the 80 junction array of an LC, predicted at 5GHz")
plt.show()

#### Basic plots for new model debugging
# model = array_transmission_lorentzian.ArrayLorentzian(n_jct, lj, cj, cg, ct, "cable_c")
# model.compute_z_in()
# model.compute_peaks()
# plt.plot(model.ws_init/2/np.pi, model.log_sqr_abs, "--", label="New")
# model = array_transmission_lorentzian.ArrayLorentzian(n_jct, lj, cj, cg, ct, "ground_c")
# model.compute_z_in()
# model.compute_peaks()
# plt.plot(model.ws_init/2/np.pi, model.log_sqr_abs, "--", label="Old")
# model = array_transmission_lorentzian.ArrayLorentzian(n_jct, lj, cj, cg, ct, "perfect")
# model.compute_z_in()
# model.compute_peaks()
# plt.plot(model.ws_init/2/np.pi, model.log_sqr_abs, "--", label="Perfect")
# plt.legend()
# plt.grid()
# plt.show()


### Sweep parameters
results_qs = []
results_fs = []
n_point_sweep = 20

r_1m_ground = [1.76392e7, 2.164235e-5, 1.863691e-3, 9.2967e8, 2.20448e8]
#rjp_1m, rjs_1m_bloc, rjs_1m, rg_1m, rt_1m = r_1m_ground

r_1m_cable = [1.340811e7, 4.2421585e-3, 1.961172e10, 1.674497e8]
rjp_1m, rjs_1m, rg_1m, rt_1m = r_1m_cable

vals_1m = [rjp_1m, rjs_1m, rg_1m, rt_1m]
labels_sweep = [r"$R_{jp}$", r"$R_{js}$", r"$R_g$", r"$R_t$"]

# variations = [np.logspace(np.log(rjp_1m)/np.log(10)-2, np.log(rjp_1m)/np.log(10)+2, n_point_sweep),
#               np.logspace(np.log(rjs_1m)/np.log(10)-2, np.log(rjs_1m)/np.log(10)+2, n_point_sweep)[::-1],
#               np.logspace(np.log(rg_1m)/np.log(10)-2, np.log(rg_1m)/np.log(10)+2, n_point_sweep),
#               np.logspace(np.log(rt_1m)/np.log(10)-2, np.log(rt_1m)/np.log(10)+2, n_point_sweep)]

# RJP_L_L = [variations[0], [np.inf], [np.inf], [np.inf]]
# RJS_L_L = [[0], variations[1], [0], [0]]
# RG_L_L = [[np.inf], [np.inf], variations[2], [np.inf]]
# RT_L_L = [[np.inf], [np.inf], [np.inf], variations[3]]
#
# for rjp_l, rjs_l, rg_l, rt_l in zip(RJP_L_L, RJS_L_L, RG_L_L, RT_L_L):
#     model.sweep_losses(rjp_l, rjs_l, rg_l, rt_l)
#     results_qs.append(model.variations_qs)
#     results_fs.append([np.array(w) / 2 / np.pi for w in model.variations_wps])
#
# results_qs, results_fs = np.array(results_qs), np.array(results_fs)
#
# normalized_variations = [v/v_1m for v, v_1m in zip(variations, vals_1m)]
# normalized_variations[1] = 1/normalized_variations[1]
# print(normalized_variations)
#
# n_mode = len(results_qs[0][0])
#
# for mode in range(n_mode):
#     fig, axs = plt.subplots(2, 1)
#     for ind, norm_var in enumerate(normalized_variations):
#         for i, r in enumerate([results_qs, results_fs]):
#             axs[i].plot(norm_var, r[ind, :, mode], label=labels_sweep[ind])
#     axs[0].set_ylabel("Q")
#     axs[1].set_ylabel("f (GHz)")
#     axs[1].set_xlabel("Normalized resistance")
#     fig.suptitle(f"Mode {mode+1}")
#     axs[0].set_title("Q vs R")
#     axs[1].set_title("f vs R")
#
#     for i in range(2):
#         axs[i].grid()
#         axs[i].legend()
#         axs[i].set_xscale('log')
#         axs[i].set_yscale('log')
#     plt.show()

# RJP_L_L = [[vals_1m[0]], [np.inf], [np.inf], [np.inf]]
# RJS_L_L = [[0], [vals_1m[1]], [0], [0]]
# RG_L_L = [[np.inf], [np.inf], [vals_1m[2]], [np.inf]]
# RT_L_L = [[np.inf], [np.inf], [np.inf], [vals_1m[3]]]
#
# for rjp_l, rjs_l, rg_l, rt_l in zip(RJP_L_L, RJS_L_L, RG_L_L, RT_L_L):
#     model.sweep_losses(rjp_l, rjs_l, rg_l, rt_l)
#     f_res, q_res, param_res = model.get_variation()
#     results_qs.append(q_res[0])
#     results_fs.append([w / 2 / np.pi for w in f_res[0]])
#
# fig, axs = plt.subplots()
# q_tot = [1 / sum([1 / results_qs[i][m] for i in range(4)]) for m in range(len(results_qs[0]))]
# axs.plot(q_tot, "+--", label="Total")
# for i, r in enumerate([results_qs]):
#     for ind in range(len(r)):
#         axs.plot(r[ind], "+--", label=labels_sweep[ind])
#     axs.grid()
#     axs.legend()
#     axs.set_yscale('log')
# axs.set_ylabel("Q")
# axs.set_xlabel("Mode number")
# plt.show()

r_1m_ground = [1.76392e7, 1.863691e-3, 9.2967e8, 2.20448e8]
r_1m_cable = [1.340811e7, 4.2421585e-3, 1.961172e10, 1.674497e8]
sweep_vals = [[np.inf, 0, np.inf, np.inf] for _ in range(4)]

model_base = "cable"
model_type = "c"
model = array_transmission_lorentzian.ArrayLorentzian(model_base + "_" + model_type,
                                                      show_progression=True, show_fit_window=False)

# for i in range(4):
#     sweep_vals[i][i] = r_1m_cable[i]
# results = [[], [], [], []]
# n_jct_sweep = [int(el) for el in np.logspace(0, 2, 20)]
# for n_jct in tqdm(n_jct_sweep):
#     lj = 15 / n_jct
#     cj = 1 / lj / (15 * 2 * np.pi) ** 2
#     cg = 1e-6
#     ct = 2 / (lj * n_jct * (5 * 2 * np.pi) ** 2)
#     model.init_params(n_jct, lj, cj, cg, ct)
#     for i, rs in enumerate(sweep_vals):
#         model.init_losses(*rs)
#         model.compute_model()
#         results[i].append(model.get_qs()[0])
# fig, ax = plt.subplots()
# for i in range(len(results)):
#     ax.plot(n_jct_sweep, results[i], "+--", label=labels_sweep[i])
# ax.grid()
# ax.set_xlabel("$N_{jct}$")
# ax.set_ylabel("Q")
# ax.set_title("Variation of the quality factor with respect to the the number of junctions.\n(Capacity to pads model)")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.legend()
# plt.show()


# for i in range(4):
#     sweep_vals[i][i] = r_1m_cable[i]
# results_q = [[], [], [], []]
# results_w = [[], [], [], []]
# lj_sweep = np.logspace(-1, 1, 20)[:-2]
# for lj in tqdm(lj_sweep):
#     n_jct = 25
#     cj = 1 / lj / (15 * 2 * np.pi) ** 2
#     cg = 1e-6
#     ct = 2 / (lj * n_jct * (5 * 2 * np.pi) ** 2)
#     model.init_params(n_jct, lj, cj, cg, ct)
#     for i, rs in enumerate(sweep_vals):
#         model.init_losses(*rs)
#         model.compute_model(lst_to_fit=[0])
#         results_q[i].append(model.get_qs()[0])
#         results_w[i].append(model.get_pulsations()[0])
# fig, ax = plt.subplots(2, 1)
# for i in range(len(results_q)):
#     ax[0].plot(lj_sweep*n_jct, results_q[i], "+--", label=labels_sweep[i])
#     ax[1].plot(lj_sweep*n_jct, [w/2/np.pi for w in results_w[i]], "+--", label=labels_sweep[i])
# ax[0].grid()
# ax[1].grid()
# ax[1].set_xlabel("$L_{array}$")
# ax[0].set_ylabel("Q")
# ax[1].set_ylabel("f (GHz)")
# fig.suptitle("Capacity to pads model")
# ax[0].set_title("Variation of the fundamental quality factor with respect to the array inductance.")
# ax[1].set_title("Variation of the fundamental frequency with respect to the array inductance.")
# ax[1].set_xscale("log")
# ax[0].set_xscale("log")
# ax[0].set_yscale("log")
# ax[0].legend()
# ax[1].legend()
# plt.show()

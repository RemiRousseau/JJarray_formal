import array_transmission_lorentzian
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

n_jct = 80
lj = 1.4215
cj = 5.388e-5
cg = 1.58e-7
ct = 2e-5


model_base = "ground"
model_type = "c"
model = array_transmission_lorentzian.ArrayLorentzian(model_base + "_" + model_type,
                                                      show_progression=True, show_fit_window=False)
model.init_params(n_jct, lj, cj, cg, ct)
# model.data_fit_plot(plot_article_data=True)
# print(model._lj)
# print(model._cj)
# print(model._cg)
# print(model._ct)

rjp = np.inf
rjs = 0
rg = np.inf
rt = np.inf

tand_j = 0
tand_g = 0
tand_t = 0


# # model.init_losses_resistor(rjp, rjs, rg, rt)
# model.init_losses_tand(tand_j, tand_g, tand_t)
# model.compute_model()
# model.plot_z_in()
# model.plot_z_in_fits([0, 1, 10])
# model.plot_mode_report()


#### Basic plot for debugging
# model.init_params(n_jct, lj, cj, cg, ct)
# model.init_losses(rjp, rjs, rg, rt)
# model.compute_model()
# print(model.get_pulsations()[0]/2/np.pi)
# model.compute_v_junc_ground()
# model.compute_phi_zpf_ground()
# for i, (el, w) in enumerate(zip(model.get_phi_zpf()[:3], model.get_pulsations())):
#     plt.plot(el, "+--", label=f"Mode {i+1} at {w/2/np.pi:.3f} GHz")
# plt.plot([0, n_jct], [model.get_perfect_grd_phi_zpf()]*2, "-", label=r"Perfect inductance $\phi_{ZPF}$")
# plt.legend()
# plt.grid()
# plt.xlabel("Junction number")
# plt.ylabel(r"$\phi_{ZPF}$")
# plt.title(r"$\phi_{ZPF}$ of each junction in the 80 junction array of an LC, predicted at 5GHz")
# plt.show()

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


####### tan dela plots

# n_jct = 25
# lj = 15/ n_jct
# cj = 1 / lj / (15 * 2 * np.pi) ** 2
# cg = 5e-7
# ct = 2 / (lj * n_jct * (5 * 2 * np.pi) ** 2)
# ground_deltas_1m_80 = [1.01393e-5, 7.26137e-5, 1.12665e-6]
# cable_deltas_1m_80 = [1.19447e-5, 6.139e-6, 1.32734e-6]
# labels_sweep = [r"$\tan \delta_J$", r"$\tan \delta_g$", r"$\tan \delta_t$"]
# signs_sweep = ["--^", "--o", "--v"]
#
# model_base = "cable"
# model_type = "c"
#
# if model_base == "cable":
#     cg*=2
# delta_1m = ground_deltas_1m_80 if model_base == "ground" else cable_deltas_1m_80
# print_model = "Ground" if model_base == "ground" else "Cable"
# model = array_transmission_lorentzian.ArrayLorentzian(model_base + "_" + model_type,
#                                                       show_progression=True, show_fit_window=False)
# model.init_params(n_jct, lj, cj, cg, ct)
#model.plot_Q_mode(0, np.logspace(-7, -4, 10), 2, resistor_losses=False)

# tandj_l = [[delta_1m[0]], [0], [0]]
# tandg_l = [[0], [delta_1m[1]], [0]]
# tandt_l = [[0], [0], [delta_1m[2]]]
# for tandj, tandg, tandt in zip(tandj_l, tandg_l, tandt_l):
#     model.sweep_losses_tand(tandj, tandg, tandt)
#     f_res, q_res, param_res = model.get_variation()
#     results_qs.append(q_res[0])
#     results_fs.append([w / 2 / np.pi for w in f_res[0]])
#
# fig, axs = plt.subplots()
# q_tot = [1 / sum([1 / results_qs[i][m] for i in range(3)]) for m in range(len(results_qs[0]))]
# axs.plot(q_tot, "+--", label="Total")
# for i, r in enumerate([results_qs]):
#     for ind in range(len(r)):
#         axs.plot(r[ind], signs_sweep[ind], label=labels_sweep[ind])
#     axs.grid()
#     axs.legend()
#     axs.set_yscale('log')
# axs.set_ylabel("Q")
# axs.set_xlabel("Mode number")
# plt.show()


# sweep_vals = [[0, 0, 0] for _ in range(3)]
# for i in range(3):
#     sweep_vals[i][i] = delta_1m[i]
# results = [[], [], []]
# n_jct_sweep = np.logspace(1, 2, 20)
# for n_jct in tqdm(n_jct_sweep):
#     lj = 15 / n_jct
#     cj = 1 / lj / (15 * 2 * np.pi) ** 2
#     ct = 2 / (lj * n_jct * (5 * 2 * np.pi) ** 2)
#     model.init_params(n_jct, lj, cj, cg, ct)
#     for i, rs in enumerate(sweep_vals):
#         model.init_losses_tand(*rs)
#         model.compute_model(lst_to_fit=[0])
#         results[i].append(model.get_qs()[0])
# fig, ax = plt.subplots()
# for i in range(len(results)):
#     ax.plot(n_jct_sweep, results[i], signs_sweep[i], label=labels_sweep[i])
# ax.grid()
# ax.set_xlabel("$N_{jct}$")
# ax.set_ylabel("Q")
# ax.set_title("Variation of the quality factor with respect to the the number of junctions.\n("+print_model+" model)")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.legend()
# plt.show()

# sweep_vals = [[0, 0, 0] for _ in range(3)]
# for i in range(3):
#     sweep_vals[i][i] = delta_1m[i]
# results_q = [[], [], []]
# results_w = [[], [], []]
# lj_sweep = np.logspace(-1, 1, 20)
# for lj in tqdm(lj_sweep):
#     n_jct = 25
#     cj = 1 / lj / (15 * 2 * np.pi) ** 2
#     ct = 2 / (lj * n_jct * (5 * 2 * np.pi) ** 2)
#     model.init_params(n_jct, lj, cj, cg, ct)
#     for i, rs in enumerate(sweep_vals):
#         model.init_losses_tand(*rs)
#         model.compute_model(lst_to_fit=[0])
#         results_q[i].append(model.get_qs()[0])
#         results_w[i].append(model.get_pulsations()[0])
# fig, ax = plt.subplots(2, 1)
# for i in range(len(results_q)):
#     ax[0].plot(lj_sweep*n_jct, results_q[i], signs_sweep[i], label=labels_sweep[i])
#     ax[1].plot(lj_sweep*n_jct, [w/2/np.pi for w in results_w[i]], signs_sweep[i], label=labels_sweep[i])
# ax[0].grid()
# ax[1].grid()
# ax[1].set_xlabel("$L_{array}$")
# ax[0].set_ylabel("Q")
# ax[1].set_ylabel("f (GHz)")
# fig.suptitle(print_model+" model")
# ax[0].set_title("Variation of the fundamental quality factor with respect to the array inductance.")
# ax[1].set_title("Variation of the fundamental frequency with respect to the array inductance.")
# ax[1].set_xscale("log")
# ax[0].set_xscale("log")
# ax[0].set_yscale("log")
# ax[0].legend()
# ax[1].legend()
# plt.show()

# sweep_vals = [[0, 0, 0] for _ in range(3)]
# for i in range(3):
#     sweep_vals[i][i] = delta_1m[i]
# results_q = [[], [], []]
# results_w = [[], [], []]
# f_sweep = np.linspace(1,8 , 20)
# for f_res in tqdm(f_sweep):
#     ct = 2 / (lj * n_jct * (f_res * 2 * np.pi) ** 2)
#     model.init_params(n_jct, lj, cj, cg, ct)
#     for i, rs in enumerate(sweep_vals):
#         model.init_losses_tand(*rs)
#         model.compute_model(lst_to_fit=[0])
#         results_q[i].append(model.get_qs()[0])
#         results_w[i].append(model.get_pulsations()[0])
# fig, ax = plt.subplots(2, 1)
# for i in range(len(results_q)):
#     ax[0].plot(f_sweep, results_q[i], signs_sweep[i], label=labels_sweep[i])
#     ax[1].plot(f_sweep, [w / 2 / np.pi for w in results_w[i]], signs_sweep[i], label=labels_sweep[i])
# ax[0].grid()
# ax[1].grid()
# ax[1].set_xlabel("$f (GHz)$")
# ax[0].set_ylabel("Q")
# ax[1].set_ylabel("f (GHz)")
# fig.suptitle(print_model+" model")
# ax[0].set_title("Variation of the fundamental quality factor with respect to the targeted frequency.")
# ax[1].set_title("Variation of the fundamental frequency with respect to the targeted frequency.")
# ax[0].set_yscale("log")
# ax[0].legend()
# ax[1].legend()
# plt.show()

# sweep_vals = [[0, 0, 0] for _ in range(3)]
# for i in range(3):
#     sweep_vals[i][i] = delta_1m[i]
# results_q = [[], [], []]
# results_w = [[], [], []]
# cg_sweep = np.logspace(-8, -2, 20)
# if model_base == "cable":
#     cg_sweep*=2
# for cg in tqdm(cg_sweep):
#     model.init_params(n_jct, lj, cj, cg, ct)
#     for i, rs in enumerate(sweep_vals):
#         model.init_losses_tand(*rs)
#         model.compute_model(lst_to_fit=[0])
#         results_q[i].append(model.get_qs()[0])
#         results_w[i].append(model.get_pulsations()[0])
# fig, ax = plt.subplots(2, 1)
# for i in range(len(results_q)):
#     ax[0].plot(cg_sweep, results_q[i], signs_sweep[i], label=labels_sweep[i])
#     ax[1].plot(cg_sweep, [w / 2 / np.pi for w in results_w[i]], signs_sweep[i], label=labels_sweep[i])
# ax[0].grid()
# ax[1].grid()
# ax[1].set_xlabel("$C_g$")
# ax[0].set_ylabel("Q")
# ax[1].set_ylabel("$f (GHz)$")
# fig.suptitle(print_model+" model")
# ax[0].set_title("Variation of the fundamental quality factor with respect to the "+model_base+" capacity.")
# ax[1].set_title("Variation of the fundamental frequency with respect to the "+model_base+" capacity.")
# ax[0].set_yscale("log")
# ax[0].set_xscale("log")
# ax[1].set_xscale("log")
# ax[0].legend()
# ax[1].legend()
# plt.show()


n_jct = 26
lj = 15/ n_jct
cj = 1 / lj / (15 * 2 * np.pi) ** 2
cg = 1e-6
ct = 2 / (lj * n_jct * (5 * 2 * np.pi) ** 2)

for model_base, factor in zip(["ground", "cable"], [1, 2]):
    model = array_transmission_lorentzian.ArrayLorentzian(model_base + "_c",
                                                          show_progression=True, show_fit_window=False)
    model.init_params(n_jct, lj, cj, cg*factor, ct)
    model.compute_model()
    plt.plot(model._ws_init/2/np.pi, model._log_sqr_abs, label=model_base)
plt.plot([5, 5], [-10, 10], "r+--", label="perfect")
plt.grid()
plt.legend()
plt.show()

model_base = "cable"
model_type = "c"
model = array_transmission_lorentzian.ArrayLorentzian(model_base + "_" + model_type,
                                                      show_progression=True, show_fit_window=False)
model.init_params(n_jct, lj, cj, cg, ct)
model.compute_model()

phi_zpf = model.get_phi_zpf()
perfect_phi_zpf = model.get_perfect_grd_phi_zpf()
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot([1, n_jct], [perfect_phi_zpf]*2, label="Perfect")
for i, m in enumerate(phi_zpf[:4]):
    ax.plot(list(range(1, n_jct+1)), m, "+--", label=f"Mode {i}")
ax.grid()
ax.legend()
ax.set_ylabel("Junction number")
ax.set_ylabel(r"$\varphi_{ZPF}$")
ax.set_title(r"$\varphi_{ZPF}$ per junction for the first 4 modes.")
plt.show()

model.compute_kerr()
kerr = model.get_kerr()
n_mode = len(kerr)
plt.plot([kerr[i, i]*1e3 for i in range(kerr.shape[0])], "+--", label="self-kerr")
for i in range(n_mode-1):
    plt.plot(range(i+1, n_mode), [v*1e3 for v in kerr[i][i+1:]], "+--", label=f"cross-kerr with mode {i}")
plt.grid()
plt.legend()
plt.xlabel("Mode number")
plt.ylabel("Kerr (MHz)")
plt.title("Kerr")
plt.show()

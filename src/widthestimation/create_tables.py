import numpy as np
from matplotlib import pyplot as plt
import os
import utils
from matplotlib.pyplot import figure
import matplotlib
from scipy.ndimage.filters import gaussian_filter

# plot properties
matplotlib.rcParams.update({'font.size': 45})
fig = plt.figure(figsize=(18, 12), dpi=80)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# inspected structures
root_path = "./data"
structures = ["BridgeB", "BridgeS", "BridgeU", "Indoors"]
measurements = {
    "BridgeB": 12,
    "BridgeS": 76,
    "BridgeU": 8,
    "Indoors": 19
}

# load data
stats = np.empty((0, 16), np.float32)
profiles = np.empty((0, 30), np.float32)
widths = np.empty((0, 3), np.float32)

all_info = np.empty((0, 16), np.float32)
all_prof = np.empty((0, 30), np.float32)

""" DATA OVERVIEW TABLE """

tex_table_data = ""

# loop over structures
for struct in structures:
    # set path and variables
    path = os.path.join(root_path, struct)
    globals()[struct + "_info"] = np.empty((0, 16), np.float32)
    globals()[struct + "_prof"] = np.empty((0, 30), np.float32)

    # loop over csv files
    for i, f in enumerate(os.listdir(path)):
        file_path = os.path.join(path, f)

        with open(file_path, 'r') as tmp:
            content = tmp.read()

        info = content.split('\n')[1].split(',')
        prof = content.split('\n')[-1].split(',')

        info = np.array(info, np.float32)[:30]
        prof = np.array(prof, np.float32)[:30]

        prof = utils.centralize_profline(prof)

        globals()[struct + "_info"] = np.append(globals()[struct + "_info"], info.reshape(1, 16), axis=0)
        globals()[struct + "_prof"] = np.append(globals()[struct + "_prof"], prof.reshape(1, 30), axis=0)

    # create statistics table
    infos = globals()[struct + "_info"]
    profs = globals()[struct + "_prof"]

    tex_line = f"{struct} & \hfil{infos.shape[0]} & \hfil {measurements[struct]} & \hfil {str(np.round(infos.shape[0] / measurements[struct], 1))} " \
               f"& {np.mean(infos[:, -6]):.3f} & {np.median(infos[:, -6]):.3f} & {np.var(infos[:, -6]):.3f} " \
               f"& {np.mean(infos[:, -2]):.3f} & {np.median(infos[:, -2]):.3f} & {np.var(infos[:, -2]):.3f} " \
               f"& \hfil{np.mean(infos[:, -1]) // 1 / 100} \\\\ \n"
    tex_table_data += tex_line

    all_info = np.append(all_info, globals()[struct + "_info"], axis=0)
    all_prof = np.append(all_prof, globals()[struct + "_prof"], axis=0)

infos = all_info
profs = all_prof

tex_line = f"\hline Total & \hfil {infos.shape[0]} & \hfil {sum(measurements.values())} & \hfil {str(np.round(infos.shape[0] / sum(measurements.values()), 1))} " \
           f"& {np.mean(infos[:, -6]):.3f} & {np.median(infos[:, -6]):.3f} & {np.var(infos[:, -6]):.3f} " \
           f"& {np.mean(infos[:, -2]):.3f} & {np.median(infos[:, -2]):.3f} & {np.var(infos[:, -2]):.3f} " \
           f"& \hfil {np.mean(infos[:, -1]) // 1 / 100} \\\\ \n"
tex_table_data += tex_line

with open("content/table_data_overview.tex", 'w') as f:
    f.write(tex_table_data)

""" AVERAGE CRACK PROFILE """

trans_prof = np.zeros((profs.shape[0], 10 * 30))

for i in range(profs.shape[0]):
    # vertical normalization
    context_med = np.median(np.roll(profs[i, :], 10)[:20])
    prof_tmp = (profs[i, :] - np.min(profs[i, :])) / (context_med - np.min(profs[i, :]))

    # horizontal normalization
    orig_x = np.arange(0, 30, 0.1)
    orig_y = np.interp(orig_x, np.arange(0, 30), prof_tmp)
    w_half = infos[i, -2] / 2
    trans_x = 0.5 / w_half * (orig_x - 15) + 15
    trans_prof[i, :] = np.interp(np.arange(0, 30, 0.1), trans_x, orig_y)

    plt.plot(np.arange(-15, 15, 0.1), trans_prof[i, :], color="gray", alpha=0.03)

linewidth = 3
# plot bottom line
plt.plot([-10, 10], [0, 0], '--', color='gray', alpha=0.9, linewidth=linewidth)
# plot idealized profile
plt.plot([-15, -0.499, -0.5, 0.499, 0.5, 16], [1, 1, 0, 0, 1, 1], color="red", alpha=0.9, linewidth=linewidth,
         label="Idealized")

# averages
mean = np.mean(trans_prof, axis=0)
median = np.median(trans_prof, axis=0)
percentile80 = np.percentile(trans_prof, 80, axis=0)

# plot averages
plt.plot(np.arange(-15, 15, 0.1), mean, '--', linewidth=linewidth, alpha=0.99, label="Mean",
         color="purple")  # alpha=0.7)
plt.plot(np.arange(-15, 15, 0.1), median, '-.', linewidth=linewidth, alpha=0.99, label="Median",
         color="green")  # alpha=0.7)
plt.plot(np.arange(-15, 15, 0.1), percentile80, ':', linewidth=linewidth, alpha=0.99, label="80\,\%-Percentile",
         color="blue")  # alpha=0.7)

# plot a, b, c, d
plt.text(4.3, 0.98, 'a', fontsize=40, color="dimgray")  # 16, color="dimgray")
plt.text(4.3, -0.02, 'b', fontsize=40, color="dimgray")  # 16, color="dimgray")
plt.text(-0.57, -0.07, 'd', fontsize=40, color="dimgray")  # 16, color="dimgray")
plt.text(0.405, -0.07, 'c', fontsize=40, color="dimgray")  # 16, color="dimgray")

# plot properties
plt.xlabel("Normalized Position")
plt.ylabel("Normalized Gray-scale Value")
plt.xticks(np.arange(-10, 10, 1))
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
plt.xlim([-4.2, 4.2])
plt.ylim([-0.1, 1.19])
plt.rc('legend', fontsize=32)
plt.legend(loc='lower right', bbox_to_anchor=(0.995, 0.09))

fig.savefig("content/crack_profile.pdf", dpi=80, bbox_inches='tight')
fig.savefig("content/crack_profile.png", dpi=80, bbox_inches='tight')

""" RESULTS TABLE """

widths = np.empty((profs.shape[0], 10), np.float32)

for i in range(profs.shape[0]):
    blurred = gaussian_filter(profs[i, :], sigma=2)
    blurred2 = gaussian_filter(profs[i, :], sigma=4)

    # ground truth
    widths[i, 0] = infos[i, -2]

    # rectangle transform
    widths[i, 1] = utils.rectangle_transform(profs[i, :], base=20, height=0.9)
    widths[i, 2] = utils.rectangle_transform(blurred, base=20, height=0.9)
    widths[i, 7] = utils.rectangle_transform(blurred2, base=20, height=0.9)

    # naive intersection
    widths[i, 3] = utils.intersection_approach(profs[i, :], 0.3)
    widths[i, 4] = utils.intersection_approach(blurred, 0.3)
    widths[i, 8] = utils.intersection_approach(blurred2, 0.3)

    # parabola intersection
    widths[i, 5] = utils.fit_parabola(profs[i, :])
    widths[i, 6] = utils.fit_parabola(blurred)
    widths[i, 9] = utils.fit_parabola(blurred2)

    # plt.plot(blurred)
    # plt.show()


def fill_row(pred, cond, widths, infos):
    res = f"& {np.nanmean(np.abs(pred[cond] - widths[cond, 0])):.3f} & {np.nanmean(np.abs(pred[cond] * infos[cond, -3] - infos[cond, -6])):.3f} " \
          f"& \hfil {np.nanmean(np.abs(pred[cond] - widths[cond, 0]) / widths[cond, 0]):.3f} " \
          f" & \hfill {np.count_nonzero(~np.isnan(pred[cond])) / len(pred[cond]) * 100:.1f}\,\% " \
          f" & \hfill {np.sum(np.where(((infos[cond, -6] - 0.025) / infos[cond, -3] <= pred[cond]) * (pred[cond] <= (infos[cond, -6] + 0.025) / infos[cond, -3]), 1, 0) / np.sum(cond * 1)) * 100:.0f}\,\% " \
          f" & \hfill {np.sum(np.where(((infos[cond, -6] - 0.050) / infos[cond, -3] <= pred[cond]) * (pred[cond] <= (infos[cond, -6] + 0.050) / infos[cond, -3]), 1, 0) / np.sum(cond * 1)) * 100:.0f}\,\% " \
          f" & \hfill {np.sum(np.where(((infos[cond, -6] - 0.100) / infos[cond, -3] <= pred[cond]) * (pred[cond] <= (infos[cond, -6] + 0.100) / infos[cond, -3]), 1, 0) / np.sum(cond * 1)) * 100:.0f}\,\% " \
          f" & \hfill {np.sum(np.where(((infos[cond, -6] - 0.200) / infos[cond, -3] <= pred[cond]) * (pred[cond] <= (infos[cond, -6] + 0.200) / infos[cond, -3]), 1, 0) / np.sum(~np.isnan(pred[cond]))) * 100:.0f}\,\% \\\\ \n"

    return res


# create table
tex_table_results = ""

# condition
cond = np.where((infos[:, -2] <= 2), True, False)
tex_table_results += " \multirow{3}{*}{\hfil $0 < w \leq 2$} "
tex_table_results += f" & Intersect "
tex_table_results += fill_row(pred=widths[:, 3], cond=cond, widths=widths, infos=infos)

cond = np.where((infos[:, -2] <= 2), True, False)
tex_table_results += f" & Parabola "
tex_table_results += fill_row(pred=widths[:, 5], cond=cond, widths=widths, infos=infos)

cond = np.where((infos[:, -2] <= 2), True, False)
tex_table_results += f" & Rectangle "
tex_table_results += fill_row(pred=widths[:, 1], cond=cond, widths=widths, infos=infos)

tex_table_results += "\hline "

tex_table_results += " \multirow{3}{*}{\hfil $2 < w \leq 15$} "
cond = np.where((infos[:, -2] > 2), True, False)
tex_table_results += f" & Intersect "
tex_table_results += fill_row(pred=widths[:, 3], cond=cond, widths=widths, infos=infos)

cond = np.where((infos[:, -2] > 2), True, False)
tex_table_results += f" & Parabola "
tex_table_results += fill_row(pred=widths[:, 5], cond=cond, widths=widths, infos=infos)

cond = np.where((infos[:, -2] > 2), True, False)
tex_table_results += f" & Rectangle "
tex_table_results += fill_row(pred=widths[:, 1], cond=cond, widths=widths, infos=infos)

tex_table_results += "\hline "

tex_table_results += " \multirow{3}{*}{\hfil $0 < w \leq 15$} "
cond = np.where((infos[:, -2] >= 0), True, False)
tex_table_results += f" & Intersect "
tex_table_results += fill_row(pred=widths[:, 3], cond=cond, widths=widths, infos=infos)

cond = np.where((infos[:, -2] >= 0), True, False)
tex_table_results += f" & Parabola "
tex_table_results += fill_row(pred=widths[:, 5], cond=cond, widths=widths, infos=infos)

cond = np.where((infos[:, -2] >= 0), True, False)
tex_table_results += f" & Rectangle "
tex_table_results += fill_row(pred=widths[:, 1], cond=cond, widths=widths, infos=infos)

tex_table_results += "\hlineB{2} "

# blur
tex_table_results += " \multirow{3}{*}{\shortstack{\hfil $0 < w \leq 15$, \\\\ blur, $\sigma=2$}} "  # (0,15]
cond = np.where((infos[:, -2] >= 0), True, False)
tex_table_results += f" & Intersect "
tex_table_results += fill_row(pred=widths[:, 4], cond=cond, widths=widths, infos=infos)

cond = np.where((infos[:, -2] >= 0), True, False)
tex_table_results += f" & Parabola "
tex_table_results += fill_row(pred=widths[:, 6], cond=cond, widths=widths, infos=infos)

cond = np.where((infos[:, -2] >= 0), True, False)
tex_table_results += f" & Rectangle "
tex_table_results += fill_row(pred=widths[:, 2], cond=cond, widths=widths, infos=infos)

tex_table_results += "\hline "

tex_table_results += " \multirow{3}{*}{\shortstack{\hfil $0 < w \leq 15$, \\\\ blur, $\sigma=4$}} "  # (0,15]
cond = np.where((infos[:, -2] >= 0), True, False)
tex_table_results += f" & Intersect "
tex_table_results += fill_row(pred=widths[:, 8], cond=cond, widths=widths, infos=infos)

cond = np.where((infos[:, -2] >= 0), True, False)
tex_table_results += f" & Parabola "
tex_table_results += fill_row(pred=widths[:, 9], cond=cond, widths=widths, infos=infos)

cond = np.where((infos[:, -2] >= 0), True, False)
tex_table_results += f" & Rectangle "
tex_table_results += fill_row(pred=widths[:, 7], cond=cond, widths=widths, infos=infos)

with open("content/table_result_overview.tex", 'w') as f:
    f.write(tex_table_results)


""" SCATTER PLOT """

plt.clf()
plt.plot(widths[:, 0], widths[:, 1] - widths[:, 0], 'o', color='#1b9e77', alpha=0.4, markeredgewidth=0.0,
         markersize=11.0)
plt.plot(widths[:, 0], widths[:, 3] - widths[:, 0], '^', color='#d95f02', alpha=0.2, markeredgewidth=0.0,
         markersize=11.0)
cond = ~np.isnan(widths[:, 5])
plt.plot(widths[cond, 0], widths[cond, 5] - widths[cond, 0], 's', color='#7570b3', alpha=0.4, markeredgewidth=0.0,
         markersize=10.0)

plt.plot((-1, 15), (0, 0), '--', color='gray', alpha=0.7)

plt.xlabel("True Width [px]")
plt.ylabel("Mean Absolute Error [px]")
plt.xlim([0, 6])
plt.ylim([-2, 3])

plt.plot(np.nan, np.nan, 'o', color='#1b9e77', alpha=1.0, markeredgewidth=0.0, markersize=11.0, label="Rectangle")
plt.plot(np.nan, np.nan, '^', color='#d95f02', alpha=1.0, markeredgewidth=0.0, markersize=11.0, label="Intersection")
plt.plot(np.nan, np.nan, 's', color='#7570b3', alpha=1.0, markeredgewidth=0.0, markersize=10.0, label="Parabola")
plt.legend(loc='lower right', bbox_to_anchor=(0.995, 0.73))

fig.savefig("content/results_scatter.pdf", dpi=80, bbox_inches='tight')
fig.savefig("content/results_scatter.png", dpi=80, bbox_inches='tight')

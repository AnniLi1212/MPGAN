import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak

myfile = uproot.open("./datasets/testCalo.root")
mytree = myfile["cell_tree"]
mydata = mytree.arrays()


LAYER_SPECS = [(3, 96), (12, 12), (12, 6)]
Lnt = [LAYER_SPECS[i][0] * LAYER_SPECS[i][1] for i in range(3)]

# plt.hist(ak.count(mydata["EnergyVector"], axis=1), bins=np.linspace(0, 100, 101))


def ak_to_np(arr: ak.Array, clip_target: int = 100):
    return ak.fill_none(ak.pad_none(arr, clip_target, axis=1, clip=True), 0, axis=None).to_numpy()


layers_pos = []
layers_energy = []

# The first is the total energy
layers_pos.append(
    ak_to_np(
        mydata["PositionVector"][
            (mydata["PositionVector"] < Lnt[0]) * (mydata["PositionVector"] >= 0)
        ]
    ).astype(int)
)


layers_energy.append(
    ak_to_np(
        mydata["EnergyVector"][
            (mydata["PositionVector"] < Lnt[0]) * (mydata["PositionVector"] >= 0)
        ]
    )
)

layers_pos.append(
    ak_to_np(
        mydata["PositionVector"][
            (mydata["PositionVector"] < Lnt[0] + Lnt[1]) * (mydata["PositionVector"] >= Lnt[0])
        ]
        - Lnt[0]
    ).astype(int)
)

layers_energy.append(
    ak_to_np(
        mydata["EnergyVector"][
            (mydata["PositionVector"] < Lnt[0] + Lnt[1]) * (mydata["PositionVector"] >= Lnt[0])
        ]
    )
)

# The last three are overflow
layers_pos.append(
    ak_to_np(
        mydata["PositionVector"][
            (mydata["PositionVector"] >= Lnt[0] + Lnt[1])
            * (mydata["PositionVector"] < Lnt[0] + Lnt[1] + Lnt[2])
        ]
        - Lnt[0]
        - Lnt[1]
    ).astype(int)
)

layers_energy.append(
    ak_to_np(
        mydata["EnergyVector"][
            (mydata["PositionVector"] >= Lnt[0] + Lnt[1])
            * (mydata["PositionVector"] < Lnt[0] + Lnt[1] + Lnt[2])
        ]
    )
)


layers_pos2d = []


# convert 1D coordinates into 2D, and pad z value
for i in range(len(LAYER_SPECS)):
    phi, eta = np.unravel_index(layers_pos[i], LAYER_SPECS[i])
    phi = (phi + 0.5) / (LAYER_SPECS[i][0])
    eta = (eta + 0.5) / (LAYER_SPECS[i][1])

    layers_pos2d.append(
        np.pad(
            np.stack((eta, phi), axis=-1),
            ((0, 0), (0, 0), (0, 1)),
            constant_values=(i + 0.5) / 3.0,
        )
    )


layers = []

for i in range(len(LAYER_SPECS)):
    layers.append(np.concatenate((layers_pos2d[i], layers_energy[i][..., np.newaxis]), axis=2))

dataset = np.concatenate(layers, axis=1)

NUM_HITS = 30

clipped_dataset = np.array(list(map(lambda x: x[x[:, 3].argsort()][-NUM_HITS:], dataset)))[:, ::-1]


mask = (clipped_dataset[:, :, 3:] != 0).astype(float)

final_dataset = np.concatenate((clipped_dataset, mask), axis=2)

np.save("./datasets/calogan_graph_30_hits_etaphizEmask.npy", final_dataset)


final_dataset

# plt.hist(final_dataset[:)

final_dataset.shape
mask.shape


hits = final_dataset[mask[:, :, 0].astype(bool)]

# layer_hits = []
layer_etas = []
layer_phis = []
layer_Es = []

for i in range(3):
    layer_hits = hits[(hits[:, 2] > i * 0.33) * (hits[:, 2] < (i + 1) * 0.33)]
    layer_etas.append(((layer_hits[:, 0] * LAYER_SPECS[i][1]) - 0.5).astype(int))
    layer_phis.append(((layer_hits[:, 1] * LAYER_SPECS[i][0]) - 0.5).astype(int))
    layer_Es.append(layer_hits[:, 3])


import mplhep as hep
from matplotlib.colors import LogNorm

plt.switch_backend("agg")
plt.rcParams.update({"font.size": 12})
plt.style.use(hep.style.CMS)

fig = plt.figure(figsize=(22, 22), constrained_layout=True)
fig.suptitle(" ")

subfigs = fig.subfigures(nrows=3, ncols=1)

for i, subfig in enumerate(subfigs):
    subfig.suptitle(f"Layer {i + 1}")

    # create 1x3 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=3)
    axs[0].hist(layer_etas[i], bins=np.arange(LAYER_SPECS[i][1] + 1), histtype="step", label="test")
    axs[0].ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
    axs[0].set_xlabel(r"Hit $\eta$s")
    axs[0].set_ylabel("Number of Hits")
    axs[0].legend()

    axs[1].hist(layer_phis[i], bins=np.arange(LAYER_SPECS[i][0] + 1), histtype="step")
    axs[1].ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
    axs[1].set_xlabel(r"Hit $\varphi$s")
    axs[1].set_ylabel("Number of Hits")

    bins = np.linspace(0, 50, 51) if i < 2 else np.linspace(0, 4, 51)
    axs[2].hist(layer_Es[i] / 1000, bins=bins, histtype="step")
    axs[2].set_xlabel("Hit Energies (GeV)")
    axs[2].set_ylabel("Number of Hits")
    axs[2].set_yscale("log")


def to_image(phi_size, eta_size, phis, etas, Es):
    jet_image = np.zeros((phi_size, eta_size))

    for eta, phi, pt in zip(phis, etas, Es):
        if eta >= 0 and eta < eta_size and phi >= 0 and phi < phi_size:
            jet_image[phi, eta] += pt

    return jet_image


real_showers = final_dataset

real_ims = []

NUM_IMS = 1000

for j in range(3):
    real_layer_ims = []
    for i in range(NUM_IMS):
        real_layer_hits = real_showers[i][
            (real_showers[i][:, 2] > j * 0.33) * (real_showers[i][:, 2] < (j + 1) * 0.33)
        ]
        real_etas = ((real_layer_hits[:, 0] * LAYER_SPECS[j][1]) - 0.5).astype(int)
        real_phis = ((real_layer_hits[:, 1] * LAYER_SPECS[j][0]) - 0.5).astype(int)
        real_Es = real_layer_hits[:, 3]
        real_layer_ims.append(
            to_image(LAYER_SPECS[j][0], LAYER_SPECS[j][1], real_etas, real_phis, real_Es)
        )

    real_ims.append(real_layer_ims)


fig = plt.figure(figsize=(24, 32), constrained_layout=True)
fig.suptitle(" ")

subfigs = fig.subfigures(nrows=1, ncols=3)

vmins = [1e-2, 1e-2, 1e-4]
vmaxs = [1e5, 1e3, 10]

for i, subfig in enumerate(subfigs):
    subfig.suptitle(f"Layer {i + 1}")

    axs = subfig.subplots(nrows=6, ncols=1)

    for j in range(5):
        # print(f"{i} {j}")
        # print(real_ims[j][i])

        pos = axs[j].imshow(
            real_ims[i][j],
            aspect="auto",
            cmap="binary",
            vmin=vmins[i],
            vmax=vmaxs[i],
            norm=LogNorm(),
            extent=(0, LAYER_SPECS[i][1], 0, LAYER_SPECS[i][0]),
        )
        axs[j].set_xlabel(r"$\eta$")
        axs[j].set_ylabel(r"$\varphi$")
        fig.colorbar(pos, ax=axs[j])

    j = 5
    axs[j].set_title("Average Image")
    pos = axs[j].imshow(
        np.mean(np.array(real_ims[i]), axis=0),
        aspect="auto",
        cmap="binary",
        vmin=vmins[i],
        vmax=vmaxs[i],
        norm=LogNorm(),
        extent=(0, LAYER_SPECS[i][1], 0, LAYER_SPECS[i][0]),
    )
    axs[j].set_xlabel(r"$\eta$")
    axs[j].set_ylabel(r"$\varphi$")
    fig.colorbar(pos, ax=axs[j])


real_ims[0][0]

j = 0
plt.imshow(
    real_ims[0][j],
    aspect="auto",
    cmap="binary",
    vmin=1e-1,
    vmax=1000,
    norm=LogNorm(),
    extent=(0, LAYER_SPECS[j][1], 0, LAYER_SPECS[j][0]),
)
plt.xlabel(r"$\eta$")
plt.ylabel(r"$\varphi$")


final_dataset[i]
import json
import numpy as np
import matplotlib.pyplot as plt


with open("times/3d-3dig.json") as infile:
    d_1 = json.load(infile)

with open("times/3d-4dig.json") as infile:
    d_2 = json.load(infile)


fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(15, 5))

t = np.logspace(2, 7, num=100, base=10.0, dtype=int)
ax[0].loglog(t, d_1["pulp"]["mean"], c="firebrick", label="pulp")
ax[0].loglog(t, d_1["dioph"]["mean"], c="navy", label="dioph")
ax[0].grid(ls="--", c="silver", alpha=0.7)
ax[0].legend()
ax[0].set_xlabel("rhs")
ax[0].set_ylabel("time [ns]")

ax[1].loglog(t, d_2["pulp"]["mean"], c="firebrick", label="pulp")
ax[1].loglog(t, d_2["dioph"]["mean"], c="navy", label="dioph")
ax[1].set_xlabel("rhs")
ax[1].grid(ls="--", c="silver", alpha=0.7)

plt.savefig("static/3d-digits.pdf")

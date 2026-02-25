import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

def box(ax, x, y, w, h, text, fontsize=11):
    """(x,y) is bottom-left in axes coords (0..1)."""
    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.6)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

def arrow(ax, x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->",
        mutation_scale=13,
        linewidth=1.6,
        connectionstyle=f"arc3,rad={rad}"
    ))

# --- Canvas ---
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# --- Sizes ---
W_small, H_small = 0.14, 0.14
W_med,   H_med   = 0.16, 0.18
W_big,   H_big   = 0.18, 0.22

# --- Left inputs ---
box(ax, 0.05, 0.73, W_small, H_small, "Source\nFrame\n($x_t$)", fontsize=12)
box(ax, 0.05, 0.50, W_small, H_small, "Target\nFrame\n($x_{t+1}$)", fontsize=12)

# --- Visual encoder ---
box(ax, 0.24, 0.56, 0.12, 0.30, "Visual\nEncoder", fontsize=12)

arrow(ax, 0.05+W_small, 0.73+H_small*0.50, 0.24, 0.56+0.30*0.75)
arrow(ax, 0.05+W_small, 0.50+H_small*0.50, 0.24, 0.56+0.30*0.25)

# --- Upper features (directly after encoder) ---
box(ax, 0.40, 0.72, W_small, H_small, "Source\nFeatures\n($z_t$)", fontsize=12)
box(ax, 0.40, 0.52, W_small, H_small, "Target\nFeatures\n($z_{t+1}$)", fontsize=12)

arrow(ax, 0.24+0.12, 0.56+0.30*0.75, 0.40, 0.72+H_small*0.50)
arrow(ax, 0.24+0.12, 0.56+0.30*0.25, 0.40, 0.52+H_small*0.50)

# --- Midway inverse dynamics ---
box(ax, 0.58, 0.61, W_big, H_big, "Midway\nInverse\nDynamics\n(Motion\nInference)", fontsize=12)

arrow(ax, 0.40+W_small, 0.72+H_small*0.50, 0.58, 0.61+H_big*0.65)
arrow(ax, 0.40+W_small, 0.52+H_small*0.50, 0.58, 0.61+H_big*0.35)

# --- Motion latents ---
box(ax, 0.80, 0.70, W_small, H_small, "Motion\nLatents\n($m$)", fontsize=12)
arrow(ax, 0.58+W_big, 0.61+H_big*0.50, 0.80, 0.70+H_small*0.50)

# --- Lower branch features (feeding forward dynamics) ---
box(ax, 0.40, 0.28, W_small, H_small, "Source\nFeatures\n($z_t$)", fontsize=12)
box(ax, 0.40, 0.08, W_small, H_small, "Target\nFeatures\n($z_{t+1}$)", fontsize=12)

# --- Forward dynamics predictor ---
box(ax, 0.58, 0.18, W_big, H_big, "Forward\nDynamics\nPredictor", fontsize=12)

arrow(ax, 0.40+W_small, 0.28+H_small*0.50, 0.58, 0.18+H_big*0.65)
arrow(ax, 0.40+W_small, 0.08+H_small*0.50, 0.58, 0.18+H_big*0.35)

# motion latent feeds forward predictor (downward then left-ish)
arrow(ax, 0.80+W_small*0.50, 0.70, 0.72, 0.18+H_big, rad=0.0)

# --- Predicted target features ---
box(ax, 0.80, 0.26, 0.18, 0.16, "Predicted\nTarget\nFeatures\n($\\hat{z}_{t+1}$)", fontsize=12)
arrow(ax, 0.58+W_big, 0.18+H_big*0.50, 0.80, 0.26+0.16*0.50)

# --- Prediction loss ---
box(ax, 0.88, 0.50, 0.11, 0.14, "Prediction\nLoss", fontsize=12)

# predicted -> loss
arrow(ax, 0.80+0.18, 0.26+0.16*0.70, 0.88, 0.50+0.14*0.55)

# target features -> loss (long bottom wire then up)
# (draw as two arrows to mimic the L-shaped wire cleanly)
arrow(ax, 0.40+W_small*0.50, 0.08, 0.90, 0.08)                 # along bottom
arrow(ax, 0.90, 0.08, 0.90, 0.50)                              # up
arrow(ax, 0.90, 0.50, 0.88, 0.50+0.14*0.40)                    # into loss

plt.tight_layout()
plt.savefig("midway_gemini_style_fixed.png", dpi=300, bbox_inches="tight")
plt.show()
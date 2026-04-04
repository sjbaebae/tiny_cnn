"""
Animated training visualization for tinyCNN on MNIST.
Generates a polished MP4 showing the model learning in real-time.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from mnist_cnn import CNN_Base
from trainer import load_data

# ── CONFIG ──────────────────────────────────────────────────────────────────
NUM_EPOCHS = 3
SNAPSHOT_EVERY_N_BATCHES = 10        # capture a frame every N batches
FPS = 24
OUTPUT_FILE = "tinycnn_training.mp4"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── COLORS / THEME ──────────────────────────────────────────────────────────
BG        = "#0d1117"
PANEL_BG  = "#161b22"
ACCENT1   = "#58a6ff"   # blue
ACCENT2   = "#3fb950"   # green
ACCENT3   = "#f78166"   # orange
ACCENT4   = "#d2a8ff"   # purple
TEXT      = "#e6edf3"
TEXT_DIM  = "#8b949e"
GRID_CLR  = "#21262d"

DIGIT_CMAP = LinearSegmentedColormap.from_list("digit", ["#0d1117", "#58a6ff", "#ffffff"])
FILTER_CMAP = "coolwarm"

# ── SNAPSHOT COLLECTOR ──────────────────────────────────────────────────────
class TrainingRecorder:
    def __init__(self):
        self.batch_losses = []
        self.batch_accs = []
        self.epoch_train_losses = []
        self.epoch_test_losses = []
        self.epoch_test_accs = []
        self.filter_snapshots = []      # conv1 weights over time
        self.prediction_snapshots = []  # (images, true_labels, pred_labels, confidences, probs)
        self.training_images = []       # actual batch images the model just trained on
        self.global_step = 0

    def snapshot_filters(self, model):
        w = model.conv1.weight.detach().cpu().clone()
        self.filter_snapshots.append(w)

    def snapshot_predictions(self, model, sample_images, sample_labels):
        model.eval()
        with torch.no_grad():
            logits = model(sample_images.to(DEVICE))
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
        model.train()
        self.prediction_snapshots.append((
            sample_images.cpu(),
            sample_labels.cpu(),
            preds.cpu(),
            confs.cpu(),
            probs.cpu(),
        ))

    def snapshot_training_batch(self, x, y, logits):
        """Capture a few images from the current training batch."""
        probs = F.softmax(logits.detach(), dim=1)
        confs, preds = probs.max(dim=1)
        # take first 4 from the batch
        n = min(4, x.shape[0])
        self.training_images.append((
            x[:n].detach().cpu(),
            y[:n].detach().cpu(),
            preds[:n].detach().cpu(),
            confs[:n].detach().cpu(),
        ))


def train_and_record():
    print("Loading MNIST...")
    train_loader, test_loader = load_data(flatten=False)

    # grab 10 fixed sample images (one per digit)
    all_imgs, all_lbls = [], []
    for imgs, lbls in test_loader:
        all_imgs.append(imgs)
        all_lbls.append(lbls)
    all_imgs = torch.cat(all_imgs)
    all_lbls = torch.cat(all_lbls)

    sample_indices = []
    for digit in range(10):
        idx = (all_lbls == digit).nonzero(as_tuple=True)[0][0].item()
        sample_indices.append(idx)
    sample_images = all_imgs[sample_indices]
    sample_labels = all_lbls[sample_indices]

    print(f"Training on {DEVICE}...")
    model = CNN_Base(in_channels=1, out_channels=10).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    rec = TrainingRecorder()

    # initial snapshot
    rec.snapshot_filters(model)
    rec.snapshot_predictions(model, sample_images, sample_labels)
    rec.training_images.append(None)  # placeholder for frame 0
    rec.batch_losses.append(None)
    rec.batch_accs.append(None)

    total_batches = len(train_loader)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_acc = (logits.argmax(1) == y).float().mean().item()
            epoch_loss += batch_loss

            rec.batch_losses.append(batch_loss)
            rec.batch_accs.append(batch_acc)
            rec.global_step += 1

            if batch_idx % SNAPSHOT_EVERY_N_BATCHES == 0:
                rec.snapshot_filters(model)
                rec.snapshot_predictions(model, sample_images, sample_labels)
                rec.snapshot_training_batch(x, y, logits)
                pct = (batch_idx / total_batches) * 100
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  batch {batch_idx}/{total_batches} ({pct:.0f}%)  loss={batch_loss:.4f}  acc={batch_acc:.2%}")

        avg_train_loss = epoch_loss / total_batches
        rec.epoch_train_losses.append(avg_train_loss)

        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                test_loss += loss_fn(logits, yb).item()
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        rec.epoch_test_losses.append(test_loss / len(test_loader))
        rec.epoch_test_accs.append(correct / total)
        print(f"  => Epoch {epoch+1} done | train_loss={avg_train_loss:.4f} | test_acc={correct/total:.2%}")

    return rec


def render_animation(rec: TrainingRecorder):
    import matplotlib.animation as animation

    num_frames = len(rec.filter_snapshots)
    steps_per_frame = SNAPSHOT_EVERY_N_BATCHES

    print(f"Rendering {num_frames} frames at {FPS} fps...")

    # ── FIGURE LAYOUT ───────────────────────────────────────────────────────
    #  Row 0: [Loss curve (2 cols)]  [Accuracy curve (2 cols)]
    #  Row 1: [Conv1 Filters 4x4 grid (2 cols)]  [Current image + confidence bars (2 cols)]
    #  Row 2: [Live predictions strip - all 10 digits (4 cols)]
    fig = plt.figure(figsize=(16, 9), facecolor=BG, dpi=120)
    gs = gridspec.GridSpec(3, 4, figure=fig,
                           left=0.06, right=0.97, top=0.90, bottom=0.07,
                           hspace=0.50, wspace=0.35)

    title_text = fig.text(0.5, 0.96, "tinyCNN  Training on MNIST",
                          ha="center", va="center", fontsize=22, fontweight="bold",
                          color=TEXT, fontfamily="monospace",
                          path_effects=[patheffects.withStroke(linewidth=3, foreground=BG)])
    subtitle_text = fig.text(0.5, 0.925, "",
                             ha="center", va="center", fontsize=11,
                             color=TEXT_DIM, fontfamily="monospace")

    def make_ax(gs_spec, title=""):
        ax = fig.add_subplot(gs_spec, facecolor=PANEL_BG)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold",
                     fontfamily="monospace", pad=8)
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)
        return ax

    # Row 0: Loss + Accuracy
    ax_loss = make_ax(gs[0, :2], "Loss")
    ax_loss.set_xlabel("batch", color=TEXT_DIM, fontsize=8)
    ax_loss.set_ylabel("cross-entropy", color=TEXT_DIM, fontsize=8)
    ax_loss.grid(True, color=GRID_CLR, alpha=0.5, linewidth=0.5)
    loss_line, = ax_loss.plot([], [], color=ACCENT1, linewidth=1.2, alpha=0.4)
    loss_smooth_line, = ax_loss.plot([], [], color=ACCENT3, linewidth=2.2)

    ax_acc = make_ax(gs[0, 2:], "Accuracy")
    ax_acc.set_xlabel("batch", color=TEXT_DIM, fontsize=8)
    ax_acc.set_ylabel("batch accuracy", color=TEXT_DIM, fontsize=8)
    ax_acc.grid(True, color=GRID_CLR, alpha=0.5, linewidth=0.5)
    ax_acc.set_ylim(-0.05, 1.05)
    acc_line, = ax_acc.plot([], [], color=ACCENT2, linewidth=1.2, alpha=0.4)
    acc_smooth_line, = ax_acc.plot([], [], color=ACCENT2, linewidth=2.2)

    # Row 1 left: Conv1 filters (4x4 grid, upscaled)
    ax_filters = make_ax(gs[1, :2], "Conv1 Filters")
    ax_filters.axis("off")

    # Row 1 right: split into current image + confidence bars
    # Use a nested gridspec for the right panel
    gs_right = gs[1, 2:].subgridspec(1, 2, width_ratios=[1, 2.5], wspace=0.15)
    ax_cur_img = fig.add_subplot(gs_right[0, 0], facecolor=PANEL_BG)
    ax_cur_img.axis("off")
    ax_conf = fig.add_subplot(gs_right[0, 1], facecolor=PANEL_BG)
    ax_conf.set_title("Prediction Confidence", color=TEXT, fontsize=10,
                      fontweight="bold", fontfamily="monospace", pad=8)

    # Row 2: Live predictions (all 10 digits)
    ax_preds = make_ax(gs[2, :], "Live Predictions  (10 fixed test digits)")
    ax_preds.axis("off")

    def ema(data, alpha=0.05):
        out = []
        val = data[0]
        for d in data:
            val = alpha * d + (1 - alpha) * val
            out.append(val)
        return out

    # ── ANIMATION FUNCTION ──────────────────────────────────────────────────
    def update(frame_idx):
        if frame_idx == 0:
            n_batches = 1
        else:
            n_batches = min(frame_idx * steps_per_frame, len(rec.batch_losses))

        raw_losses = [l for l in rec.batch_losses[1:n_batches] if l is not None]
        raw_accs = [a for a in rec.batch_accs[1:n_batches] if a is not None]

        # ── Subtitle ────────────────────────────────────────────────────────
        batches_per_epoch = len(rec.batch_losses) // max(NUM_EPOCHS, 1)
        epoch_est = 0
        if batches_per_epoch > 0 and len(raw_losses) > 0:
            epoch_est = min(len(raw_losses) // batches_per_epoch, NUM_EPOCHS - 1)
        cur_loss = raw_losses[-1] if raw_losses else 0
        cur_acc = raw_accs[-1] if raw_accs else 0
        subtitle_text.set_text(
            f"epoch {epoch_est+1}/{NUM_EPOCHS}   batch {len(raw_losses)}   "
            f"loss {cur_loss:.3f}   acc {cur_acc:.1%}"
        )

        # ── Loss panel ──────────────────────────────────────────────────────
        if raw_losses:
            xs = list(range(len(raw_losses)))
            loss_line.set_data(xs, raw_losses)
            smoothed = ema(raw_losses)
            loss_smooth_line.set_data(xs, smoothed)
            ax_loss.set_xlim(0, max(len(rec.batch_losses), len(xs) + 10))
            ymax = max(raw_losses[:50]) if len(raw_losses) >= 50 else max(raw_losses) * 1.1
            ax_loss.set_ylim(0, min(ymax, 5))

        # ── Accuracy panel ──────────────────────────────────────────────────
        if raw_accs:
            xs = list(range(len(raw_accs)))
            acc_line.set_data(xs, raw_accs)
            smoothed = ema(raw_accs)
            acc_smooth_line.set_data(xs, smoothed)
            ax_acc.set_xlim(0, max(len(rec.batch_accs), len(xs) + 10))

        # ── Filters panel (4x4 grid, upscaled) ──────────────────────────────
        ax_filters.clear()
        ax_filters.set_title("Conv1 Filters", color=TEXT, fontsize=10,
                             fontweight="bold", fontfamily="monospace", pad=8)
        ax_filters.axis("off")
        ax_filters.set_facecolor(PANEL_BG)

        filters = rec.filter_snapshots[min(frame_idx, len(rec.filter_snapshots) - 1)]
        n_filters = min(filters.shape[0], 16)
        grid_rows, grid_cols = 4, 4
        global_vmax = max(s[:, 0].abs().max().item() for s in rec.filter_snapshots)
        upscale = 8  # 3x3 -> 24x24

        for i in range(n_filters):
            r, c = divmod(i, grid_cols)
            pad = 0.02
            w = (1.0 - pad * (grid_cols + 1)) / grid_cols
            h = (1.0 - pad * (grid_rows + 1)) / grid_rows
            x0 = pad + c * (w + pad)
            y0 = 1.0 - pad - (r + 1) * (h + pad) + pad
            inset = ax_filters.inset_axes([x0, y0, w, h])
            f = filters[i, 0].numpy()
            f_up = np.repeat(np.repeat(f, upscale, axis=0), upscale, axis=1)
            inset.imshow(f_up, cmap=FILTER_CMAP, interpolation="nearest",
                        vmin=-global_vmax, vmax=global_vmax)
            inset.axis("off")
            inset.set_facecolor(PANEL_BG)

        # ── Current image + Confidence bars ─────────────────────────────────
        # Cycle through digits: each frame shows a different sample
        snap_idx = min(frame_idx, len(rec.prediction_snapshots) - 1)
        imgs, true_lbls, pred_lbls, confs, probs = rec.prediction_snapshots[snap_idx]
        n_samples = imgs.shape[0]  # 10 digits
        cycle_idx = frame_idx % n_samples

        # Left: show the current digit image large
        ax_cur_img.clear()
        ax_cur_img.axis("off")
        ax_cur_img.set_facecolor(PANEL_BG)
        ax_cur_img.imshow(imgs[cycle_idx, 0].numpy(), cmap=DIGIT_CMAP,
                          interpolation="nearest")
        true_lbl = true_lbls[cycle_idx].item()
        pred_lbl = pred_lbls[cycle_idx].item()
        conf_val = confs[cycle_idx].item()
        correct = true_lbl == pred_lbl
        clr = ACCENT2 if correct else ACCENT3
        ax_cur_img.set_title(f"true: {true_lbl}", color=TEXT_DIM, fontsize=9,
                             fontfamily="monospace", pad=4)
        ax_cur_img.text(0.5, -0.05, f"pred: {pred_lbl}  ({conf_val:.0%})",
                       ha="center", va="top", fontsize=9, color=clr,
                       fontweight="bold", fontfamily="monospace",
                       transform=ax_cur_img.transAxes)

        # Right: confidence bar chart for this digit
        ax_conf.clear()
        ax_conf.set_title("Prediction Confidence", color=TEXT, fontsize=10,
                          fontweight="bold", fontfamily="monospace", pad=8)
        ax_conf.set_facecolor(PANEL_BG)

        p = probs[cycle_idx].numpy()
        bar_colors = []
        for i in range(10):
            if i == true_lbl and i == pred_lbl:
                bar_colors.append(ACCENT2)   # correct prediction = green
            elif i == pred_lbl:
                bar_colors.append(ACCENT3)   # wrong prediction = orange
            elif i == true_lbl:
                bar_colors.append(ACCENT4)   # true label (not predicted) = purple
            else:
                bar_colors.append(ACCENT1)   # other = blue
        ax_conf.barh(range(10), p, color=bar_colors, height=0.7, alpha=0.85)
        ax_conf.set_yticks(range(10))
        ax_conf.set_yticklabels([str(i) for i in range(10)], fontsize=8,
                                 fontfamily="monospace", color=TEXT)
        ax_conf.set_xlim(0, 1)
        ax_conf.set_xlabel("probability", color=TEXT_DIM, fontsize=8)
        ax_conf.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax_conf.spines.values():
            spine.set_color(GRID_CLR)
        ax_conf.invert_yaxis()

        # ── Live predictions strip (all 10 digits) ─────────────────────────
        ax_preds.clear()
        ax_preds.set_title("Live Predictions  (10 fixed test digits)", color=TEXT,
                           fontsize=10, fontweight="bold", fontfamily="monospace", pad=8)
        ax_preds.axis("off")
        ax_preds.set_facecolor(PANEL_BG)

        for i in range(n_samples):
            inset = ax_preds.inset_axes([i / n_samples + 0.005, 0.25,
                                         0.85 / n_samples, 0.65])
            inset.imshow(imgs[i, 0].numpy(), cmap=DIGIT_CMAP, interpolation="nearest")
            inset.axis("off")
            inset.set_facecolor(PANEL_BG)

            is_correct = pred_lbls[i].item() == true_lbls[i].item()
            clr = ACCENT2 if is_correct else ACCENT3
            c = confs[i].item()

            # highlight the currently-cycling digit
            if i == cycle_idx:
                for spine in inset.spines.values():
                    spine.set_visible(True)
                    spine.set_color(ACCENT4)
                    spine.set_linewidth(2)

            inset.set_title(f"{pred_lbls[i].item()} ({c:.0%})",
                           fontsize=7, color=clr, fontweight="bold",
                           fontfamily="monospace", pad=2)
            # show true label below
            inset.text(0.5, -0.12, f"[{true_lbls[i].item()}]",
                      ha="center", va="top", fontsize=6, color=TEXT_DIM,
                      fontfamily="monospace", transform=inset.transAxes)

        return []

    anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                    interval=1000 // FPS, blit=False)

    print(f"Saving to {OUTPUT_FILE} ...")
    writer = animation.FFMpegWriter(fps=FPS, bitrate=4000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    anim.save(OUTPUT_FILE, writer=writer)
    print(f"Done! Saved {OUTPUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    rec = train_and_record()
    render_animation(rec)

import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('results/paper_figures', exist_ok=True)

# -------------------------------------------------------------
# Figure 3: Geometric and alignment analysis (PCA Features)
# -------------------------------------------------------------
def plot_fig3():
    kernels = ['TFK', 'CM-FQK', 'FQK', 'PQK']
    g_values = [486.85, 406.12, 384.76, 327.02]
    
    # Adding RBF baseline for KTA
    kta_kernels = ['TFK', 'CM-FQK', 'FQK', 'PQK', 'RBF']
    kta_values = [0.1825, 0.1897, 0.1840, 0.1475, 0.1309]
    delta_kta = ['+0.052', '+0.059', '+0.053', '+0.017', '']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left subplot: g values
    colors_g = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c']
    bars = ax1.bar(kernels, g_values, color=colors_g, edgecolor='black')
    ax1.set_title('Geometric Difference ($g$)', fontsize=14, pad=15)
    ax1.set_ylabel('$g$ value', fontsize=12)
    ax1.set_ylim(0, 550)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=11)
    
    # Right subplot: KTA values
    colors_kta = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c', '#95a5a6']
    bars2 = ax2.bar(kta_kernels, kta_values, color=colors_kta, edgecolor='black')
    ax2.set_title(r'Task Alignment ($\mathrm{KTA}_c$)', fontsize=14, pad=15)
    ax2.set_ylabel(r'$\mathrm{KTA}_c$', fontsize=12)
    ax2.set_ylim(0, 0.22)
    
    for i, bar in enumerate(bars2):
        # Base KTA text
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=11)
        # Delta KTA annotation (if not RBF)
        if delta_kta[i]:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                     rf"$\Delta$ {delta_kta[i]}", ha='center', va='center', 
                     fontsize=10, color='white', fontweight='bold')
            
    # Draw horizontal line for RBF baseline
    ax2.axhline(y=0.1309, color='#7f8c8d', linestyle='--', zorder=0)

    plt.suptitle('Figure 3: Geometric and Alignment Analysis (Fused PCA)', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig('results/paper_figures/fig3_geometric_alignment.png', dpi=300, bbox_inches='tight')
    print("Saved fig3_geometric_alignment.png")

# -------------------------------------------------------------
# Figure 7: Feature regime comparison (Macro-F1)
# -------------------------------------------------------------
def plot_fig7():
    kernels = ['FQK', 'CM-FQK', 'PQK', 'TFK', 'PIQFM']
    
    # Values from Table II, III, VI
    pca_f1 = [0.184, 0.186, 0.186, 0.184, 0.0]  # PIQFM N/A for PCA
    phys_f1 = [0.469, 0.469, 0.451, 0.439, 0.459]
    
    x = np.arange(len(kernels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out PIQFM for PCA
    rects1 = ax.bar(x[:-1] - width/2, pca_f1[:-1], width, label='PCA Features', color='#3498db', edgecolor='black')
    rects2 = ax.bar(x + width/2, phys_f1, width, label='Physics Features', color='#e67e22', edgecolor='black')
    
    # Add tuned RBF ceiling
    ax.axhline(y=0.467, color='#c0392b', linestyle='--', linewidth=2, label='Tuned Class. Ceiling (0.467)')
    
    ax.set_ylabel('Macro-F1 Score', fontsize=12)
    ax.set_title('Figure 7: Macro-F1 Across Feature Regimes', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, fontsize=11)
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=11)
    
    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                
    plt.tight_layout()
    plt.savefig('results/paper_figures/fig7_feature_regime.png', dpi=300, bbox_inches='tight')
    print("Saved fig7_feature_regime.png")

# -------------------------------------------------------------
# Figure 1: Pipeline Diagram
# -------------------------------------------------------------
def plot_fig1():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    kwargs = dict(ha='center', va='center', fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='#34495e', lw=1.5))
    
    # SAR Path
    ax.text(1.5, 4.5, "SAR\nPatches", **kwargs)
    ax.annotate('', xy=(3.0, 4.5), xytext=(2.2, 4.5), arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))
    ax.text(4.5, 4.5, "Feature Extraction\n(PCA or Physics)", **kwargs)
    ax.annotate('', xy=(6.5, 4.5), xytext=(5.8, 4.5), arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))
    ax.text(8.0, 4.5, "4 Features\n$\\to$ Qubits 0-3", **kwargs)

    # Optical Path
    ax.text(1.5, 2.5, "Optical\nPatches", **kwargs)
    ax.annotate('', xy=(3.0, 2.5), xytext=(2.2, 2.5), arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))
    ax.text(4.5, 2.5, "Feature Extraction\n(PCA or Physics)", **kwargs)
    ax.annotate('', xy=(6.5, 2.5), xytext=(5.8, 2.5), arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))
    ax.text(8.0, 2.5, "4 Features\n$\\to$ Qubits 4-7", **kwargs)
    
    # CM-FQK Box
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((6.8, 1.8), 2.4, 3.2, boxstyle="round,pad=0.2", alpha=0.1, color='#e74c3c', zorder=-1)
    ax.add_patch(rect)
    ax.text(8.0, 1.0, "CM-FQK:\ncross-modal ZZ\non top-3 MI pairs", ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8', edgecolor='#c0392b', lw=1.5))
    
    # Dotted line linking them to the CM-FQK box
    ax.annotate('', xy=(8.0, 1.6), xytext=(8.0, 2.2), arrowprops=dict(arrowstyle='-', linestyle='--', lw=1.5, color='#c0392b'))

    plt.suptitle('Figure 1: End-to-end Pipeline', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('results/paper_figures/fig1_pipeline.png', dpi=300, bbox_inches='tight')
    print("Saved fig1_pipeline.png")


# -------------------------------------------------------------
# Figure 2: Circuit Diagrams
# -------------------------------------------------------------
def plot_fig2():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, 8)
        ax.axis('off')
        # Draw 8 qubit lines
        for i in range(8):
            ax.hlines(i, 0.5, 9.5, color='gray', lw=1)
            ax.text(0.2, i, f"$q_{i}$", va='center', ha='center', fontsize=12)

    # Function to draw basic gates
    def draw_rep(ax, start_x, draw_cross_modal=False):
        # Hadamard
        for i in range(8):
            ax.add_patch(plt.Rectangle((start_x, i-0.3), 0.5, 0.6, facecolor='#aed6f1', edgecolor='black'))
            ax.text(start_x+0.25, i, 'H', va='center', ha='center', fontsize=10)
        
        # Rz
        for i in range(8):
            ax.add_patch(plt.Rectangle((start_x+0.8, i-0.3), 0.8, 0.6, facecolor='#d7bde2', edgecolor='black'))
            ax.text(start_x+1.2, i, r'$R_z$', va='center', ha='center', fontsize=9)
            
        # ZZ (nearest neighbour)
        for i in range(7):
            x_pos = start_x + 1.9 + (i * 0.3)
            # ZZ vertical line
            ax.vlines(x_pos, i, i+1, color='#34495e', lw=2)
            # control dots
            ax.plot([x_pos, x_pos], [i, i+1], 'o', color='#34495e', markersize=6)

        # Repetition block indication
        ax.axvspan(start_x-0.2, start_x + 3.8, alpha=0.1, color='gray', zorder=-1)

        # CM-FQK (Cross-modal ZZ pairs)
        if draw_cross_modal:
            # Example top-3 pairs (e.g. 0-4, 1-6, 3-7)
            pairs = [(0, 4), (1, 6), (3, 7)]
            for idx, (q1, q2) in enumerate(pairs):
                x_pos = start_x + 4.5 + (idx * 0.5)
                ax.vlines(x_pos, q1, q2, color='#e74c3c', lw=2, linestyle='--')
                ax.plot([x_pos, x_pos], [q1, q2], 'o', color='#e74c3c', markersize=6)
                ax.text(x_pos+0.1, (q1+q2)/2, 'ZZ', va='center', ha='left', fontsize=8, color='#e74c3c')

    # Top diagram: Baseline FQK (2 reps)
    ax1.set_title("Base Architecture: FQK / TFK / PIQFM", fontsize=14, fontweight='bold')
    draw_rep(ax1, 1)
    ax1.text(6.0, 3.5, "$\\times$ 2 Repetitions", fontsize=14, bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9e79f', edgecolor='black'))
    
    # Bottom diagram: CM-FQK (2 reps)
    ax2.set_title("Cross-Modal Architecture: CM-FQK", fontsize=14, fontweight='bold')
    draw_rep(ax2, 1, draw_cross_modal=True)
    ax2.text(7.5, 3.5, "$\\times$ 2 Repetitions", fontsize=14, bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9e79f', edgecolor='black'))

    plt.tight_layout()
    plt.savefig('results/paper_figures/fig2_circuit.png', dpi=300, bbox_inches='tight')
    print("Saved fig2_circuit.png")


if __name__ == '__main__':
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig7()

import os
import zipfile
import shutil

def create_overleaf_zip():
    target_files = [
        ("main.tex", "main.tex"),
        ("figures/fig1_system_architecture.png", "figures/fig1_system_architecture.png"),
        ("figures/fig2_network_architecture.png", "figures/fig2_network_architecture.png"),
        ("figures/fig3_training_dynamics.png", "figures/fig3_training_dynamics.png"),
        ("figures/fig4_augmented_canopy.png", "figures/fig4_augmented_canopy.png"),
        ("figures/fig5_graph_extraction_gap_bridging.png", "figures/fig5_graph_extraction_gap_bridging.png"),
        ("figures/fig6_centrality_resilience.png", "figures/fig6_centrality_resilience.png"),
        ("figures/fig7_ablation_study.png", "figures/fig7_ablation_study.png"),
        ("figures/fig8_qualitative_comparison.png", "figures/fig8_qualitative_comparison.png"),
    ]

    zip_destinations = [
        "overleaf_paper.zip",
        "docs/overleaf_paper.zip"
    ]

    for zip_path in zip_destinations:
        os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for src, arcname in target_files:
                if not os.path.exists(src):
                    raise FileNotFoundError(f"Source file missing: {src}")
                z.write(src, arcname)
                print(f"[{zip_path}] Added {arcname} ({os.path.getsize(src):,} bytes)")

        # Verify integrity
        with zipfile.ZipFile(zip_path, 'r') as z:
            test_res = z.testzip()
            if test_res is not None:
                raise ValueError(f"Corrupt file in zip: {test_res}")
            print(f"Verified {zip_path}: size = {os.path.getsize(zip_path):,} bytes, all files intact.\n")

if __name__ == '__main__':
    create_overleaf_zip()

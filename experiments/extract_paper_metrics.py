import os
import json
import pandas as pd

results_dir = "results"

def load_json(path):
    fullpath = os.path.join(results_dir, path)
    if os.path.exists(fullpath):
        with open(fullpath, 'r') as f:
            return json.load(f)
    return None

def main():
    print("=== EXTRACTED METRICS CROSS-CHECK ===")
    
    # 1. Geometric & Alignment (Table II)
    print("\n--- Geometric & Alignment (PCA) ---")
    geom_csv = os.path.join(results_dir, "geometric_difference/summary_table.csv")
    if os.path.exists(geom_csv):
        df = pd.read_csv(geom_csv)
        print(df.to_string())
    else:
        print("geometric_difference/summary_table.csv not found.")

    # Classification (Table III) - PCA
    print("\n--- Classification (PCA) ---")
    class_pca = load_json("classification/exp2_results.json")
    if class_pca:
        for k, v in class_pca.items():
            if isinstance(v, dict) and 'macro_f1' in v:
                print(f"{k}: F1={v['macro_f1']:.3f}, Acc={v.get('accuracy', 0)*100:.1f}%")
            elif k == "metrics":
                for mk, mv in v.items():
                    print(f"{mk}: F1={mv.get('macro_f1')}")
    else:
        print("classification/exp2_results.json not found.")

    # 2. Controlled MI (Table IV)
    print("\n--- Controlled MI (Table IV) ---")
    cmi_csv = os.path.join(results_dir, "controlled_mi/controlled_mi_summary.csv")
    if os.path.exists(cmi_csv):
        df2 = pd.read_csv(cmi_csv)
        print(df2.to_string())
    else:
        print("controlled_mi_summary.csv not found.")
        
    # 3. Dequantization (Table V)
    print("\n--- Dequantization (Table V) ---")
    deq = load_json("dequantization/exp6_results.json")
    if deq:
        if "fqk" in deq:
            print(f"FQK: F1={deq['fqk'].get('macro_f1')}, KTA={deq['fqk'].get('kta')}")
        for d in ['64', '128', '256', '512']:
            if d in deq:
                print(f"RFF {d}: F1={deq[d].get('macro_f1')}, KTA={deq[d].get('kta')}, FrobErr={deq[d].get('frobenius_error')}")
    else:
        print("dequantization/exp6_results.json not found.")

    # 4. Physics (Table VI)
    print("\n--- Physics (Table VI) ---")
    phys_csv = os.path.join(results_dir, "physics/physics_summary.csv")
    if os.path.exists(phys_csv):
        df_phys = pd.read_csv(phys_csv)
        print(df_phys.to_string())
    else:
        print("physics_summary.csv not found.")
        
if __name__ == '__main__':
    main()

from pathlib import Path
import ray
import scanpy as sc

from expression_copilot import ExpressionCopilotModel


@ray.remote(num_cpus=2, num_gpus=0.25, max_retries=1)
def ray_mi(model:ExpressionCopilotModel, save_path, method='mlp'):
    sc.pp.filter_genes(model.adata, min_counts=6)
    sc.pp.scale(model.adata, max_value=10)
    _ = model.calc_metrics_per_gene()
    _, _ = model.calc_baseline_metrics(method=method)
    model.save_results(save_path)

# adata = sc.read_h5ad('./data/human_adata_with_tile_emb.h5ad')
adata_marker = sc.read_h5ad('./results/human_adata_marker_genes_log1p.h5ad')

ray_results = []
Path('gene_marker').mkdir(exist_ok=True)
ray.init(ignore_reinit_error=True)
for slide_name in adata_marker.obs['__slide'].unique():
    sub_adata_marker = adata_marker[adata_marker.obs['__slide'] == slide_name].copy()
    model = ExpressionCopilotModel(sub_adata_marker.copy(), image_key='X_uni')

    save_path = Path(f'gene_marker/{slide_name}.pkl').resolve()
    if save_path.exists():
        print(f"{save_path} already processed, skip ...")
        continue
    ray_results.append(ray_mi.remote(model, save_path))
ray.get(ray_results)
ray.shutdown()

from pathlib import Path
import ray
import scanpy as sc
from expression_copilot import ExpressionCopilotModel


@ray.remote(num_cpus=2, num_gpus=0.25, max_retries=1)
def ray_mi(model: ExpressionCopilotModel, save_path, method):
    _ = model.calc_metrics_per_gene()
    _, _ = model.calc_baseline_metrics(method=method)
    model.save_results(save_path)

adata = sc.read_h5ad('./data/human_adata_with_tile_emb.h5ad')

ray_results = []
ray.init(ignore_reinit_error=True)
for slide_name in adata.obs['__slide'].unique():
    sub_adata = adata[adata.obs['__slide'] == slide_name].copy()
    model = ExpressionCopilotModel(sub_adata, 'X_uni')
    for method in ['mlp', 'linear', 'ridge', 'ensemble']:
        save_path = Path(f'hest-method:{method}/{slide_name}.pkl').resolve()
        if save_path.exists():
            print(f"{save_path} already processed, skip ...")
            continue
        ray_results.append(ray_mi.remote(model, save_path, method))
ray.get(ray_results)
ray.shutdown()

from cross_domain_pipeline import PipelineConfig, BenchmarkRunner, prepare_windows

cfg = PipelineConfig.from_yaml("configs/base.yaml")
manifest = prepare_windows(cfg)
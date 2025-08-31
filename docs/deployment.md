# Deployment Guide

This guide outlines best practices for deploying **tscv-vision** in production environments.

## Model versioning

Use `ModelRegistry` to record encoder versions, performance metrics, and deployment status.

```python
from tscv_vision.mlops import ModelRegistry

registry = ModelRegistry()
registry.register("gaf", "1.0", {"acc": 0.92})
registry.set_status("gaf", "1.0", "deployed")
```

## A/B testing

`ABTester` compares feature extractors or configurations in production.

```python
from tscv_vision.mlops import ABTester

tester = ABTester()
tester.add("A", 0.6)
tester.add("B", 0.7)
print(tester.compare())
```

## Monitoring

`create_monitoring_app` exposes FastAPI endpoints with optional Prometheus metrics for health checks and drift detection.

```python
from tscv_vision.mlops import DriftDetector, create_monitoring_app

app = create_monitoring_app(DriftDetector())
```

## Graceful degradation

`safe_encode` falls back to an alternate encoder if the primary one fails or exceeds a latency threshold.

```python
from tscv_vision.mlops import safe_encode

encoded = safe_encode(series, primary_encoder, backup_encoder, timeout=0.05)
```

## Batch processing

`batch_process` handles large datasets efficiently with progress callbacks and resumption support.

```python
from tscv_vision.mlops import batch_process

features = batch_process(data, extractor, batch_size=128, start=0)
```

## Docker deployment

A sample `Dockerfile` is provided at the project root:

```bash
docker build -t tscv-vision .
```

## Kubernetes deployment

A basic Kubernetes manifest lives under `k8s/feature-service.yaml`:

```bash
kubectl apply -f k8s/feature-service.yaml
```


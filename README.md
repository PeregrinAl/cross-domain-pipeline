# cross-domain-pipeline

MVP: Source-Free Domain Adaptation for Anomaly Recognition in Non-Stationary Signals

Цель MVP
- source domain и target domain;
- source-only деградирует;
- source-free adaptation частично восстанавливает качество;
- двухветвевое представление устойчивее, чем одно.

Текущая архитектура MVP

- raw temporal window;
- STFT или CWT representation;
- source-only detector;
- source-free adaptation;
- comparison.

Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Первый запуск

```bash
python -m src.train --config configs/base.yaml
```


Статус
Day 1: project skeleton, config, data interface, windowing, smoke test
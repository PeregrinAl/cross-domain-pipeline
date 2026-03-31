
# Cross-Domain Anomaly Recognition MVP

MVP для междоменного распознавания аномалий в нестационарных сигналах с использованием:

- двух представлений сигнала:
  - raw temporal window
  - time-frequency representation на основе STFT
- source-only baseline
- source-free domain adaptation
- сравнения single-view и fused представлений

## Цель

Цель этого MVP — показать в рамках одного воспроизводимого экспериментального пайплайна, что:

1. source-only модель распознавания аномалий деградирует при доменном сдвиге,
2. source-free адаптация на целевом домене может частично восстановить качество на target без доступа к исходным source-данным во время адаптации,
3. fused-представления на основе нескольких view являются осмысленным кандидатом для adaptation, даже если до адаптации они могут быть более чувствительны к доменному сдвигу.

## Текущий статус

Реализовано:

- skeleton проекта и воспроизводимый pipeline, управляемый через config
- подготовка данных и сегментация окнами
- raw и STFT представления
- source-only baselines:
  - `raw_only`
  - `tfr_only`
  - `fused`
- оценка на target-domain с калиброванным threshold
- минимальная source-free адаптация для fused-модели
- вариант адаптации с consistency filtering
- итоговая ablation summary

## Основной экспериментальный результат

### Source-only baselines

| Эксперимент | Source ROC-AUC | Target ROC-AUC | Source PR-AUC | Target PR-AUC |
|---|---:|---:|---:|---:|
| raw_only | 0.5781 | 0.6588 | 0.6630 | 0.6932 |
| tfr_only | 0.6969 | 0.6531 | 0.7505 | 0.7228 |
| fused | 0.6869 | 0.4700 | 0.7136 | 0.6094 |

### Результаты адаптации для fused-модели

| Эксперимент | Source ROC-AUC | Target ROC-AUC | Source PR-AUC | Target PR-AUC |
|---|---:|---:|---:|---:|
| fused source_only | 0.6869 | 0.4700 | 0.7136 | 0.6094 |
| fused + sfda_minimal | 0.6525 | 0.6200 | 0.7548 | 0.7185 |
| fused + sfda_consistency | 0.6519 | 0.6169 | 0.7541 | 0.7173 |

## Интерпретация результатов

Текущий MVP позволяет сделать следующие выводы:

- В source-only постановке fused two-view модель показывает хорошее качество на source domain, но одновременно демонстрирует наиболее сильную деградацию на target domain.
- Минимальная source-free адаптация заметно улучшает качество ранжирования на target для fused-модели:
  - target ROC-AUC улучшается с **0.4700** до **0.6200**
  - target PR-AUC улучшается с **0.6094** до **0.7185**
- В текущем toy setup добавление cross-view consistency filtering не дает дополнительного измеримого выигрыша по сравнению с более простой adaptation loop.

## Экспериментальный пайплайн

Реализованный пайплайн:

`Preprocessing -> Windowing -> Raw/STFT views -> Source-only baseline -> Source-free adaptation -> Ablation summary`

### Source-only baseline

Source-only baseline включает:

- `RawEncoder`: 1D CNN для raw-окон
- `TFREncoder`: 2D CNN для STFT-входов
- `FusionEncoder`: конкатенацию branch embeddings с последующей fusion head
- `SourceOnlyClassifier`: supervised anomaly classifier, обучаемый на source labels

### Source-free adaptation

Минимальная adaptation loop:

1. загружает обученную fused source-only модель,
2. вычисляет source normal prototype,
3. выбирает target-окна с низким score как pseudo-normal samples,
4. дообучает fusion head на target adaptation data,
5. оценивает качество до и после адаптации.

## Структура проекта

```text
cross-domain-pipeline/
├─ configs/
│  └─ base.yaml
├─ data/
│  ├─ raw/
│  │  ├─ records.csv
│  │  └─ *.npy
│  └─ processed/
│     ├─ manifest.csv
│     └─ windows/
├─ experiments/
│  ├─ source_only_supervised/
│  │  ├─ raw_only/
│  │  ├─ tfr_only/
│  │  └─ fused/
│  ├─ day5_evaluation/
│  ├─ day6_sfda/
│  ├─ day8_ablation/
│  └─ ablation_inputs/
├─ notebooks/
├─ src/
│  ├─ data/
│  │  ├─ dataset.py
│  │  ├─ dataloader.py
│  │  ├─ transforms.py
│  │  └─ windowing.py
│  ├─ models/
│  │  ├─ raw_encoder.py
│  │  ├─ tfr_encoder.py
│  │  ├─ fusion_model.py
│  │  └─ source_only_classifier.py
│  ├─ utils/
│  │  ├─ metrics.py
│  │  └─ seed.py
│  ├─ prepare_dummy_records.py
│  ├─ prepare_data.py
│  ├─ train.py
│  ├─ train_source_only.py
│  ├─ evaluate_source_only.py
│  ├─ day5_evaluate.py
│  ├─ adapt_source_free.py
│  └─ day8_ablation_summary.py
├─ requirements.txt
└─ README.md
```

## Протокол данных

В текущем репозитории используется воспроизводимый toy protocol на основе сгенерированных `.npy` сигналов.

### Splits

* `source/train`: source-domain training signals
* `source/val`: source-domain validation signals
* `target/test`: target-domain evaluation signals
* `target/adapt`: unlabeled target-domain signals, используемые для source-free adaptation

### Формат processed manifest

Каждая строка в `data/processed/manifest.csv` содержит метаданные одного окна:

* `path`
* `label`
* `domain`
* `record_id`
* `split`
* `window_idx`
* `start`
* `end`

## Установка

Создай виртуальное окружение и установи зависимости.

### Windows

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Быстрый старт

### 1. Подготовить dummy raw signals

```bash
python -m src.prepare_dummy_records
```

### 2. Построить windowed dataset и manifest

```bash
python -m src.prepare_data
```

### 3. Запустить smoke test

```bash
python -m src.train
```

### 4. Обучить source-only baselines

```bash
python -m src.train_source_only --variant raw_only
python -m src.train_source_only --variant tfr_only
python -m src.train_source_only --variant fused
```

### 5. Посчитать метрики Day 5

```bash
python -m src.day5_evaluate
```

### 6. Запустить source-free adaptation для fused-модели

```bash
python -m src.adapt_source_free
```

### 7. Построить Day 8 ablation summary

```bash
python -m src.day8_ablation_summary
```

## Основные выходные файлы

### Source-only training

Сохраняются в:

```text
experiments/source_only_supervised/<variant>/
```

Содержат:

* `best.pt`
* `history.csv`
* `source_val_scores.csv`
* `target_test_scores.csv`
* `summary.json`

### Day 5 evaluation

Сохраняется в:

```text
experiments/day5_evaluation/
```

Содержит:

* `day5_summary.csv`
* графики распределения score
* ROC curves
* PR curves

### Day 6 adaptation

Сохраняется в:

```text
experiments/day6_sfda/fused/
```

Содержит:

* adapted checkpoint
* adaptation history
* before/after score files
* `summary.json`

### Day 8 ablation

Сохраняется в:

```text
experiments/day8_ablation/
```

Содержит:

* `ablation_summary.csv`
* bar plots для target ROC-AUC и PR-AUC
* source-vs-target scatter plot
* before/after fused plots

## Ограничения

У текущего MVP есть следующие ограничения:

* Используется toy synthetic protocol, а не полноценный реальный benchmark.
* Метка окна наследуется от метки исходного сигнала.
* Threshold-based метрики чувствительны к калибровке score после adaptation.
* Cross-view consistency filtering не показал дополнительной пользы в текущем setup.
* Текущая adaptation loop намеренно минималистична и пока не включает более продвинутую pseudo-label filtering или uncertainty estimation.

## Следующие шаги

Планируемые следующие шаги:

* увеличить объем и разнообразие target adaptation split
* протестировать более сильные и разнообразные типы domain shift
* добавить event-level evaluation
* улучшить score calibration после adaptation
* при желании протестировать CWT или wavelet multi-scale branch как stretch goal
* подготовить компактное demo и слайды

## Глоссарий

**Anomaly recognition**
Обнаружение аномального или ненормального поведения сигнала.

**Source domain**
Домен, на котором модель изначально обучается.

**Target domain**
Другой домен, в котором присутствует distribution shift и на котором оценивается transfer performance.

**Domain shift**
Изменение статистики сигнала, шума, тренда, масштаба или частотной структуры между source и target.

**Source-only model**
Модель, обученная только на source-domain data и применяемая к target domain без adaptation.

**Source-free adaptation**
Адаптация на target domain без доступа к исходным source training samples.

**Non-stationary signal**
Сигнал, статистические или спектральные свойства которого меняются во времени.

**Windowing**
Разбиение длинного сигнала на короткие сегменты фиксированной длины.

**Raw view**
Исходное временное окно сигнала, используемое напрямую как вход модели.

**TFR**
Time-frequency representation. В этом проекте сейчас реализовано как STFT magnitude.

**STFT**
Short-Time Fourier Transform. Временнó-частотное преобразование, используемое для представления локального спектрального содержимого.

**Embedding**
Обучаемый вектор признаков, производимый encoder.

**Fusion**
Объединение нескольких branch embeddings в одно совместное представление.

**Prototype**
Репрезентативный вектор, обычно средний embedding normal source samples.

**Pseudo-normal sample**
Target-domain sample, выбранный как вероятно normal на основе confidence или score filtering.

**Ranking metrics**
Метрики, основанные на порядке score, а не на одном фиксированном threshold, например ROC-AUC и PR-AUC.

**ROC-AUC**
Площадь под ROC-кривой. Оценивает качество ранжирования по всем threshold.

**PR-AUC**
Площадь под precision-recall curve. Особенно полезна для anomaly или imbalanced settings.

**F1 score**
Threshold-based гармоническое среднее precision и recall.

**Consistency filtering**
Стратегия отбора pseudo-labels, которая сохраняет только те target samples, для которых несколько view или branches согласны между собой.

## Замечание о воспроизводимости

Все эксперименты предполагается запускать из одного config file:

```text
configs/base.yaml
```

Структура репозитория организована так, чтобы вся экспериментальная история была воспроизводима:

1. подготовить данные,
2. обучить source-only baselines,
3. оценить деградацию на target,
4. запустить source-free adaptation,
5. собрать итоговые ablations.

# NEXUS-X — Python / CPU research prototype

**Рабочий первый этап**, не готовая LLM и не заявление о превосходстве над
Transformer. Модель обучается с нуля: никаких pretrained weights, внешнего LLM API
или скрытого inference engine.

```
UTF-8 bytes → Encoder + position features → causal low-rank shared mixer
           → learned top-k router → sparse experts → byte decoder
```

Вычислительный router расположен **после** shared core: он выбирает экспертов,
но пока не управляет глубиной core. Это явное ограничение первого прототипа.

- [Математика, контракты, ограничения](docs/architecture.md)
- [Отчёт этапа: все 14 пунктов](docs/stage1_report.md)
- [Измерения benchmark](docs/stage1_benchmark.json)
- [Обучение и fingerprint датасетов](docs/stage1_training.json)

## Быстрый запуск

Python **3.11+**, CPU. Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python nexus.py
```

Windows PowerShell: вместо `source` используйте `.venv\Scripts\Activate.ps1`.
Код использует переносимые NumPy/psutil/pathlib; Windows/macOS в этой сессии
**не тестировались**. Из root проекта работает и прямой `.venv/bin/python nexus.py`.

### Создать и обучить

```bash
python nexus.py create --config nexus_config.yaml
```

Default — небольшой smoke-run: два датасета text/code, 60 обновлений, параметры
20 355, контекст 64 байтовых токена. Запишет `model.nexus/`, промежуточный
`model_checkpoints/step-00000030.nexus/` и `runs/metrics.jsonl`.

**Существующие модели никогда не перезаписываются.** Для повторного запуска:

```bash
python nexus.py create --output runs/another.nexus
```

### Дообучить

```bash
python nexus.py finetune --model model.nexus --config ft_config.yaml
```

Пример использует instruction loss только на response и замораживает
`encoder`, `position`, `core`. Результат — `finetuned.nexus/`, база не меняется.
`finetune.mode`: `full`, `instruction`, `expert`, `router`.
`freeze`: проверяемые группы `encoder`, `position`, `core`, `router`, `experts`, `decoder`.
Веса архитектуры берутся из модели, не из размеров fine-tune YAML.
Для новых данных создаётся новый AdamW и новый schedule.

### Использовать

```bash
python nexus.py use --model model.nexus --prompt "NEXUS is" --max-new-tokens 32
python nexus.py use --model finetuned.nexus
```

Без `--prompt` — независимые интерактивные запросы, `/exit` завершает диалог.
Это **не conversational memory**. `--temperature 0` включает greedy decoding.
stdout one-shot режима — JSON с ответом, token IDs и метриками; логи — stderr.
Неполные/невалидные UTF-8 последовательности декодируются через replacement.
При `measure_effect: true` первый запрос отдельно измеряет turbo перед генерацией;
его накладные расходы не включены в `tokens_per_second` генерации.

### Benchmark / status

```bash
python nexus.py bench --model model.nexus --output runs/benchmark.json
python nexus.py status --model model.nexus
python nexus.py status
```

Benchmark: autoregressive generation (2 warmup + 7 measured runs), paired full
forward turbo/reference (2 warmup + 7 измерений, чередование порядка), проверка
logits и validation loss. Генерация измеряется на prompt `NEXUS`, greedy, до 32
новых токенов. `tokens/sec` — **байтовые**, не BPE/word tokens. Для tiny модели
Python dispatch и overhead значительны. Сохраняются сырые latency samples.

Контроль turbo — `einsum(optimize=False)`, а не оптимизированная сторонняя модель.
Ускорение может быть отрицательным: код не скрывает регрессии. RSS — snapshot
процесса, не peak RAM. CPU% = CPU seconds / wall seconds × 100; может быть >100
из-за native/background threads, особенно в коротком непрогретом запуске.
Deterministic mode ограничивает BLAS одним потоком. `threads` применяется при
`deterministic: false`; auto ограничен максимум четырьмя логическими CPU.

#### Необязательный Transformer-контроль

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python nexus.py bench --model model.nexus --config nexus_config.yaml --baseline
```

Отдельный torch Transformer **обучается с нуля** на тех же draws датасетов,
числе шагов, width/depth/context, CPU threads и schedule. Это контроль по шагам,
**не** по параметрам/FLOPs/времени. Используйте полный create-checkpoint и его
исходные training config/seed. Baseline не участвует в работе NEXUS.
В данной среде CPU torch установить не удалось из-за TLS; результат сравнения
**не измерен**, optional test пропущен. Отсутствующий torch не мешает основным режимам.

### Возобновление без изменения траектории обучения

```bash
python nexus.py create --max-steps 20 --output runs/part.nexus
python nexus.py create --resume --model runs/part.nexus --output runs/resumed.nexus
```

`--max-steps` ограничивает шаги **этого запуска**, не меняя исходный schedule.
Конфиг training и содержимое датасетов должны оставаться теми же. Checkpoint
содержит AdamW moments, RNG, cursor каждого источника и shuffle buffers.
Resume читает источник заново до cursor, поэтому старт — O(consumed), не O(1).
`finetune --resume --model ... --output ... --config ft_config.yaml` аналогичен:
сохраняйте mode/freeze/datasets, а новый output передавайте через CLI.
После Ctrl+C доступен последний **завершённый** checkpoint, не обязательно
последний шаг. При изменении датасетов/расписания используйте finetune, не resume.

## Датасеты и конфиг

Все относительные пути внутри YAML разрешаются относительно файла YAML;
CLI `--model/--output` — относительно текущей директории.
Неизвестные/неподдерживаемые ключи отклоняются, а не игнорируются.
CLI или выбор меню определяет фактический режим; поле YAML `mode` — метаданные,
не команда на автоматический запуск при открытии меню.

`training.datasets` / `finetune.datasets` — список с `path`, положительным `weight`,
`modality: text|code`, `format: auto|txt|jsonl`, необязательным `limit` документов.
Пример JSONL:

```json
{"text": "A document to learn from."}
{"prompt": "Say hello.", "response": "Hello!"}
```

TXT: одна строка = один документ. JSONL: одна запись = один документ.
Форматы определяются по расширению. Нет загрузки всего корпуса в RAM: память
сэмплера пропорциональна shuffle buffer и размеру текущего документа.
`steps_per_epoch` задаёт **бюджет обновлений**, а не однократный обход корпуса;
источники циклически повторяются и выбираются по весам. Качество validation —
воспроизводимая выборка с повторениями, не полный проход по файлам.
Пользователь отвечает за отсутствие leakage между train и validation.

## Код и зависимости

```
nexus.py                  меню и CLI, единственная пользовательская точка входа
nexus_config.yaml         default create/bench config
ft_config.yaml            instruction-tuning example
nexus/orchestrator.py     общий программный API для всех режимов
nexus/context.py          config + model + RNG + backend + state + metrics
nexus/tensor.py           собственный reverse-mode autodiff
nexus/{encoder,position,attention,core,router,experts,decoder}.py
nexus/turbo/              BLAS thread scopes, reference kernels, profiler, int8 codec
nexus/training/           streaming mixture, AdamW, schedule, engine, checkpoint
nexus/format/             versioned SafeTensors directories
nexus/benchmarks/         paired speed + optional torch baseline + JSON report
nexus/tests/              numerical/unit/integration tests
```

Core dependencies — NumPy, PyYAML, SafeTensors, threadpoolctl, psutil.
NumPy предоставляет native массивы/BLAS, **не** готовую архитектуру или autograd.
Основной neural/training код полностью Python. Torch — только optional baseline.
Программно используйте `Orchestrator(Config.load(...))`: те же `create`,
`finetune`, `load`, `use`, `bench`, `status`, без альтернативной логики.
Один context не предназначен для конкурентной генерации: у него mutable RNG/state.
Pool scopes между независимыми context сериализованы, потому что BLAS pools
process-global. Это не work-stealing runtime.

### Формат

`.nexus/`: manifest + hashes, tokenizer/core/router/expert safetensors,
metadata.json и optional optimizer.safetensors. Запись в sibling temporary dir,
затем rename на **новое** имя. Это transactional publish, не fsync-гарантия при
аварии питания. Metadata JSON не исполняется, pickle не используется;
пути manifest проверяются. SHA256 обнаруживает повреждение, **не** доказывает
подлинность модели. Загружаются все веса, lazy mmap и expert unloading ещё нет.
Модели/логи игнорируются Git, демонстрационные тексты и отчёты включены.

## Проверки

```bash
pip install pytest
python -m pytest -q
python -m compileall -q nexus.py nexus
pip check
```

Проверяются: finite-difference gradients, broadcasting/index/scatter/cumsum,
causality, top-k/skip экспертов, top-1 gradient limitation, loss reduction,
AdamW clipping, exact resume с grad accumulation, freeze modes, sampler weights,
instruction masks, UTF-8, serialization/corruption/path traversal, int8 error
bound, inference memory retention, thread scopes, config, CLI и все три режима меню.
В зафиксированном запуске: **48 passed, 1 skipped** (optional torch baseline).

## Чего нет на этом этапе

Нет early exit/adaptive depth, L0–L5 memory, Think/reasoning/verification,
семантических экспертов, continual replay, speculative decoding, квантованного
inference, mmap/prefetch/pinning/NUMA/custom SIMD, мультимодальности, GUI,
HTTP/plugins/tools/Telegram/DLL. Search пункт меню явно обозначен будущим этапом;
`search` CLI возвращает ошибку, а не фиктивные результаты.

Обучение на предоставленных файлах **не делает модель способной отвечать на
вопросы**: генерация пока бессвязна и повторяется. Нужны серьёзные данные,
контролируемые ablation, несколько seeds и matched-budget baselines, прежде чем
делать вывод об intelligence per computation.

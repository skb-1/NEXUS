# NEXUS-X v0.1: причинный sparse summary mixer

Это экспериментальный **первый этап**, не доказанно новая в литературе архитектура.
Готовые LLM, pretrained embeddings и Transformer в основном ядре не используются.
NumPy отвечает за массивы/BLAS, собственный reverse-mode engine — за градиенты.

## Математика

Байтовый словарь: 0…255 UTF-8, PAD=256, BOS=257, EOS=258, vocab=259.
Для batch B, последовательности T, ширины D и ранга R:

`h_t = E[x_t] + phi(t) P`, где `phi(t) = [t/C, (t/C)^2, sin(t/s_i), cos(t/s_i)]`.
Две частоты s=(1,16), всего шесть фиксированных признаков. Позиции сбрасываются
при обрезке контекста; отдельной системы дальних абсолютных позиций пока нет.

Для каждого слоя l (параметры общие для всех экспертов):

```
z_t = RMSNorm(h_t)
local_t = mean(z_max(0,t-W+1) ... z_t)
summary_t = (sum_(j=0)^t z_j)/(t+1)
u_t = tanh((z_t + a_l * local_t + b_l * summary_t) A_l) B_l
h_t = h_t + u_t / sqrt(depth)
```

`A_l: D×R`, `B_l: R×D`, a,b — обучаемые векторы D. Нет Q/K/V,
попарных attention scores и квадратичной матрицы T×T. Локальный и глобальный
префиксы вычисляются cumsum; стоимость O(BTDR), память O(BTD) на слой.
Это простое lossy сжатие истории, не доказанный механизм reasoning/retrieval.

После core выполняется разреженная диспетчеризация:

```
s_t = RMSNorm(h_t) W_router
I_t = stable_top_k(s_t)
p_t = softmax(s_t)
g_te = p_te / sum_(j in I_t) p_tj
f_e(h) = tanh(RMSNorm(h) U_e) V_e
h'_t = h_t + sum_(e in I_t) g_te f_e(h_t)
logits_t = RMSNorm(h'_t) W_out + bias
```

Каждый эксперт получает **только выбранные строки**, остальные expert matmul
не вызываются. Матрица router остаётся плотной O(BTDE). Градиент проходит
через выбранные gates и экспертов, но не через дискретный top-k.
При k=1 gate равен единице: task-loss не обучает router! Поэтому default k=2.
Для всех k добавляется Switch-подобный auxiliary loss:
`lambda * E * sum_e mean(p_e) * stopgrad(fraction_selected_e)`.
Это эвристика балансировки, не семантическая специализация экспертов.

Loss — masked next-byte cross-entropy + auxiliary balance loss. AdamW,
cosine decay, global gradient clipping, accumulation по microbatch.
Каждый microbatch усредняется по своим непустым target; accumulation усредняет
microbatch, а не все токены совокупно. Padding не вносит вклад в task loss и router
balance. Данные — независимые документы/чанки с BOS/EOS; состояния между ними нет.

## Активные параметры

`total = sum(size(parameters))`.
`shared = total - E*(2DR)`.
Для одного токена оцениваем `shared + k*(2DR)`; для forward batch/контекста:
`shared + |union(selected experts)|*(2DR)`. Encoder conservatively counted in full.
Это число параметров потенциально затронутых веткой, **не FLOPs** и не RAM.
Все веса v0.1 резидентны: sparse compute не означает sparse storage.
На длинном prompt объединение обычно включает всех экспертов.

## NEXUS-TURBO v0.1

Один backend у каждого Tensor: fp32 NumPy ufunc и GEMM в native BLAS, GIL
освобождается библиотекой в численных kernels; Python graph/dispatch остаются
под GIL. Threadpoolctl ограничивает native pools на время операции оркестратора.
В deterministic mode один BLAS-поток + seeded Generator; побитовая переносимость
между разными BLAS/CPU не обещается. SIMD выбирает установленная сборка NumPy,
своих AVX/NEON kernels, pinning, NUMA, work-stealing и prefetch пока нет.

Контрольный путь без turbo использует `einsum(optimize=False)` для 2D matmul;
включённый — BLAS `matmul`. Benchmark измеряет **оба полных forward** с одинаковыми
весами/входом, warmup, median, проверкой logits. Такой контроль не доказывает
преимущество над уже оптимизированным NumPy/PyTorch. RSS — память процесса, CPU%
— process CPU time / wall time (может быть >100). Энергия, cache misses и IPC не
измерены. Safetensors загружается безопасно в резидентные массивы, не lazy mmap.
Int8 per-row codec — отдельно тестируемый storage primitive, не inference kernel.

## Контекст и данные

RuntimeContext владеет Config, backend, RNG, model, logger, state и metrics.
Orchestrator — общий API, nexus.py — пользовательский CLI/menu; никакой второй
логики обучения. Датасеты — воспроизводимая weighted random mixture text/code,
TXT/JSONL с ограниченным shuffle buffer. Sampler checkpoint хранит курсоры,
содержимое буферов, RNG и SHA256 входных файлов. Resume переигрывает источники до
курсора (O(consumed)), не обещает O(1) seek. Оптимизатор и training RNG также
сохраняются. Данные не исполняются; плагины/HTTP/tools на этом этапе отсутствуют.

## Границы этапа

Нет обученной языковой компетентности, early exit, memory L0–L5, Think,
verification, speculative decoding, multimodal, continual replay/гарантии против
забывания, GUI/HTTP, architecture search, lazy experts, quantized inference.
Fine-tuning: full/instruction (JSONL prompt/response, loss только на response),
expert-only, router-only и явное замораживание групп. Все имена freeze проверяются.
Исследовательская гипотеза — качество на единицу вычисления может выиграть у dense
архитектуры. Для подтверждения нужны независимые корпуса, seeds, matched-budget
Transformer benchmark и статистические интервалы, а не маленький smoke corpus.

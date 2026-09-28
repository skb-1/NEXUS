# NEXUS-X — отчёт первого этапа

Дата: **2026-09-28**. Среда: Linux x86_64, Python 3.11.2, NumPy 2.4.6,
OpenBLAS; fp32, seed=42, deterministic BLAS threads=1.

Это результат работающего прототипа, а не законченной системы NEXUS-X.
Математика: [architecture.md](architecture.md). Числа:
[stage1_benchmark.json](stage1_benchmark.json),
[stage1_training.json](stage1_training.json).

## 1. Реализованные архитектурные идеи

Причинный local-window + cumulative-summary mixer без Q/K/V и T×T attention;
low-rank shared core; обучаемый top-2 router; sparse dispatch только выбранных
строк в экспертов; byte decoder. Каждый токен выбирает 2 из 4 экспертов.
Router стоит после shared core, пока не выбирает глубину вычисления.

## 2. Стандартные компоненты

UTF-8 byte tokenization, fp32, RMSNorm, tanh, softmax, top-k, low-rank matrices,
residual connections, auxiliary load balancing, masked cross-entropy, AdamW,
cosine LR, gradient clipping/accumulation, NumPy BLAS, SafeTensors, SHA256.
Reverse-mode autodiff реализован в проекте, но сам алгоритм стандартный.

## 3. Новые экспериментальные компоненты

Исследуется **комбинация** mixer/позиционных признаков/sparse experts и её
compute-quality tradeoff. Приоритет научной новизны не заявляется.
NEXUS-TURBO v0.1 — scoped native thread management и BLAS/reference dispatch,
не собственный низкоуровневый CPU runtime. Отдельно работает int8 storage codec;
основная модель им пока не пользуется.

## 4. Параметры

**20 355** trainable параметров при pretraining.
D=32, depth=2, rank=8, vocab=259, context=64, window=8, experts=4, active=2.
60 шагов, batch=4, два источника с весами 1.0/0.4, lr=0.003→0.0003.
Сэмплер выбрал 170 text и 70 code примеров (ожидаемое соотношение вероятностное).

## 5. Активные параметры

Консервативная оценка на токен: **19 331 / 20 355 ≈ 94,97%**.
На всём inference forward union выбранных экспертов: **20 355**.
В tiny модели доминируют общие embeddings/output, поэтому экономия по числу
активных параметров мала. Не следует выдавать top-2 из 4 за снижение полной
стоимости на 50%. Все веса резидентны, это не метрика FLOPs/RAM.

## 6. Размер модели

Чистые fp32 веса: **81 420 bytes ≈ 79,5 KiB**.
Полный сохранённый checkpoint: **340 206 bytes ≈ 332,2 KiB**, включая optimizer,
metadata и sampler. Финальная обученная модель создана в `model.nexus/`,
instruction-tuned — в `finetuned.nexus/`; артефакты исключены из Git.

## 7. RAM

RSS процесса в прогретом generation benchmark: **39,63 MiB**.
В журнале pretraining наблюдалось около **50,0 MiB**. Это snapshots, не peak RAM,
и включает Python/NumPy, а не только веса. Проверка repeated inference не выявила
удержания графов/изменения весов в пределах теста; это не формальная гарантия
отсутствия любых утечек.

## 8. Inference speed

Прогретая авторегрессия: **≈2 008 байтовых токенов/с**, 32 новых токена за
**15,94 ms** на prompt `NEXUS`, greedy. Два прогрева, семь измерений;
приведён реальный запуск с медианным wall time. Контекст пересчитывается целиком,
инкрементального cache нет. Cold load и отдельный turbo microbenchmark сюда
не входят. Нельзя сравнивать эти токены/сек напрямую с BPE-токенами LLM.

## 9. CPU utilization

В медианном прогретом generation run: **≈100,0% одного логического CPU**
(process CPU / wall). Это не процент загрузки всей машины.
В коротких непрогретых вызовах наблюдалось >100% из-за process-wide native/helper
activity. Энергия, аппаратные cache misses и IPC не измерялись.

## 10. Эффект NEXUS-TURBO

Одинаковые веса, вход `[1,64]`, семь paired повторов, чередующийся порядок:

| Full forward | Медиана |
|---|---:|
| NumPy `einsum(optimize=False)` reference | 0,7930 ms |
| NumPy BLAS turbo | 0,6079 ms |
| Отношение reference/turbo | **1,304×** |
| Снижение wall time | **23,34%** |

Максимальная разница logits **1,1921×10⁻⁶**, allclose прошёл.
Это измеренное ускорение относительно **этого reference kernel**, не относительно
Transformer, оптимизированного PyTorch или другой LLM. Это не измерение энергии
и не доказательство универсального ускорения. Сырые samples сохранены; tiny
latencies подвержены шуму хоста.

## 11. Benchmark quality

На одинаковой seeded выборке отдельных validation-данных (744 target tokens,
включая EOS, с повторениями):

| Модель | Cross-entropy, nats/target |
|---|---:|
| До обучения | **5,9986** |
| После 60 шагов | **3,3094** |

Итог: perplexity **27,37**, bits/byte-token **4,7745**.
Train CE первого/последнего разных batch: 5,8992 → 3,1595; это не сопоставление
на фиксированном train batch. Отдельный тест подтверждает снижение loss на
фиксированном batch.

Instruction tuning: 20 шагов, frozen encoder/position/core, response-only loss;
CE **3,8475** на отдельном instruction validation. Эти данные отличаются от
pretraining validation, поэтому сравнивать два CE как улучшение/ухудшение нельзя.
Генерация пока бессвязная и повторяющаяся (в greedy-пробе повторялась буква «н»).

Torch Transformer baseline реализован отдельно, но **не измерен**: попытка
установить CPU torch завершилась TLS-ошибкой `download.pytorch.org`.
Optional baseline test skipped. Никакой победы над Transformer не подтверждено.

## 12. Что требует дальнейшего обучения и разработки

Нужны реальные train/validation/test корпуса, tokenizer research, несколько seeds,
matched-parameter/FLOP/time controls, ablations mixer/router/low-rank, специализация
экспертов. Текущий corpus — инженерный smoke test. Reasoning, языковая
компетентность и перенос знаний не подтверждены.

Early exit, memory, Think, meta-learning, multimodal, GUI, HTTP/plugins,
architecture search и продвинутый CPU runtime ещё не реализованы; интерфейсы
пользователя не притворяются, что эти возможности уже работают.

## 13. Подтверждённые результаты

- Реально выполнены `create`, `finetune`, `use`, `bench`, `status` через `nexus.py`.
- Все три основных режима проверены также через интерактивное меню.
- **48 тестов прошли, 1 skipped** (torch). `compileall` и `pip check` успешны.
- Numerical gradients, causality, sparse dispatch, freeze groups, quantization
  error bound, bounded inference retention, sampler/optimizer serialization
  и численная эквивалентность turbo/reference проверены.
- Непрерывное обучение и resume со всеми состояниями дали побитово одинаковые
  веса и sampler state на этом backend, включая grad accumulation.
- SafeTensors roundtrip сохраняет logits точно; corruption и path traversal
  отклоняются. База fine-tune остаётся неизменной.
- Validation next-byte loss снизился на маленьком демонстрационном наборе.

## 14. Гипотезы, не доказанные результаты

- Такая sparse summary архитектура может давать лучшее качество на единицу CPU
  вычислений, чем dense альтернативы.
- Увеличение количества экспертов может улучшить quality при малом росте compute.
- Иерархическая память и adaptive depth могут повысить эффективность.
- NUMA/cache/precision оптимизации могут дать дополнительную выгоду.

Ни один из этих пунктов не подтверждён представленным smoke-run. Вывод первого
этапа: **обучаемое, воспроизводимое, измеряемое инженерное основание готово**;
гипотеза intelligence per computation остаётся открытой.

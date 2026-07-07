# IduEdu paper — рабочая папка

Исходники статьи и воспроизводимый бенчмарк-набор для IduEdu.

- Статья: [`iduedu_paper_ru.tex`](iduedu_paper_ru.tex) (текст на русском, целевой класс ACM `sigconf`).
- Библиография: [`sample-base.bib`](sample-base.bib).
- Бенчмарки, сырые результаты и ноутбук фигур — в подпапках ниже.

## Структура

```
benchmarks/   скрипты бенчмарков B1–B4 (resume-safe: можно прерывать и перезапускать)
results/      CSV с сырыми измерениями + env_*.json (железо и версии библиотек)
figures/      paper_figures.ipynb -> PNG для статьи
iduedu_paper_ru.tex   текст статьи
sample-base.bib       библиография
.venv-bench/  отдельное окружение бенчмарков (в .gitignore)
pbf_cache/    скачанные PBF-дампы OSM (в .gitignore)
```

## Вклад работы (contributions)

- **C1. UrbanGraph — табличная GeoDataFrame-модель городского графа.** Вершины и
  рёбра хранятся таблицами, смежность — лениво строящийся кэшируемый CSR.
  Направленность задаётся булевой колонкой `oneway` (частично-направленный
  мультиграф без дублирования рёбер). Нулевая конвертация перед численными
  ядрами и нулевая стоимость интеграции с гео-стеком (GeoPandas/Shapely).
- **C2. Статический граф ОТ прямо из OSM без GTFS** — автобус/троллейбус/трамвай/
  метро, восстановление рваной геометрии маршрутов, пересадки, входы/выходы метро
  с учётом глубины.
- **C3. Масштабируемый OD-расчёт как часть pipeline** — Numba-ядра по CSR,
  cutoff-отсечка, адаптивный разворот графа, параллелизм по источникам.
- **C4. Открытый воспроизводимый бенчмарк-набор** (эта папка): сборка графов,
  мультимодальная сборка, OD, память, валидность результатов.

## Окружение

Бенчмаркам нужно отдельное venv (конкуренты OSMnx/Pyrosm/NetworKit/igraph не
входят в основные зависимости IduEdu):

```bash
cd paper
uv venv .venv-bench --python 3.11
uv pip install --python .venv-bench/Scripts/python.exe -r requirements-bench.txt
```

`iduedu` ставится editable из корня репозитория (`-e ../` в `requirements-bench.txt`).
Kernel для ноутбука регистрируется как `iduedu-bench`
(`python -m ipykernel install --user --name iduedu-bench`).

> Pyrosm застоял с 2023 г.: при конфликте с pandas 3.x соберите второе venv с
> `pandas<3` только для его прогонов — resume-safe CSV корректно сшиваются между
> окружениями.

## Бенчмарки

Все скрипты: resume-safe CSV (append + skip по ключу), медианы по ≥3 прогонам,
фиксированные seed'ы, один и тот же AOI на все библиотеки (bbox из заголовка PBF).
Железо и версии библиотек фиксируются в `results/env_*.json`.

| #  | Скрипт                | Что меряем                                                                                                          | Библиотеки                |
|----|-----------------------|-------------------------------------------------------------------------------------------------------------------|---------------------------|
| B1 | `bench_build.py`      | walk+drive графы: время, узлы/рёбра, **размер итогового объекта графа в памяти**; simplify on/off                  | IduEdu, OSMnx, Pyrosm     |
| B2 | `bench_intermodal.py` | стадии walk / PT / join: время + рёбра                                                                             | IduEdu                    |
| B3 | `bench_od.py`         | прямоугольные и квадратные OD; декомпозиция snap/convert/compute; cutoff-абляция                                   | IduEdu, NetworKit, igraph |
| B4 | `bench_validity.py`   | корректность: OD IduEdu ≡ NetworkX (max Δ); структурное сравнение walk-графов IduEdu vs OSMnx (суммарная длина, доля наибольшей компоненты) | IduEdu, NetworkX, OSMnx   |

Память меряется внутри B1 как размер уже построенного представления графа
(`representation_size_mb`): `UrbanGraph` для IduEdu и `networkx.MultiDiGraph`
для OSMnx/Pyrosm.

Города для B1/B2 (пресеты pyrosm, от малого к большому):
Helsinki, Saint Petersburg, Moscow, London, Seoul, New York.
Для B3 и B4 — мультимодальный граф Санкт-Петербурга.

### Полные прогоны

```bash
cd benchmarks
../.venv-bench/Scripts/python.exe run_all.py
```

`run_all.py` гоняет все шаги подряд (build → intermodal → od rect → od square →
validity), логирует каждый в `results/logs/` и продолжает при падении отдельного
шага. Точечные запуски:

```bash
PY=../.venv-bench/Scripts/python.exe
$PY bench_build.py                  # B1: часы; качает PBF (сотни МБ) + Overpass
$PY bench_intermodal.py             # B2: часы (Overpass, большие города)
$PY bench_od.py --mode rect         # B3a: строит граф СПб один раз, потом ~час
$PY bench_od.py --mode square       # B3b: самый долгий (большие |O| без отсечки)
$PY bench_validity.py               # B4: минуты
```

У каждого скрипта есть `--smoke` (маленькая территория, быстрая проверка
харнеса) и `--areas` / `--libraries` для частичных запусков. Результаты пишутся
построчно: прерывание безопасно, повторный запуск продолжит с места остановки.
`bench_build.py` дополнительно пропускает полностью измеренные города **до**
скачивания PBF.

## Фигуры

```bash
cd figures
../.venv-bench/Scripts/python.exe -m jupyter nbconvert --to notebook --execute paper_figures.ipynb --output paper_figures.ipynb
```

PNG кладутся в `figures/`; статья ссылается на `figures/...`.

## Протокол  сравнения OD (B3)

OD-сравнение — только **IduEdu vs NetworKit vs igraph**. Конкуренты **строятся из
`networkx.MultiDiGraph`** (стандартное представление экосистемы osmnx/networkx),
собранного из того же `UrbanGraph` — идентичные топология и веса.

- **IduEdu** принимает свой `UrbanGraph`; `od_matrix` снапит объекты внутренне
  через R-tree GeoDataFrame → `convert=0`, `snap` входит в общее время,
  `time_total = od`.
- **Конкуренты** из networkx платят реальные издержки: `convert` (nx→формат либы,
  схлопывание параллельных рёбер по минимуму) + `snap` (свой `cKDTree` по
  координатам вершин). Замеряется раздельно и суммарно
  (`time_total = snap + convert + od`).

NetworkX как маршрутизатор в OD не участвует (нет пакетного OD-API, на порядки
медленнее) — только как общее исходное представление для конкурентов.

## Ключевые наблюдения (отражены в статье)

- **Сборка (B1, walk, simplify=True):** IduEdu быстрее OSMnx в ~3.3–6× (медиана
  ~5.7×), при этом `UrbanGraph` ~в 10–11× компактнее по памяти, чем
  `MultiDiGraph` OSMnx, и ~в 90× — чем сырой граф Pyrosm.
- **Инверсия абляции simplify (B1):** для пешеходной сети упрощение теперь
  *замедляет* сборку (плотные пересечения делают геометрическое объединение
  дорогим), но сокращает число рёбер в ~2.5–3×; для автомобильной сети упрощение
  и ускоряет сборку, и сокращает рёбра. Вывод — топологически зависимый trade-off,
  а не безусловное ускорение.
- **OD (B3):** при `|O| > |D|` разворот графа стабилизирует время IduEdu, тогда
  как NetworKit/igraph растут линейно; cutoff даёт до ~17× ускорения на крупных
  квадратных матрицах (`|O|=32 768`, τ=5 мин).
- **Валидность (B4):** множества достижимости OD IduEdu ≡ NetworkX,
  max |Δ| ≈ 2·10⁻⁴ мин (округление float32 vs float64); суммарная длина walk-сети
  IduEdu и OSMnx совпадает с точностью ~1%.


## Железо и версии

Все измерения — на рабочей станции Intel Core i7-12700F, 64 GB RAM, Windows 10/11,
Python 3.11. Точные версии всех библиотек для каждого прогона — в
`results/env_*.json`.

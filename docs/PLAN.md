# HRE Implementation Plan

Goal: локальный импорт, анализ, ручная библиотека, точная сборка и диагностика.
Architecture: чистые CV-функции + SQLite store + composer + Flask UI/CLI.
Tech Stack: Python 3.13, OpenCV, SciPy, scikit-image, Pillow, NumPy, Flask.
Spec: DESIGN.md и supplied-task.txt (полученный текст обрывается).

## Global constraints
Оригиналы неизменяемы; SHA-256; язык независим; автоматика создает кандидатов;
нет подмены текста/шрифтов; provenance обязателен; фикстуры только synthetic.

## Review focus
EXIF и Unicode-пути; небезопасные ZIP; бледные чернила/сетка; пробелы покрытия;
переполнение/переанализ после разметки; локальные URL и пользовательский HTML.

## Последовательность исполнения
- [x] tests/test_engine.py: сначала импорт/архив/верификация/покрытие/геометрия,
  затем CV-синтетика, API и регрессии. Запуск pytest до реализации: fail.
- [x] hre/imaging.py: extract_ink, segment, metrics, perspective, mesh_warp,
  composite_linear; независимые функции над RGB/alpha.
- [x] hre/store.py: Store.import_file/import_zip/analyze/add_sample/update_sample,
  проверка входов и audit, immutable original + derived version.
- [x] hre/compose.py: plan_text и render, проверяемое покрытие и источники.
- [x] hre/web.py + static: локальный редактор выделений/полигонов/анкеров,
  текст/статусы/статистика/композиция и экспорт.
- [x] hre/__main__.py: CLI import/analyze/serve/render/stats/demo/doctor.
- [x] Фикстуры, lock-файл, README, исследование и отчет о проверках.
- [x] pytest, воспроизводимая demo, HTTP smoke и визуальное чтение результатов.

Исполнение инженером в текущей сессии по автономному заданию пользователя.

# Форматы и координаты

SQLite: sources(id,data JSON), samples(id,data JSON), events(seq,sample_id,data JSON).
Схема экспорта: `hre-library-v1`; текущий формат расширяет поля аддитивно.

Источник: id, source_id/root_source_id, view_id, original_name/original_path,
original, sha256, bytes, original_dimensions, width/height, image_type,
exif_orientation, icc_profile, color_space, transforms, imported_at, synthetic,
page/page_sha256, perceptual_hash, quality_metrics, detected_document_regions,
original_to_normalized/normalized_to_original, coordinate_spaces, similar_sources.

`id` исходного представления равен SHA-256, id выровненного содержит `-view-`.
`source_id`/`root_source_id` при этом остаются идентификатором оригинала.
Близкий pHash — только кандидат дубликата; файлы не объединяются и не удаляются.
Повторный SHA объединяется; независимые источники в inventory считаются по SHA.

Sample: id, source_id (id представления), root_source_id, language ru/en, kind,
text, bbox, polygon, coordinate_space, original_polygon, baseline, entry/exit,
revision, status/review_status, reviewer, origin/origin_type, evidence/evidence_sha256,
alpha/alpha_sha256, ink_color_srgb, metrics, timestamps. `origin` означает manual/
automatic разметку; `origin_type` — real/synthetic по происхождению источника.

| Рабочий status | review_status спецификации |
|---|---|
| candidate | unlabelled |
| transcribed | proposed |
| verified | manually_verified |
| rejected | rejected |

Все bbox half-open: `[x0,y0,x1,y1)`, не `[x,y,width,height]`. Полигон — список
глобальных точек, не повторяет первую точку в конце. Baseline — локальный y,
anchors — локальные [x,y]. Координаты непрерывные по границам пикселей; центр
пикселя (i,j) имеет координаты (i+.5,j+.5). EXIF-матрицы покрывают ориентации 1–8.
Для OpenCV homography сохраняются и матрица центров, и матрица границ пикселей.

`original_image → normalized_image`: матрица EXIF, затем H каждого выравнивания.
Обратная матрица возвращает полигон кропа на оригинальную фотографию.
`page_local` — нормализованная выровненная view с началом в левом верхнем углу.
`output_canvas` — независимые пиксели PNG; каждый placement хранит перевод x,y.
`physical_page_mm` пока отсутствует: нет калибровки размера клетки/линейки в мм.
Нелинейное mesh поле поддерживается только низкоуровневой функцией, не сохраняется
как аннотируемая view; это требует отдельной версии формата.

Текст хранится без автоматической Unicode-нормализации. Индексы — code points,
а не UTF-16 offsets. Латиница и кириллица не взаимозаменяются. Character имеет один
codepoint; комбинируемые последовательности пока следует хранить fragment.

Итоговый JSON: text/language/mode, tokens с start/end, placements, размеры,
настройки, версия движка, text/image SHA-256, признак синтетики и notice.
PNG без JSON не является полным воспроизводимым результатом.

# ijepa_spectra2d

Fork del I-JEPA original adaptado para imágenes GADF (Gramian Angular Difference Field)
generadas a partir de espectros NIR de suelo.

## Objetivo
Adaptar el pipeline de pretraining de I-JEPA para usar imágenes 2D GADF de 224×224 (1 canal)
en lugar de imágenes RGB de ImageNet.

## Dataset fuente (v2 — dataset activo)
- Pre-entrenamiento: /mnt/homeGPU/igarzon/Meta-Learning/SpectraI-JEPA/data/SoilDataset_NIR_agg
  - CSV sin columna de índice, 1051 longitudes de onda (400–2499 nm)
- Tareas downstream: /mnt/homeGPU/igarzon/Meta-Learning/SpectraI-JEPA/data/SoilDataset_NIR
  - 192 tareas de regresión (suelo, propiedades físico-químicas)
  - fixed_val_* contienen índices enteros (support_idx, query_idx), NO espectros
  - splits.csv asigna 185 de las 192 tareas a train/val/test; las 6 tareas de Malta
    (Soil_Malta-*) tienen solo 3 instancias cada una y no aparecen en splits.csv

## Dataset pre-computado (HDF5) — archivos activos
Las imágenes GADF se generan UNA SOLA VEZ con scripts/precompute_gadf.py (orquestado por precompute_gadf2d.sh).

Archivos activos (v2):
  data/gadf_224_v2.h5            — pre-entrenamiento, shape (N, 1, 224, 224), dtype float16
  data/gadf_224_v2_diagonal.h5   — ídem con magnitud espectral en diagonal
  data/gadf_norm_stats_v2.json   — media/std para normalización
  data/gadf_paa_global_stats_v2.json — stats PAA global (solo modo diagonal)
  data/SoilDataset_NIR_gadf2d/   — tareas downstream GADF 2D (192 tareas)
  data/SoilDataset_NIR_gadf2d_diagonal/ — ídem con diagonal

Archivos con Savitzky-Golay (opcionales, generados con --savgol):
  data/gadf_224_v2_savgol.h5                  — pretrain, SG(window=15, poly=2, deriv=0)
  data/gadf_224_v2_diagonal_savgol.h5         — pretrain, diagonal + SG
  data/SoilDataset_NIR_gadf2d_savgol/         — tareas downstream con SG
  data/SoilDataset_NIR_gadf2d_diagonal_savgol/ — ídem con diagonal + SG

Archivos antiguos (v1, NO usar para experimentos nuevos):
  data/gadf_224.h5, data/gadf_224_diagonal.h5, data/Soil_NIR_AGG_mixed_gadf2d/

GADFDataset soporta dos modos mediante el flag precomputed=True/False:
- precomputed=True (producción): carga desde el HDF5, __getitem__ es solo lectura de disco
- precomputed=False (desarrollo/debug): computa GADF on-the-fly con pyts, con SG opcional

Para regenerar todo el pipeline GADF:
  bash precompute_gadf2d.sh                    # sin diagonal, sin SG
  bash precompute_gadf2d.sh --diagonal         # con diagonal
  bash precompute_gadf2d.sh --savgol           # con Savitzky-Golay (window=15, poly=2, deriv=0)
  bash precompute_gadf2d.sh --diagonal --savgol  # diagonal + SG

IMPORTANTE (consistencia pretrain ↔ downstream): si el pretraining usa un HDF5 _savgol,
el finetune debe usar --savgol (modo on-the-fly) o apuntar al directorio _savgol precomputado.

## Cambios respecto al I-JEPA original (en orden de implementación)
1. configs/gadf_soil_nir.yaml        — config específico para GADF
2. scripts/compute_gadf_stats.py     — calcula media/std del dataset para normalización
3. scripts/precompute_gadf.py        — genera data/gadf_224.h5
4. src/datasets/gadf_dataset.py      — dataset class con modos precomputed y on-the-fly
5. src/transforms.py                 — eliminar augmentaciones RGB, añadir normalización grayscale
6. src/helper.py                     — propagar in_chans=1 al modelo
7. src/masks/multiblock.py           — enmascarar bloques simétricos (GADF es matriz simétrica)
8. src/train.py                      — conectar todo
9. main_finetune_gadf2d.py           — fine-tuning downstream (espectros → GADF 2D → ViT)
10. slurm_finetune_gadf2d.sh         — SLURM job para fine-tuning (full FT + linear probing + baseline)

## Fine-tuning downstream
main_finetune_gadf2d.py carga el target_encoder del checkpoint de pretraining y añade una
cabeza de regresión (LayerNorm + Linear).  Los espectros NIR de las tareas downstream
(X_supp.csv / X_query.csv) se convierten a imágenes GADF 2D on-the-fly con pyts.

Datos downstream: ../SpectraI-JEPA/data/SoilDataset_NIR/ (192 tareas)
  GADF pre-computado: data/SoilDataset_NIR_gadf2d/

Modos:
- Full fine-tuning (default): actualiza encoder + head
- Linear probing (--linear_probing): congela encoder, solo head
- No pretrain (baseline): pesos aleatorios, sin checkpoint

## Online Linear Probing durante pretraining
src/train.py incluye LinearProbeEvaluator que evalúa la calidad de las representaciones
cada N épocas usando tareas del split 'val' de splits.csv.

Configuración en configs/gadf_soil_nir.yaml (sección `probe`):
- freq: cada cuántas épocas evaluar (0 = desactivado, default: 25)
- data_path: ruta a datos downstream (../SpectraI-JEPA/data/SoilDataset_NIR)
- k_spt / k_qry: muestras de soporte/query por tarea (default: 25)
- epochs: épocas del linear probe por tarea (default: 50)

Flujo: carga tareas val → espectros → GADF 2D → features con target_encoder congelado →
entrena nn.Linear(D,1) con Adam → reporta R² y RMSE medios.

## Wandb
Integrado en el pretraining (sección `wandb` del YAML):
- enable: true/false
- project: nombre del proyecto W&B (default: ijepa-gadf2d)
- name: nombre del run (auto-generado si null)

Métricas logueadas: train/loss, train/lr, train/wd, train/mask_a, train/mask_b,
train/time_ms, y probe/* cuando toca evaluación.

## Alternativas de backbone pendientes de explorar
El archivo `backbone_alternatives.md` (raíz del repo) contiene un análisis
detallado de por qué la configuración actual (I-JEPA + ViT-base + ProtoNets)
se acerca pero no supera al baseline ResNet18 + ProtoNets del artículo
anterior (R²=0.49), y propone alternativas concretas a probar: ViTs más
pequeños (vit_tiny, vit_small, vit_1d_aligned), patch_size=8, hybrid stem
(conv patchifier) y plan de validación priorizado. Consultar ese archivo
antes de proponer nuevos experimentos sobre arquitectura/hiperparámetros
de la backbone.

## Modos de evaluación downstream (rama eval-kfold-cv)

Los scripts de evaluación offline (`main_finetune_gadf2d.py`, `main_protonet_gadf2d.py`,
`main_eval_joint_protonet.py`) soportan dos modos mediante `--eval_mode`:

- `fixed` (default): comportamiento legacy — usa `fixed_val_support_{k_spt}shots.csv` y
  `fixed_val_query_{k_qry}shots.csv` para fijar el split support/query. Completamente
  reproducible. Necesario para comparar con resultados anteriores.

- `cv`: K-fold cross-validation sobre el pool unificado `X_supp ∪ X_query`.
  K = ⌈N/k_spt⌉ folds disjuntos; cada instancia es support exactamente una vez.
  El `StandardScaler` se re-ajusta por fold (sólo sobre el support de ese fold).
  Más robusto: elimina el sesgo de una partición arbitraria. Resultados por tarea:
  media ± std de R²/RMSE a través de los K folds.

Los evaluadores online durante el pretraining (`LinearProbeEvaluator` en src/train.py,
`ProtoNetEvaluator` en src/train_joint.py) NO usan este flag — mantienen muestreo fijo
con `RandomState(42)` por motivos de coste computacional.

## Registro de experimentos (experiments.log)
Cada experimento lanzado con `./launch.sh <slurm_script.sh>` queda registrado en experiments.log
con el formato:
  YYYY-MM-DD HH:MM | job=<JOB_ID> | script=<script.sh> | <objetivo del experimento>

El usuario escribe una línea de explicación antes de cada lanzamiento describiendo el motivo
del experimento. Si el usuario pregunta por un experimento concreto, un job ID, o los resultados
de un run, consulta primero experiments.log para obtener contexto sobre qué se quería probar.

## Reglas
- NO modificar la arquitectura ViT (vision_transformer.py) — 224×224 cuadrado funciona sin cambios
- NO modificar masks/utils.py, masks/random.py, utils/*
- NO tocar main_distributed.py todavía
- Mantener compatibilidad con el resto de configs YAML existentes
- Conda env: /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
- SIEMPRE que se hable de modificar o lanzar un script SLURM, verificar primero que las rutas
  de los datasets referenciadas en ese script (y en el config YAML asociado) apuntan a las
  versiones activas correctas (v2). Consultar la sección "Dataset pre-computado (HDF5)" para
  confirmar qué archivos son los activos y cuáles son obsoletos (v1).
- SIEMPRE que se cree un nuevo script SLURM, verificar que el directorio referenciado en
  `--output` y `--error` existe antes de que el job se lance. SLURM abre esos ficheros
  antes de ejecutar el script, por lo que el `mkdir -p` interno llega tarde. Si el
  directorio no existe, crearlo con `mkdir -p <ruta>` inmediatamente después de escribir
  el script.

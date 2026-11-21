# TP4 – Redes convolucionales para clasificar lesiones de piel (ISIC)

Trabajo práctico de Redes Neuronales en Bioingeniería (ITBA). Parte de un problema de clasificación de **imágenes dermatoscópicas** del dataset ISIC (9 clases, ~880 imágenes) y estudia cómo una **red convolucional** aprovecha la estructura espacial de la imagen frente a una MLP que trata cada píxel como una variable independiente. Se limpia el dataset (duplicados y split propio), se compara MLP contra CNN, se estudian las decisiones de arquitectura (profundidad, kernel, filtros y resolución de entrada), se evalúa la data augmentation hecha a mano y se cierra con una evaluación única en test y una interpretación de lo que aprende la red.


## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `CNN_en_Skin_Dataset_Colab.ipynb` | Notebook con todo el trabajo (código, experimentos, análisis y conclusiones), ejecutado de punta a punta. |
| `requirements.txt` | Dependencias de Python. |
| `README.md` | Este archivo. |
| `.gitignore` | Excluye el dataset, modelos entrenados, entornos virtuales y archivos temporales. |
| `resultados/` | Resultados de la corrida entregada: tablas en CSV (EDA, duplicados, split, normalización, MLP vs CNN, arquitectura, augmentation, evaluación final, curvas por época, predicciones de test y errores de validación), `resumen_corrida.json`, `salida_consola.txt` y las figuras en `resultados/figuras/`. |

El dataset **no está incluido** en el repositorio. Debe quedar en `data/Split_smol/`, con `train/` y `val/` y una subcarpeta por clase. Tampoco se versionan modelos entrenados.

## Entorno de ejecución

- El notebook se ejecutó en **Google Colab con una GPU NVIDIA T4** (`torch 2.14.1+cu130`).
- La corrida completa, desde la celda de captura hasta el final, tomó **unos 4,3 minutos** en esa GPU. En CPU tarda unos 15-20 minutos.
- Todos los tiempos reportados (segundos por época, tiempo total de entrenamiento) se midieron en esa GPU; en otro hardware cambian los valores absolutos, no las comparaciones relativas.
- Los datos viven en memoria como tensores `uint8` y se normalizan por lote. Funciona también en CPU, pero es más lento.

## Cómo ejecutarlo

### Opción A: Google Colab

1. Subir `CNN_en_Skin_Dataset_Colab.ipynb` a Colab (o abrirlo desde Drive/GitHub).
2. Activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
3. En la celda de "Paso previo", elegir la fuente del dataset (`FUENTE`) y dejarlo en `data/Split_smol`, que es la ruta que usa `DATA_DIR`.
4. Ejecutar todas las celdas en orden. Las librerías (`torch`, `scikit-learn`, `pandas`, `matplotlib`, `Pillow`) ya vienen instaladas en Colab, así que no hace falta `pip install`.

### Opción B: ejecución local

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# Instalar PyTorch según el sistema/CUDA (https://pytorch.org/get-started/locally/)
pip install -r requirements.txt

jupyter notebook CNN_en_Skin_Dataset_Colab.ipynb
```

Ejecutar el notebook desde la carpeta que contiene `data/`.

### Notas

- Las constantes de la sección 0 se usaron **sin modificaciones**: `SIZES = [32, 64, 96]`, `SEEDS = [0, 1, 2]`, `MAX_EPOCHS = 60`, `PATIENCE = 10`, `BATCH = 32`, `AUG_COPIES = 2`, `SEED_SPLIT = 42` y `SEEDS_FINAL = [0, 1, 2, 3, 4]`. La semilla del split no debe cambiarse entre corridas.
- Las semillas están fijadas, pero los resultados **varían entre ejecuciones**, probablemente por el no determinismo de la GPU del Colab. la misma configuración (CNN base, mismas semillas) dio val_acc 0,641, 0,625 y 0,657 en tres secciones del notebook. Si se vuelve a ejecutar, hay que revisar los textos del análisis.


## Contenido del notebook

| Sección | Qué hace |
|---|---|
| **Paso previo (Colab)** | Verifica la GPU, deja el dataset en `data/Split_smol` y activa la captura de figuras (`resultados/figuras/`) y de la consola (`resultados/salida_consola.txt`). |
| **0. Configuración** | Importaciones, constantes (resoluciones, semillas, épocas, paciencia, batch, copias de augmentation) y detección de dispositivo. |
| **1. Análisis exploratorio** | Imágenes por clase, resoluciones y relación de aspecto, color medio por clase y búsqueda de duplicados exactos (MD5) y casi-duplicados (hash perceptual). Observaciones propias de EDA. |
| **2. Duplicados y split** | Elimina una copia de cada grupo duplicado y vuelve a dividir en train/val/test (60/20/20) de forma estratificada y con semilla fija. Calcula media y desvío por canal **solo con train**. |
| **3. Data augmentation a mano** | Flips, rotaciones de 90° y brillo/contraste con operaciones de tensores (sin librerías), aplicadas solo a train y después del split. |
| **4. Modelos y entrenamiento** | `MLP` de referencia y `CNN` con bloques `Conv2d → BatchNorm → ReLU → MaxPool(2)` y cabeza `Linear`; cálculo del campo receptivo (`rf_full`); entrenamiento con Adam y early stopping. Tabla capa por capa (formas y parámetros), explicación del código convolucional y chequeo rápido de que el modelo aprende. |
| **5. De la MLP a la CNN** | Comparación a igual resolución en parámetros, val_acc, brecha, tiempo por época y robustez a traslaciones (3 semillas). **Análisis 1.** |
| **6. Decisiones de arquitectura** | Un factor a la vez respecto de la CNN base: profundidad, kernel, filtros y resolución de entrada, con campo receptivo teórico (3 semillas). **Análisis 2.** |
| **7. Data augmentation** | Cuatro niveles (ninguna, flips, flips + rot90, flips + rot90 + color) con la CNN base (3 semillas). Demostración de overfitting con números. **Análisis 3.** |
| **8. Evaluación final en test** | Elección greedy de la configuración final con validación y evaluación **única** en test de la CNN final y la MLP de 64 px (5 semillas). Matriz de confusión y reporte por clase. **Análisis 4.** |
| **9. ¿Qué aprende la red?** | Filtros de la primera convolución, mapas de activación de la primera y la última capa convolucional, y galería de imágenes de validación mal clasificadas. |
| **Declaración de uso de IA** | Detalle de qué herramientas de IA se usaron y para qué. |

## Resultados principales

**Datos.** El dataset original (878 imágenes) traía **34 grupos de imágenes idénticas** (22 en Atopic Dermatitis y 12 en Tinea), 33 de ellos repetidos entre `train/` y `val/`. Se eliminaron y quedaron 844 imágenes: train 506, val 169, test 169.

**MLP contra CNN** (validación, media ± desvío de 3 semillas):

| Modelo | Parámetros | val_acc | Brecha train − val | s/época |
|---|---|---|---|---|
| MLP 32 px | 1.640.201 | 0,495 ± 0,039 | 0,243 | 0,037 |
| MLP 64 px | 6.358.793 | 0,525 ± 0,043 | 0,180 | 0,057 |
| CNN 32 px | 156.057 | 0,673 ± 0,028 | 0,146 | 0,062 |
| CNN 64 px | 549.273 | 0,641 ± 0,015 | 0,131 | 0,087 |

**Evaluación en el test** (media ± desvío entre 5 semillas; error estándar binomial del test ≈ 0,037):

| Modelo | Parámetros | Test acc | Macro-F1 |
|---|---|---|---|
| CNN final (4 bloques, 64 px, sin augmentation) | 361.113 | **0,641 ± 0,053** | **0,640 ± 0,059** |
| MLP 64 px | 6.358.793 | 0,504 ± 0,053 | 0,487 ± 0,057 |

- La **CNN supera a la MLP** en validación (+0,12 a +0,18) y en test (+0,14), con unas 10 a 18 veces menos parámetros. La diferencia es mayor que el ruido entre semillas y que el error estándar. Con la semilla 0 (la usada para la matriz de confusión) la diferencia se reduce a 0,592 contra 0,568 y no es distinguible.
- **Arquitectura:** ningún factor mejora con claridad más allá del ruido. 4 bloques y 8 filtros dan +0,04, y los kernels grandes y los 96 px quedan algo por debajo. Los parámetros se concentran en la `Linear` de la cabeza (95 % en la CNN base), y no en las convoluciones.
- **Data augmentation:** ningún nivel mejoró la val_acc ni redujo la brecha; cada época cuesta unas 2,7 veces más. Se eligió "ninguna".
- **Robustez a traslaciones:** la caída es parecida en MLP y CNN (0,06 a 0,09), así que los datos no respaldan que la CNN sea más robusta.
- **Por clase:** mejor desempeño en Vascular lesion, Atopic Dermatitis y Benign keratosis; peor en Actinic keratosis y Squamous cell carcinoma, que se confunden entre sí, y en nevus contra melanoma. El recall de melanoma fue 0,55.
- Más detalle (EDA, explicación del código, análisis 1 a 4, sesgos y limitaciones) en las celdas de análisis del notebook.

## Limitaciones

- Dataset chico (844 imágenes), una sola partición y val/test de 169 imágenes, por lo que las diferencias menores a unos 0,04 no son distinguibles del ruido.
- La resolución, el aspecto y el color casi identifican a algunas clases (posible atajo por fuente de imagen).
- Tinea tiene 70 de 76 imágenes con prefijo `aug_` (augmentations pre-hechas, 58 nombres base distintos) que podrían generar fugas entre splits.
- Clases artificialmente balanceadas e imágenes reducidas a 64 px y a cuadrado: los números no se transfieren a un uso clínico.